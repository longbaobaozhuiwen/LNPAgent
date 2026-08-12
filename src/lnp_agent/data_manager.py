"""数据与模型预加载管理器 (v1.5 - 含 benchmark 序列化、批量预测、融合特征)。"""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataManager:
    """预加载 Core 数据和模型，供 domain tools 共享使用。

    v1.3 新增:
    - Benchmark 序列化到磁盘 (三级缓存)
    - batch_predict_formulations(): 向量化批量预测

    v1.5 新增:
    - _build_features(): 支持 design_agile_morgan_fusion 融合特征集
    """

    def __init__(self, source_of_truth: Path | None = None):
        if source_of_truth is None:
            from lnp_agent.paths import SOURCE_OF_TRUTH
            source_of_truth = SOURCE_OF_TRUTH
        self.source_of_truth = Path(source_of_truth)
        self._df: pd.DataFrame | None = None
        self._clean_df: pd.DataFrame | None = None
        self._cleaning_log: pd.DataFrame | None = None
        self._split_df: pd.DataFrame | None = None
        self._model_bench: pd.DataFrame | None = None

        # v1.2: 内存缓存
        self._trained_models: dict[tuple, dict] = {}
        self._oof_residuals: dict[tuple, list[float]] = {}

        # v1.2: 磁盘缓存目录
        from lnp_agent.paths import RESULTS_DIR
        self._cache_dir = RESULTS_DIR / "checkpoints"

    @property
    def df(self) -> pd.DataFrame:
        """清洗后的数据 (100 行，含 template_key 和 design_cell_key)。"""
        if self._df is None:
            from lnp_core.data_cleaning import load_and_clean_v5_3
            df, _ = load_and_clean_v5_3(self.source_of_truth)
            self._df = df
            logger.info(f"Loaded and cleaned {len(self._df)} rows from {self.source_of_truth}")
        return self._df

    @property
    def split_df(self) -> pd.DataFrame:
        """Leave-template-out 5-fold 划分。"""
        if self._split_df is None:
            from lnp_core.data_contract import build_split_matrix
            self._split_df = build_split_matrix(self.df)
        return self._split_df

    def get_model_benchmark(self) -> pd.DataFrame:
        """运行 18 个 model configs 并缓存结果 (三级缓存)。"""
        # 1. 内存缓存
        if self._model_bench is not None:
            return self._model_bench

        # 2. 磁盘缓存 (v1.3 新增)
        cache_path = self._cache_dir / "benchmark" / "leave_template_out_v53.joblib"
        if cache_path.exists():
            self._model_bench = joblib.load(cache_path)
            logger.info(f"Benchmark loaded from {cache_path}")
            return self._model_bench

        # 3. 计算 (~250s)
        from lnp_core.model_evaluation import run_leave_template_out_evaluation
        logger.info("Running model benchmark (18 configs)...")
        self._model_bench = run_leave_template_out_evaluation(self.df, self.split_df)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model_bench, cache_path)
        logger.info(f"Benchmark saved to {cache_path}")
        return self._model_bench

    def get_smiles_lookup(self) -> dict[str, dict[str, str]]:
        """获取脂质名称到 SMILES 的映射。"""
        df = self.df
        lookup: dict[str, dict[str, str]] = {}
        for lipid_col in ["lipid1", "lipid2", "lipid3", "lipid4"]:
            smiles_col = f"{lipid_col}_smiles"
            if lipid_col in df.columns and smiles_col in df.columns:
                mapping = (
                    df[[lipid_col, smiles_col]]
                    .drop_duplicates()
                    .set_index(lipid_col)[smiles_col]
                    .to_dict()
                )
                lookup[lipid_col] = mapping
        return lookup

    # --- v1.5 融合特征支持 ---

    def _build_features(self, df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
        """构建特征集，支持融合特征集。

        v1.5 新增: 当 feature_set_name 为 design_agile_morgan_fusion 时，
        使用 fusion_features 模块，否则委托给 lnp_core。
        """
        if feature_set_name == "design_agile_morgan_fusion":
            from lnp_agent.fusion_features import design_agile_morgan_fusion
            return design_agile_morgan_fusion(df)
        from lnp_core.model_evaluation import build_feature_set_v5_3
        return build_feature_set_v5_3(df, feature_set_name)

    # --- v1.2 磁盘序列化辅助方法 ---

    def _get_model_cache_path(self, key: tuple) -> Path:
        endpoint, model_name, feature_set = key
        return self._cache_dir / "models" / f"{endpoint}_{model_name}_{feature_set}.joblib"

    def _get_residuals_cache_path(self, key: tuple) -> Path:
        endpoint, model_name, feature_set = key
        return self._cache_dir / "residuals" / f"{endpoint}_{model_name}_{feature_set}.pkl"

    def _save_model_to_disk(self, key: tuple, entry: dict) -> None:
        path = self._get_model_cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(entry, path)
        logger.info(f"Model saved to {path}")

    def _load_model_from_disk(self, key: tuple) -> dict | None:
        path = self._get_model_cache_path(key)
        if path.exists():
            entry = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return entry
        return None

    def _save_residuals_to_disk(self, key: tuple, residuals: list[float]) -> None:
        path = self._get_residuals_cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(residuals, f)
        logger.info(f"Residuals saved to {path}")

    def _load_residuals_from_disk(self, key: tuple) -> list[float] | None:
        path = self._get_residuals_cache_path(key)
        if path.exists():
            with open(path, "rb") as f:
                residuals = pickle.load(f)
            logger.info(f"Residuals loaded from {path} ({len(residuals)} values)")
            return residuals
        return None

    def clear_cache(self) -> None:
        """清除所有内存和磁盘缓存。"""
        self._trained_models.clear()
        self._oof_residuals.clear()
        self._model_bench = None
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        logger.info("All caches cleared.")

    # --- v1.1 方法 (保持) ---

    def get_best_config(self, endpoint: str) -> tuple[str, str]:
        """获取某 endpoint 的最佳 (model_name, feature_set_name)。

        基于 leave-template-out benchmark 中 spearman 最高的配置。
        """
        bench = self.get_model_benchmark()
        ep_bench = bench[(bench["endpoint"] == endpoint) & (~bench["invalid_for_selection"])]
        if ep_bench.empty:
            return ("ridge", "design_one_hot")
        best = ep_bench.loc[ep_bench["mean_spearman"].idxmax()]
        return (best["model_name"], best["feature_set"])

    def get_or_train_model(self, endpoint: str, model_name: str,
                           feature_set_name: str) -> dict:
        """获取或训练模型，返回 {model, scaler, feature_columns}。

        三级缓存: 内存 -> 磁盘 -> 训练。
        """
        key = (endpoint, model_name, feature_set_name)

        # 1. 内存缓存
        if key in self._trained_models:
            return self._trained_models[key]

        # 2. 磁盘缓存 (v1.2 新增)
        loaded = self._load_model_from_disk(key)
        if loaded is not None:
            self._trained_models[key] = loaded
            return loaded

        # 3. 训练
        from lnp_core.model_evaluation import V53_MODEL_CONFIGS
        from sklearn.preprocessing import StandardScaler

        config = V53_MODEL_CONFIGS[model_name]
        features = self._build_features(self.df, feature_set_name)
        X_all = features.values
        y_all = self.df[endpoint].values.astype(float)

        scaler = None
        if config["needs_scaling"]:
            scaler = StandardScaler()
            X_all = scaler.fit_transform(X_all)

        model = config["model_class"](**config["fixed_params"])
        model.fit(X_all, y_all)

        entry = {
            "model": model,
            "scaler": scaler,
            "feature_columns": features.columns.tolist(),
        }
        self._trained_models[key] = entry
        self._save_model_to_disk(key, entry)  # v1.2: 保存到磁盘
        logger.info(f"Trained and cached model: {key}")
        return entry

    def get_oof_residuals(self, endpoint: str, model_name: str,
                          feature_set_name: str) -> list[float]:
        """计算 leave-template-out OOF 残差 (用于 conformal interval)。

        三级缓存: 内存 -> 磁盘 -> 计算。
        """
        key = (endpoint, model_name, feature_set_name)

        # 1. 内存缓存
        if key in self._oof_residuals:
            return self._oof_residuals[key]

        # 2. 磁盘缓存 (v1.2 新增)
        loaded = self._load_residuals_from_disk(key)
        if loaded is not None:
            self._oof_residuals[key] = loaded
            return loaded

        # 3. 计算
        from lnp_core.model_evaluation import V53_MODEL_CONFIGS
        from sklearn.preprocessing import StandardScaler

        config = V53_MODEL_CONFIGS[model_name]
        features = self._build_features(self.df, feature_set_name)
        split_df = self.split_df

        residuals: list[float] = []
        for fold_id in sorted(split_df["fold_id"].unique()):
            fold_split = split_df[split_df["fold_id"] == fold_id]
            train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
            test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values
            if len(test_idx) == 0:
                continue

            X_tr = features.loc[train_idx].values
            X_te = features.loc[test_idx].values
            y_tr = self.df.loc[train_idx, endpoint].values.astype(float)
            y_te = self.df.loc[test_idx, endpoint].values.astype(float)

            if config["needs_scaling"]:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            mdl = config["model_class"](**config["fixed_params"])
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            residuals.extend(np.abs(y_te - y_pred).tolist())

        self._oof_residuals[key] = residuals
        self._save_residuals_to_disk(key, residuals)  # v1.2: 保存到磁盘
        logger.info(f"Computed OOF residuals for {key}: {len(residuals)} values")
        return residuals

    def predict_formulation(self, formulation_row: pd.DataFrame,
                            endpoint: str) -> dict:
        """对单个配方预测某 endpoint 的性能。

        Args:
            formulation_row: 单行 DataFrame，包含 lipid1-4, ratio1-4, np_ratio, aq_org_ratio, *_smiles
            endpoint: "immune_signal_a", "immune_signal_b", 或 "tx_log1p"

        Returns:
            {"y_hat": float, "pi_low": float, "pi_high": float,
             "interval_width": float, "model_name": str, "feature_set": str}
        """
        model_name, feature_set = self.get_best_config(endpoint)
        cached = self.get_or_train_model(endpoint, model_name, feature_set)

        # 构建新配方特征
        new_features = self._build_features(formulation_row, feature_set)

        # 对齐列: 缺失列填 0.0，顺序对齐
        missing_cols = set(cached["feature_columns"]) - set(new_features.columns)
        for c in missing_cols:
            new_features[c] = 0.0
        extra_cols = set(new_features.columns) - set(cached["feature_columns"])
        for c in extra_cols:
            new_features = new_features.drop(columns=[c])
        new_features = new_features[cached["feature_columns"]]

        X_new = new_features.values
        if cached["scaler"] is not None:
            X_new = cached["scaler"].transform(X_new)

        y_hat = float(cached["model"].predict(X_new)[0])

        # Conformal interval: 使用 OOF 残差的 P90 作为 q_value
        residuals = self.get_oof_residuals(endpoint, model_name, feature_set)
        if residuals:
            q_value = float(np.percentile(residuals, 90))
        else:
            q_value = float("nan")

        return {
            "y_hat": round(y_hat, 4),
            "pi_low": round(y_hat - q_value, 4),
            "pi_high": round(y_hat + q_value, 4),
            "interval_width": round(2 * q_value, 4),
            "model_name": model_name,
            "feature_set": feature_set,
        }

    # --- v1.3 批量预测 ---

    def batch_predict_formulations(
        self,
        formulations_df: pd.DataFrame,
        endpoints: list[str] | None = None,
    ) -> pd.DataFrame:
        """向量化批量预测多个配方的所有 endpoint 性能。

        Args:
            formulations_df: DataFrame，包含 lipid1-4, ratio1-4, np_ratio, aq_org_ratio, *_smiles
            endpoints: 预测的 endpoint 列表 (默认全部 3 个)

        Returns:
            原始 DataFrame 附加 y_hat_{ep}, pi_low_{ep}, pi_high_{ep}, interval_width_{ep} 列
        """
        if endpoints is None:
            endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]

        result_df = formulations_df.copy()

        for ep in endpoints:
            model_name, feature_set = self.get_best_config(ep)
            cached = self.get_or_train_model(ep, model_name, feature_set)

            # 批量构建特征
            new_features = self._build_features(formulations_df, feature_set)

            # 列对齐
            missing_cols = set(cached["feature_columns"]) - set(new_features.columns)
            for c in missing_cols:
                new_features[c] = 0.0
            extra_cols = set(new_features.columns) - set(cached["feature_columns"])
            for c in extra_cols:
                new_features = new_features.drop(columns=[c])
            new_features = new_features[cached["feature_columns"]]

            X_batch = new_features.values
            if cached["scaler"] is not None:
                X_batch = cached["scaler"].transform(X_batch)

            y_hats = cached["model"].predict(X_batch)

            # Conformal interval
            residuals = self.get_oof_residuals(ep, model_name, feature_set)
            q_value = float(np.percentile(residuals, 90)) if residuals else float("nan")

            result_df[f"y_hat_{ep}"] = np.round(y_hats, 4)
            result_df[f"pi_low_{ep}"] = np.round(y_hats - q_value, 4)
            result_df[f"pi_high_{ep}"] = np.round(y_hats + q_value, 4)
            result_df[f"interval_width_{ep}"] = round(2 * q_value, 4)

        return result_df
