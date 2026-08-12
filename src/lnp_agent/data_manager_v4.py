"""v2.6 数据管理器: 特征选择降维 + 分位数回归个性化不确定性。

v2.5 继承:
- XGBoost (GPU 加速: tree_method="hist" + device="cuda")
- MLPRegressor (3 层: 512-256-128)
- immune signal B endpoint 强制使用 XGBoost
- UCB/EI 采集函数兼容

v2.6 变更:
- VarianceThreshold + SelectKBest 特征选择管道 (8194 → ~300)
- XGBoost/HGBR 分位数回归 (alpha=0.05/0.95) 个性化 interval_width
- Per-endpoint 特征选择器缓存
- batch_predict_formulations 使用分位数区间替代 conformal 常数区间
"""

from __future__ import annotations

import gc
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lnp_agent.data_manager_v3 import (
    IMMUNE_SIGNAL_B_MODEL_STRATEGY,
    DataManagerV3,
    EXTRA_MODELS,
)
from lnp_agent.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 特征选择配置
# ═══════════════════════════════════════════════════════════

FEATURE_SELECTION_CONFIG = {
    "variance_threshold": 0.01,   # 去除方差 < 0.01 的指纹位
    "select_k": 300,             # 保留 F-statistic 最高的 300 个特征
}

# ═══════════════════════════════════════════════════════════
# 分位数回归配置
# ═══════════════════════════════════════════════════════════

QUANTILE_ALPHAS = [0.05, 0.95]


def _build_quantile_config(base_config: dict, alpha: float) -> dict | None:
    """从基础模型配置构建分位数模型配置。

    Returns:
        分位数配置字典，或 None（如果模型不支持分位数回归）。
    """
    qc = deepcopy(base_config)
    model_cls_name = base_config["model_class"].__name__

    if model_cls_name == "XGBRegressor":
        qc["fixed_params"]["objective"] = "reg:quantileerror"
        qc["fixed_params"]["quantile_alpha"] = alpha
        # 移除与分位数目标不兼容的参数
        qc["fixed_params"].pop("eval_metric", None)
    elif model_cls_name == "HistGradientBoostingRegressor":
        qc["fixed_params"]["loss"] = "quantile"
        qc["fixed_params"]["quantile"] = alpha
    else:
        # Ridge / Huber / MLP 不支持原生分位数回归
        return None

    return qc


class DataManagerV4(DataManagerV3):
    """v2.6: 特征选择 + 分位数回归。

    在 DataManagerV3 基础上:
    1. VarianceThreshold + SelectKBest 特征选择 (8194 → ~300)
    2. XGBoost/HGBR 分位数回归 (alpha=0.05/0.95) 个性化不确定性
    3. Per-endpoint 特征选择器缓存
    """

    def __init__(self, source_of_truth: Path | None = None):
        super().__init__(source_of_truth)

    # ═══════════════════════════════════════════════════════════
    # 特征选择
    # ═══════════════════════════════════════════════════════════

    def _fit_feature_selector(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """为特定 endpoint 训练特征选择管道。

        Args:
            X: 特征矩阵 (n_samples, 8194)
            y: 目标变量 (n_samples,)

        Returns:
            fitted Pipeline (VarianceThreshold + SelectKBest)
        """
        config = FEATURE_SELECTION_CONFIG
        n_features = X.shape[1]
        k = min(config["select_k"], n_features)

        pipeline = Pipeline([
            ("var_threshold", VarianceThreshold(threshold=config["variance_threshold"])),
            ("select_k", SelectKBest(f_regression, k=k)),
        ])
        pipeline.fit(X, y)

        # 日志：记录降维效果
        var_step = pipeline.named_steps["var_threshold"]
        X_after_var = var_step.transform(X[:1])
        n_after_var = X_after_var.shape[1]
        logger.info(
            f"Feature selection: {n_features} → {n_after_var} (VarianceThreshold) "
            f"→ {k} (SelectKBest f_regression)"
        )
        return pipeline

    # ═══════════════════════════════════════════════════════════
    # 分位数回归
    # ═══════════════════════════════════════════════════════════

    def _train_quantile_models(self, endpoint: str, model_name: str,
                                X_selected: np.ndarray,
                                y: np.ndarray) -> dict[float, object]:
        """为指定 endpoint 训练分位数模型。

        Args:
            X_selected: 降维后的特征矩阵
            y: 目标变量

        Returns:
            {0.05: model_low, 0.95: model_high} 或 {} (如果模型不支持)
        """
        # 获取基础配置
        if model_name in self.high_capacity_configs:
            base_config = self.high_capacity_configs[model_name]
        else:
            from lnp_core.model_evaluation import V53_MODEL_CONFIGS
            if model_name not in V53_MODEL_CONFIGS:
                logger.warning(f"Unknown model {model_name}, skipping quantile training")
                return {}
            base_config = V53_MODEL_CONFIGS[model_name]

        quantile_models = {}
        for alpha in QUANTILE_ALPHAS:
            q_config = _build_quantile_config(base_config, alpha)
            if q_config is None:
                logger.info(f"Model {model_name} does not support quantile regression, "
                           f"will use conformal fallback for {endpoint}")
                return {}

            q_model = q_config["model_class"](**q_config["fixed_params"])
            q_model.fit(X_selected, y)
            quantile_models[alpha] = q_model
            logger.info(f"Trained quantile model: {endpoint}/{model_name}/alpha={alpha}")

        return quantile_models

    # ═══════════════════════════════════════════════════════════
    # Override: 模型训练 (含特征选择 + 分位数回归)
    # ═══════════════════════════════════════════════════════════

    def _train_high_capacity_model(self, endpoint: str, hc_model_name: str,
                                    feature_set_name: str) -> dict:
        """v2.6: 训练高容量模型 (含特征选择 + 分位数回归)。"""
        config = self.high_capacity_configs[hc_model_name]
        features = self._build_features(self.df, feature_set_name)
        X_all = features.values
        y_all = self.df[endpoint].values.astype(float)

        # v2.6: 特征选择
        selector = self._fit_feature_selector(X_all, y_all)
        X_selected = selector.transform(X_all)

        scaler = None
        if config.get("needs_scaling"):
            scaler = StandardScaler()
            X_selected = scaler.fit_transform(X_selected)

        model = config["model_class"](**config["fixed_params"])
        logger.info(f"Training high-capacity model ({hc_model_name}) for {endpoint} "
                     f"on {X_selected.shape[0]} samples, {X_selected.shape[1]} features "
                     f"(reduced from {X_all.shape[1]})...")
        model.fit(X_selected, y_all)

        # v2.6: 训练分位数模型
        quantile_models = self._train_quantile_models(
            endpoint, hc_model_name, X_selected, y_all)

        key = (endpoint, hc_model_name, feature_set_name)
        entry = {
            "model": model,
            "scaler": scaler,
            "feature_columns": features.columns.tolist(),
            "selector": selector,           # v2.6: 特征选择器
            "quantile_models": quantile_models,  # v2.6: 分位数模型
        }
        self._trained_models[key] = entry
        self._save_model_to_disk(key, entry)
        logger.info(f"Trained and cached high-capacity model: {key} "
                     f"(features: {X_selected.shape[1]}, quantile: {bool(quantile_models)})")
        return entry

    def get_or_train_model(self, endpoint: str, model_name: str,
                           feature_set_name: str,
                           param_override: dict | None = None) -> dict:
        """v2.6: 获取或训练模型 (含特征选择 + 分位数回归)。

        immune signal B endpoint 路由到高容量模型 (XGBoost)。
        其他 endpoint 使用标准流程 (继承 DataManagerV2)。
        """
        # immune signal B 强制使用 XGBoost (忽略 LLM 可能选的任何模型)
        if endpoint == "immune_signal_b" and model_name in ("ridge", "hgbr", "huber"):
            # 仍然训练标准模型 (用于 residual 计算)
            entry = super().get_or_train_model(endpoint, model_name, feature_set_name,
                                               param_override=param_override)
            # 同时确保 XGBoost 也被训练
            hc_key = (endpoint, IMMUNE_SIGNAL_B_MODEL_STRATEGY, feature_set_name)
            if hc_key not in self._trained_models:
                self._train_high_capacity_model(endpoint, IMMUNE_SIGNAL_B_MODEL_STRATEGY,
                                                 feature_set_name)
            return entry

        # 高容量模型名直接路由
        if model_name in self.high_capacity_configs:
            key = (endpoint, model_name, feature_set_name)
            if key in self._trained_models:
                return self._trained_models[key]
            loaded = self._load_model_from_disk(key)
            if loaded is not None:
                self._trained_models[key] = loaded
                return loaded
            return self._train_high_capacity_model(endpoint, model_name, feature_set_name)

        # 标准模型: 使用特征选择后训练
        key = (endpoint, model_name, feature_set_name)

        # 1. 缓存检查
        if key in self._trained_models:
            return self._trained_models[key]
        loaded = self._load_model_from_disk(key)
        if loaded is not None:
            self._trained_models[key] = loaded
            return loaded

        # 2. 构建特征 + 选择
        from lnp_core.model_evaluation import V53_MODEL_CONFIGS
        config = V53_MODEL_CONFIGS[model_name]

        if param_override:
            config = deepcopy(config)
            config["fixed_params"].update(param_override)

        features = self._build_features(self.df, feature_set_name)
        X_all = features.values
        y_all = self.df[endpoint].values.astype(float)

        # v2.6: 特征选择
        selector = self._fit_feature_selector(X_all, y_all)
        X_selected = selector.transform(X_all)

        scaler = None
        if config.get("needs_scaling"):
            scaler = StandardScaler()
            X_selected = scaler.fit_transform(X_selected)

        model = config["model_class"](**config["fixed_params"])
        logger.info(f"Training standard model ({model_name}) for {endpoint} "
                     f"on {X_selected.shape[0]} samples, {X_selected.shape[1]} features "
                     f"(reduced from {X_all.shape[1]})...")
        model.fit(X_selected, y_all)

        # v2.6: 训练分位数模型
        quantile_models = self._train_quantile_models(
            endpoint, model_name, X_selected, y_all)

        entry = {
            "model": model,
            "scaler": scaler,
            "feature_columns": features.columns.tolist(),
            "selector": selector,
            "quantile_models": quantile_models,
        }
        self._trained_models[key] = entry
        self._save_model_to_disk(key, entry)
        logger.info(f"Trained and cached standard model: {key} "
                     f"(features: {X_selected.shape[1]}, quantile: {bool(quantile_models)})")
        return entry

    def force_retrain_all(self) -> dict:
        """v2.6: 重训练所有模型 (标准 9 + 高容量额外模型)。

        标准 9: 3 endpoints × 3 models (Ridge/HGBR/Huber)
        额外: immune signal B × XGBoost
        所有模型均使用特征选择 + 分位数回归。
        """
        logger.info("Force retraining all models (v2.6: feature selection + quantile regression)...")

        # 0. 清理 VRAM
        clear_gpu_memory("before force_retrain_all")

        # 1. 清除缓存
        self._invalidate_model_caches()

        # 2. 标准模型重训练 (继承自 DataManagerV2, 但会使用 v2.6 的 get_or_train_model)
        results = {}
        endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]
        standard_models = ["ridge", "hgbr", "huber"]
        feature_set = "design_morgan_fp_weighted"

        from lnp_agent.data_manager_v2 import IMMUNE_SIGNAL_B_HGBR_OVERRIDE

        for ep in endpoints:
            for model_name in standard_models:
                key = f"{ep}_{model_name}_{feature_set}"
                try:
                    override = None
                    if ep == "immune_signal_b" and model_name == "hgbr":
                        override = IMMUNE_SIGNAL_B_HGBR_OVERRIDE
                    entry = self.get_or_train_model(ep, model_name, feature_set,
                                                    param_override=override)
                    residuals = self.get_oof_residuals(ep, model_name, feature_set)
                    results[key] = {
                        "status": "success",
                        "residuals_count": len(residuals),
                        "residuals_mean": round(float(np.mean(residuals)), 4) if residuals else 0,
                        "has_quantile": bool(entry.get("quantile_models")),
                        "n_features": entry.get("selector", Pipeline([]))
                                      .transform(np.zeros((1, 8194))).shape[1]
                                      if entry.get("selector") else "N/A",
                    }
                except Exception as e:
                    results[key] = {"status": "error", "error": str(e)}
                    logger.error(f"Failed to retrain {key}: {e}")
                gc.collect()

        # 3. 高容量模型训练 (immune signal B XGBoost)
        for ep, extra_model_names in EXTRA_MODELS.items():
            for hc_model_name in extra_model_names:
                key = f"{ep}_{hc_model_name}_{feature_set}"
                try:
                    entry = self._train_high_capacity_model(ep, hc_model_name, feature_set)
                    results[key] = {
                        "status": "success",
                        "model_type": "high_capacity",
                        "has_quantile": bool(entry.get("quantile_models")),
                    }
                    logger.info(f"High-capacity model trained: {key}")
                except Exception as e:
                    results[key] = {"status": "error", "error": str(e)}
                    logger.error(f"Failed to train high-capacity {key}: {e}")
                gc.collect()

        success = sum(1 for v in results.values() if v["status"] == "success")
        logger.info(f"Retraining complete: {success}/{len(results)} models successful")
        clear_gpu_memory("after force_retrain_all")
        return results

    # ═══════════════════════════════════════════════════════════
    # Override: 批量预测 (使用分位数区间)
    # ═══════════════════════════════════════════════════════════

    def batch_predict_formulations(
        self,
        formulations_df: pd.DataFrame,
        endpoints: list[str] | None = None,
    ) -> pd.DataFrame:
        """v2.6: 批量预测 — 特征选择 + 分位数回归个性化区间。"""
        if endpoints is None:
            endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]

        result_df = formulations_df.copy()

        for ep in endpoints:
            model_name, feature_set = self.get_best_config(ep)
            cached = self.get_or_train_model(ep, model_name, feature_set)

            # 构建全维度特征
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

            # v2.6: 应用特征选择器
            selector = cached.get("selector")
            if selector is not None:
                X_batch = selector.transform(X_batch)

            # 应用 scaler
            if cached["scaler"] is not None:
                X_batch = cached["scaler"].transform(X_batch)

            # 点预测
            y_hats = cached["model"].predict(X_batch)

            # v2.6: 分位数回归预测区间
            quantile_models = cached.get("quantile_models", {})
            if quantile_models and 0.05 in quantile_models and 0.95 in quantile_models:
                pi_low = quantile_models[0.05].predict(X_batch)
                pi_high = quantile_models[0.95].predict(X_batch)
                # 安全保护：确保 pi_low <= y_hat <= pi_high
                pi_low = np.minimum(pi_low, y_hats)
                pi_high = np.maximum(pi_high, y_hats)
                widths = pi_high - pi_low
                logger.info(f"Quantile regression intervals for {ep}: "
                           f"mean_width={np.mean(widths):.4f}, "
                           f"std_width={np.std(widths):.4f}, "
                           f"min={np.min(widths):.4f}, max={np.max(widths):.4f}")
            else:
                # Fallback: conformal prediction (v2.5 行为)
                residuals = self.get_oof_residuals(ep, model_name, feature_set)
                q_value = float(np.percentile(residuals, 90)) if residuals else float("nan")
                pi_low = y_hats - q_value
                pi_high = y_hats + q_value
                logger.info(f"Conformal prediction intervals for {ep}: "
                           f"constant_width={2 * q_value:.4f}")

            result_df[f"y_hat_{ep}"] = np.round(y_hats, 4)
            result_df[f"pi_low_{ep}"] = np.round(pi_low, 4)
            result_df[f"pi_high_{ep}"] = np.round(pi_high, 4)
            result_df[f"interval_width_{ep}"] = np.round(pi_high - pi_low, 4)

        return result_df

    # ═══════════════════════════════════════════════════════════
    # Override: OOF 残差 (使用特征选择)
    # ═══════════════════════════════════════════════════════════

    def get_oof_residuals(self, endpoint: str, model_name: str,
                          feature_set_name: str) -> list[float]:
        """v2.6: 计算 OOF 残差 — 使用特征选择。"""
        key = (endpoint, model_name, feature_set_name)

        # 内存缓存
        if key in self._oof_residuals:
            return self._oof_residuals[key]

        # 磁盘缓存
        loaded = self._load_residuals_from_disk(key)
        if loaded is not None:
            self._oof_residuals[key] = loaded
            return loaded

        # 获取模型配置
        if model_name in self.high_capacity_configs:
            config = self.high_capacity_configs[model_name]
        else:
            from lnp_core.model_evaluation import V53_MODEL_CONFIGS
            if model_name not in V53_MODEL_CONFIGS:
                return super().get_oof_residuals(endpoint, model_name, feature_set_name)
            config = V53_MODEL_CONFIGS[model_name]

        features = self._build_features(self.df, feature_set_name)
        X_all = features.values
        y_all = self.df[endpoint].values.astype(float)
        split_df = self.split_df

        residuals: list[float] = []
        for fold_id in sorted(split_df["fold_id"].unique()):
            fold_split = split_df[split_df["fold_id"] == fold_id]
            train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
            test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values
            if len(test_idx) == 0:
                continue

            X_tr = X_all[train_idx]
            X_te = X_all[test_idx]
            y_tr = y_all[train_idx]
            y_te = y_all[test_idx]

            # v2.6: fold-specific 特征选择
            fold_selector = self._fit_feature_selector(X_tr, y_tr)
            X_tr_sel = fold_selector.transform(X_tr)
            X_te_sel = fold_selector.transform(X_te)

            if config.get("needs_scaling"):
                sc = StandardScaler()
                X_tr_sel = sc.fit_transform(X_tr_sel)
                X_te_sel = sc.transform(X_te_sel)

            mdl = config["model_class"](**config["fixed_params"])
            mdl.fit(X_tr_sel, y_tr)
            y_pred = mdl.predict(X_te_sel)
            residuals.extend(np.abs(y_te - y_pred).tolist())

        self._oof_residuals[key] = residuals
        self._save_residuals_to_disk(key, residuals)
        logger.info(f"Computed OOF residuals for {key}: {len(residuals)} values "
                     f"(with feature selection)")
        return residuals
