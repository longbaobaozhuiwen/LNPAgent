"""v2.5 数据管理器: 高容量预测架构 — XGBoost + MLP。

v2.4 继承:
- 增量追加湿实验结果 + 全量重训练 + VRAM 隔离
- immune signal B HGBR 宽松超参 (max_depth=6, lr=0.1)
- Benchmark 缓存保留

v2.5 变更:
- 集成 XGBoost (GPU 加速: tree_method="hist" + device="cuda")
- 集成 MLPRegressor (3 层: 512-256-128)
- immune signal B endpoint 强制使用 XGBoost 替代 benchmark 选出的模型
- 覆盖 force_retrain_all() 添加 XGBoost/MLP 训练
- 覆盖 get_or_train_model() 支持高容量模型路由
"""

from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from lnp_agent.data_manager_v2 import DataManagerV2, IMMUNE_SIGNAL_B_HGBR_OVERRIDE
from lnp_agent.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 高容量模型配置
# ═══════════════════════════════════════════════════════════

def _build_high_capacity_configs():
    """延迟导入 XGBoost 并构建配置。"""
    from xgboost import XGBRegressor
    from sklearn.neural_network import MLPRegressor

    return {
        "xgboost": {
            "model_class": XGBRegressor,
            "fixed_params": {
                "max_depth": 8,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cuda",
                "random_state": 42,
            },
            "needs_scaling": False,
        },
        "mlp": {
            "model_class": MLPRegressor,
            "fixed_params": {
                "hidden_layer_sizes": (512, 256, 128),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "batch_size": 32,
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "n_iter_no_change": 20,
                "validation_fraction": 0.1,
                "random_state": 42,
            },
            "needs_scaling": True,
        },
    }


# immune signal B 强制使用 XGBoost
IMMUNE_SIGNAL_B_MODEL_STRATEGY = "xgboost"

# 每个端点可用的额外高容量模型 (除标准 Ridge/HGBR/Huber 外)
EXTRA_MODELS = {
    "immune_signal_a": [],           # immune signal A 使用 benchmark 选出的模型
    "immune_signal_b": ["xgboost"],  # immune signal B 额外训练 XGBoost
    "tx_log1p": [],       # TX 使用 benchmark 选出的模型
}


class DataManagerV3(DataManagerV2):
    """v2.5: 高容量预测架构。

    在 DataManagerV2 基础上:
    1. immune signal B 强制使用 XGBoost 替代 benchmark 选出的模型
    2. 额外训练高容量模型 (XGBoost/MLP)
    3. get_or_train_model() 支持高容量模型路由
    """

    def __init__(self, source_of_truth: Path | None = None):
        super().__init__(source_of_truth)
        self._high_capacity_configs = None

    @property
    def high_capacity_configs(self):
        """延迟加载高容量模型配置。"""
        if self._high_capacity_configs is None:
            self._high_capacity_configs = _build_high_capacity_configs()
        return self._high_capacity_configs

    def force_retrain_all(self) -> dict:
        """v2.5: 重训练所有模型 (标准 9 + 高容量额外模型)。

        标准 9: 3 endpoints × 3 models (Ridge/HGBR/Huber)
        额外: immune signal B × XGBoost
        """
        logger.info("Force retraining all models (v2.5: standard + high-capacity)...")

        # 0. 清理 VRAM
        clear_gpu_memory("before force_retrain_all")

        # 1. 清除缓存
        self._invalidate_model_caches()

        # 2. 标准模型重训练 (继承自 DataManagerV2)
        results = {}
        endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]
        standard_models = ["ridge", "hgbr", "huber"]
        feature_set = "design_morgan_fp_weighted"

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

    def _train_high_capacity_model(self, endpoint: str, hc_model_name: str,
                                    feature_set_name: str) -> dict:
        """训练高容量模型 (XGBoost/MLP)。"""
        config = self.high_capacity_configs[hc_model_name]
        features = self._build_features(self.df, feature_set_name)
        X_all = features.values
        y_all = self.df[endpoint].values.astype(float)

        scaler = None
        if config.get("needs_scaling"):
            scaler = StandardScaler()
            X_all = scaler.fit_transform(X_all)

        model = config["model_class"](**config["fixed_params"])
        logger.info(f"Training high-capacity model ({hc_model_name}) for {endpoint} "
                     f"on {X_all.shape[0]} samples, {X_all.shape[1]} features...")
        model.fit(X_all, y_all)

        key = (endpoint, hc_model_name, feature_set_name)
        entry = {
            "model": model,
            "scaler": scaler,
            "feature_columns": features.columns.tolist(),
        }
        self._trained_models[key] = entry
        self._save_model_to_disk(key, entry)
        logger.info(f"Trained and cached high-capacity model: {key}")
        return entry

    def get_or_train_model(self, endpoint: str, model_name: str,
                           feature_set_name: str,
                           param_override: dict | None = None) -> dict:
        """v2.5: 获取或训练模型。

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

        # 标准模型: 委托给父类
        return super().get_or_train_model(endpoint, model_name, feature_set_name,
                                          param_override=param_override)

    def get_oof_residuals(self, endpoint: str, model_name: str,
                          feature_set_name: str) -> list[float]:
        """v2.5: 计算 OOF 残差，支持高容量模型。"""
        key = (endpoint, model_name, feature_set_name)

        # 内存缓存
        if key in self._oof_residuals:
            return self._oof_residuals[key]

        # 磁盘缓存
        loaded = self._load_residuals_from_disk(key)
        if loaded is not None:
            self._oof_residuals[key] = loaded
            return loaded

        # 高容量模型: 使用自己的配置计算 OOF
        if model_name in self.high_capacity_configs:
            config = self.high_capacity_configs[model_name]
        else:
            # 标准模型: 委托给父类
            return super().get_oof_residuals(endpoint, model_name, feature_set_name)

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

            if config.get("needs_scaling"):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)

            mdl = config["model_class"](**config["fixed_params"])
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            residuals.extend(np.abs(y_te - y_pred).tolist())

        self._oof_residuals[key] = residuals
        self._save_residuals_to_disk(key, residuals)
        logger.info(f"Computed OOF residuals for high-capacity {key}: {len(residuals)} values")
        return residuals

    def get_best_config(self, endpoint: str) -> tuple[str, str]:
        """v2.5: 获取最佳模型配置。

        immune signal B 强制返回 ("xgboost", "design_morgan_fp_weighted")。
        其他 endpoint 使用父类 benchmark 选出的最佳配置。
        """
        if endpoint == "immune_signal_b":
            return (IMMUNE_SIGNAL_B_MODEL_STRATEGY, "design_morgan_fp_weighted")
        return super().get_best_config(endpoint)

    def get_best_model_for_prediction(self, endpoint: str,
                                       feature_set_name: str = "design_morgan_fp_weighted") -> dict:
        """v2.5: 获取用于预测的最佳模型。

        immune signal B 返回 XGBoost, 其他返回 benchmark 选出的最佳模型。
        """
        if endpoint == "immune_signal_b":
            key = (endpoint, IMMUNE_SIGNAL_B_MODEL_STRATEGY, feature_set_name)
            if key in self._trained_models:
                return self._trained_models[key]
            # fallback: 训练 XGBoost
            return self._train_high_capacity_model(endpoint, IMMUNE_SIGNAL_B_MODEL_STRATEGY,
                                                    feature_set_name)

        # 其他 endpoint: 使用 benchmark 选出的最佳模型
        return self.get_best_config(endpoint, feature_set_name)
