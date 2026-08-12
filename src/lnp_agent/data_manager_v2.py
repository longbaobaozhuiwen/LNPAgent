"""v2.3 数据管理器: 扩展 DataManager 支持增量追加湿实验结果 + 全量重训练 + VRAM 隔离。

v2.2 新增:
- append_wet_lab_results(): 追加湿实验结果到训练集
- force_retrain_all(): 清除缓存并重训练所有模型
- df 属性重写: 100 行基础 + N×20 行湿实验数据
- 列名映射: measured_* → 训练格式
- KFold 替代 leave-template-out (Ugi-3CR 每个产物是独特模板)

v2.3 变更:
- force_retrain_all() 中集成 VRAM 清理 (gpu_utils.clear_gpu_memory)
- 每个模型训练后执行 gc.collect() 释放中间对象
- 训练前后 clear_gpu_memory 确保显存隔离
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

from lnp_agent.data_manager import DataManager
from lnp_agent.gpu_utils import clear_gpu_memory

logger = logging.getLogger(__name__)

# 固定脂质 SMILES (与 active_learning.py 一致)
CHOLESTEROL_SMILES = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"
DSPC_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCCCCCC"
PEG2000_DMG_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(CO)OC"


class DataManagerV2(DataManager):
    """v2.3: 扩展 v1.6 DataManager，支持主动学习数据追加、模型重训练和 VRAM 清理。

    关键变更:
    1. df 属性: 返回增强 DataFrame (原始 100 行 + 湿实验累积数据)
    2. split_df 属性: 使用 KFold 替代 leave-template-out
    3. append_wet_lab_results(): 追加新实验数据
    4. force_retrain_all(): 全量重训练，含 VRAM 清理和 gc.collect
    """

    def __init__(self, source_of_truth: Path | None = None):
        super().__init__(source_of_truth)
        self._wet_lab_accumulator: list[pd.DataFrame] = []
        self._augmented_df: pd.DataFrame | None = None
        self._augmented_split_df: pd.DataFrame | None = None
        self._is_augmented = False

    @property
    def df(self) -> pd.DataFrame:
        """返回增强后的训练数据 (原始 100 行 + 湿实验累积数据)。"""
        if self._augmented_df is not None:
            return self._augmented_df

        # 加载原始 100 行
        base_df = super().df.copy()

        if self._wet_lab_accumulator:
            wet_lab_df = pd.concat(self._wet_lab_accumulator, ignore_index=True)
            wet_lab_df = self._normalize_wet_lab_columns(wet_lab_df)
            self._augmented_df = pd.concat([base_df, wet_lab_df], ignore_index=True)
            self._augmented_df = self._recompute_keys(self._augmented_df)
            logger.info(f"Augmented dataset: {len(self._augmented_df)} rows "
                        f"({len(base_df)} base + {len(wet_lab_df)} wet-lab)")
            return self._augmented_df

        return base_df

    @property
    def split_df(self) -> pd.DataFrame:
        """KFold 5-fold 划分 (替代 leave-template-out)。"""
        if self._augmented_split_df is not None:
            return self._augmented_split_df

        # 如果没有增强数据，使用父类的 leave-template-out
        if not self._wet_lab_accumulator:
            return super().split_df

        # 增强数据: 使用 KFold
        current_df = self.df
        self._augmented_split_df = self._build_kfold_split(current_df, n_folds=5)
        return self._augmented_split_df

    def append_wet_lab_results(self, wet_lab_df: pd.DataFrame) -> int:
        """追加湿实验结果到累积器，返回新的总行数。

        Parameters
        ----------
        wet_lab_df : pd.DataFrame
            湿实验结果 DataFrame，含 measured_immune_signal_a/immune_signal_b/tx_log1p 列。

        Returns
        -------
        int
            追加后的数据集总行数。
        """
        logger.info(f"Appending {len(wet_lab_df)} wet-lab results to training data")
        self._wet_lab_accumulator.append(wet_lab_df.copy())
        # 强制下次访问重建
        self._augmented_df = None
        self._augmented_split_df = None
        self._is_augmented = True
        # 清除模型缓存 (数据已变化)
        self._invalidate_model_caches()
        new_size = len(self.df)
        logger.info(f"Dataset size after append: {new_size}")
        return new_size

    def force_retrain_all(self) -> dict:
        """清除所有缓存并重训练所有模型配置。

        v2.3: 在训练前后清理 VRAM，每个模型训练后 gc.collect()。

        Returns
        -------
        dict
            每个配置的训练结果 {config_key: {status, ...}}。
        """
        logger.info("Force retraining all models...")

        # 0. 清理 VRAM
        clear_gpu_memory("before force_retrain_all")

        # 1. 清除缓存
        self._invalidate_model_caches()

        # 2. 重训练 9 个最佳配置 (3 endpoints × 3 models × 1 feature set)
        results = {}
        endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]
        models = ["ridge", "hgbr", "huber"]
        feature_set = "design_morgan_fp_weighted"

        for ep in endpoints:
            for model_name in models:
                key = f"{ep}_{model_name}_{feature_set}"
                try:
                    entry = self.get_or_train_model(ep, model_name, feature_set)
                    residuals = self.get_oof_residuals(ep, model_name, feature_set)
                    results[key] = {
                        "status": "success",
                        "residuals_count": len(residuals),
                        "residuals_mean": round(float(np.mean(residuals)), 4) if residuals else 0,
                    }
                except Exception as e:
                    results[key] = {
                        "status": "error",
                        "error": str(e),
                    }
                    logger.error(f"Failed to retrain {key}: {e}")
                gc.collect()

        success = sum(1 for v in results.values() if v["status"] == "success")
        logger.info(f"Retraining complete: {success}/{len(results)} models successful")
        clear_gpu_memory("after force_retrain_all")
        return results

    def _invalidate_model_caches(self) -> None:
        """清除所有模型缓存 (内存 + 磁盘)。"""
        # 清内存
        self._trained_models.clear()
        self._oof_residuals.clear()
        self._model_bench = None

        # 清磁盘
        if self._cache_dir and self._cache_dir.exists():
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared model cache: {self._cache_dir}")

    def _normalize_wet_lab_columns(self, wet_lab_df: pd.DataFrame) -> pd.DataFrame:
        """将湿实验列名映射到训练数据格式。

        Wet-Lab → Training:
        - measured_immune_signal_a → immune_signal_a
        - measured_immune_signal_b → immune_signal_b
        - measured_tx_log1p → tx_log1p
        - 添加缺失的固定列 (ratio2, ratio3, aq_org_ratio, lipid names, SMILES)
        """
        df = wet_lab_df.copy()

        # 1. 端点列映射
        col_map = {
            "measured_immune_signal_a": "immune_signal_a",
            "measured_immune_signal_b": "immune_signal_b",
            "measured_tx_log1p": "tx_log1p",
        }
        for old_col, new_col in col_map.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        # 2. 转染效率 (反变换: tx = expm1(tx_log1p))
        if "tx_log1p" in df.columns and "transfection_efficiency" not in df.columns:
            df["transfection_efficiency"] = np.expm1(df["tx_log1p"])

        # 3. 确保脂质名称列存在
        if "lipid1" not in df.columns:
            # 尝试从 label 获取 (Ugi-3CR 格式)
            if "label" in df.columns:
                df["lipid1"] = df["label"].astype(str)
            elif "id" in df.columns:
                df["lipid1"] = df["id"].astype(str)

        if "lipid2" not in df.columns:
            df["lipid2"] = "DSPC"
        if "lipid3" not in df.columns:
            df["lipid3"] = "Chol"
        if "lipid4" not in df.columns:
            df["lipid4"] = "PEG2000-DMG"

        # 4. 确保 SMILES 列存在
        if "lipid1_smiles" not in df.columns and "combined_smiles" in df.columns:
            df["lipid1_smiles"] = df["combined_smiles"]
        if "lipid2_smiles" not in df.columns:
            df["lipid2_smiles"] = DSPC_SMILES
        if "lipid3_smiles" not in df.columns:
            df["lipid3_smiles"] = CHOLESTEROL_SMILES
        if "lipid4_smiles" not in df.columns:
            df["lipid4_smiles"] = PEG2000_DMG_SMILES

        # 5. 确保比例列存在
        if "ratio2" not in df.columns:
            df["ratio2"] = 10.0
        if "ratio4" not in df.columns:
            df["ratio4"] = 1.5
        if "ratio3" not in df.columns:
            df["ratio3"] = 100.0 - df.get("ratio1", 50.0) - df["ratio2"] - df["ratio4"]
        if "aq_org_ratio" not in df.columns:
            df["aq_org_ratio"] = 3.0
        if "np_ratio" not in df.columns:
            df["np_ratio"] = 0.08

        # 6. 确保必须的端点列存在
        required_endpoints = ["immune_signal_a", "immune_signal_b", "tx_log1p"]
        for ep in required_endpoints:
            if ep not in df.columns:
                logger.warning(f"Missing endpoint column: {ep}")

        return df

    def _recompute_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """为增强 DataFrame 重新计算 template_key 和 design_cell_key。"""
        if "template_key" not in df.columns:
            df["template_key"] = (
                df["lipid1"].astype(str) + "_"
                + df["lipid2"].astype(str) + "_"
                + df["lipid4"].astype(str)
            )
        if "design_cell_key" not in df.columns:
            df["design_cell_key"] = (
                df["template_key"] + "_np" + df["np_ratio"].astype(str)
            )
        return df

    def _build_kfold_split(self, df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
        """构建 KFold 划分矩阵 (用于 OOF 残差计算)。"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        rows = []
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(df)):
            for idx in train_idx:
                rows.append({
                    "fold_id": fold_id,
                    "row_index": int(idx),
                    "split_role": "train",
                })
            for idx in test_idx:
                rows.append({
                    "fold_id": fold_id,
                    "row_index": int(idx),
                    "split_role": "test",
                })
        split_df = pd.DataFrame(rows)
        logger.info(f"Built KFold split: {n_folds} folds, {len(df)} rows")
        return split_df
