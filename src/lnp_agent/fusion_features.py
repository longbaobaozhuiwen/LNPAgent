"""AGILE + Morgan FP 融合特征构建器。

融合策略:
  AGILE 嵌入: 4 脂质 × 512d = 2048d (不加权，保留独立空间表征)
  Morgan FP:  8194d (已含 np_ratio, aq_org_ratio)
  总计:       2048 + 8194 = 10242d
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from lnp_agent.paths import AGILE_CHECKPOINT

AGILE_CKPT = AGILE_CHECKPOINT

LIPID_SMILES_COLS = ["lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles"]

# 懒加载 AGILEPredictor 单例
_agile_predictor = None


def _get_agile_predictor():
    global _agile_predictor
    if _agile_predictor is None:
        from lnp_agent.agile_predictor import AGILEPredictor
        _agile_predictor = AGILEPredictor(str(AGILE_CKPT), device="cpu")
    return _agile_predictor


@functools.lru_cache(maxsize=512)
def _cached_encode(smiles: str) -> tuple:
    """缓存 SMILES → 512d 嵌入 (返回 tuple 用于 hashability)。"""
    if not smiles or not isinstance(smiles, str) or smiles == "nan":
        return tuple(np.zeros(512, dtype=np.float32).tolist())
    pred = _get_agile_predictor()
    emb = pred.encode_single(smiles)
    return tuple(emb.tolist())


def design_agile_morgan_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """构建 AGILE + Morgan 融合特征。总计 10242d。

    AGILE: 4 脂质 × 512d = 2048d (不加权)
    Morgan: 8194d (已含连续特征)

    Returns:
        DataFrame, shape (n_samples, 10242)
    """
    from lnp_core.feature_engineering import design_morgan_fp_weighted_v5_3

    # Morgan FP 特征 (8194d)
    morgan_features = design_morgan_fp_weighted_v5_3(df)

    # AGILE 嵌入 (4 × 512 = 2048d)
    agile_parts = []
    for col in LIPID_SMILES_COLS:
        if col not in df.columns:
            # 缺少 SMILES 列 → 零嵌入
            zero_df = pd.DataFrame(
                np.zeros((len(df), 512), dtype=np.float32),
                columns=[f"agile_{col}_{i}" for i in range(512)],
                index=df.index,
            )
            agile_parts.append(zero_df)
            continue
        embeddings = np.array(
            [np.array(_cached_encode(str(s))) for s in df[col]],
            dtype=np.float32,
        )
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"agile_{col}_{i}" for i in range(512)],
            index=df.index,
        )
        agile_parts.append(emb_df)

    agile_features = pd.concat(agile_parts, axis=1)

    # 拼接: 2048d AGILE + 8194d Morgan = 10242d
    result = pd.concat([agile_features, morgan_features], axis=1)
    result.index = df.index
    logger.info(f"Fusion features: {result.shape[1]}d for {len(df)} samples")
    return result
