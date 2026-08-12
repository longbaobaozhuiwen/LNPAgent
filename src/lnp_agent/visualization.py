"""科研级可视化模块 — Nature Style。

生成 Pareto 前沿散点图和化学空间 UMAP 降维图。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_nature_style():
    """配置 matplotlib Nature 期刊风格。"""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_pareto_front(
    pred_df: pd.DataFrame,
    exploitation_idx: list[int] | None = None,
    exploration_idx: list[int] | None = None,
    save_path: str | Path | None = None,
) -> str | None:
    """生成 Pareto 前沿散点图。

    横轴: 炎症评分 (min-max 归一化 immune signal A + immune signal B)
    纵轴: 转染效率 (Tx_log1p)
    颜色: Exploitation (红), Exploration (蓝), 其余 (灰)

    Args:
        pred_df: 包含 y_hat_immune_signal_a, y_hat_immune_signal_b, y_hat_tx_log1p 列
        exploitation_idx: Exploitation 选择的行索引
        exploration_idx: Exploration 选择的行索引
        save_path: 图片保存路径

    Returns:
        保存路径或 None
    """
    setup_nature_style()

    immune_signal_a = pred_df["y_hat_immune_signal_a"].values
    immune_signal_b = pred_df["y_hat_immune_signal_b"].values
    tx = pred_df["y_hat_tx_log1p"].values

    # 归一化炎症指标
    def _norm(arr):
        rng = arr.max() - arr.min()
        if rng < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr.min()) / rng

    inflammation_score = _norm(immune_signal_a) + _norm(immune_signal_b)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    # 所有点 (灰色)
    ax.scatter(inflammation_score, tx, c="#BDBDBD", s=15, alpha=0.5,
               label="All candidates", zorder=1)

    # Exploration (蓝色)
    if exploration_idx:
        ax.scatter(inflammation_score[exploration_idx], tx[exploration_idx],
                   c="#4285F4", s=25, alpha=0.8, label="Exploration", zorder=2)

    # Exploitation (红色)
    if exploitation_idx:
        ax.scatter(inflammation_score[exploitation_idx], tx[exploitation_idx],
                   c="#EA4335", s=25, alpha=0.8, label="Exploitation", zorder=3)

    ax.set_xlabel("Inflammation Score (immune signal A + immune signal B)")
    ax.set_ylabel("Transfection Efficiency (log1p)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        logger.info(f"Pareto front plot saved to {save_path}")
    plt.close(fig)
    return str(save_path) if save_path else None


def plot_chemical_umap(
    features: np.ndarray,
    exploitation_idx: list[int] | None = None,
    exploration_idx: list[int] | None = None,
    save_path: str | Path | None = None,
    random_state: int = 42,
) -> str | None:
    """生成化学空间 UMAP 降维图。

    Args:
        features: (N, D) 特征矩阵
        exploitation_idx: Exploitation 行索引
        exploration_idx: Exploration 行索引
        save_path: 保存路径

    Returns:
        保存路径或 None
    """
    import umap
    setup_nature_style()

    n_neighbors = min(30, max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.5,
        spread=1.0,
        metric="cosine",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    ax.scatter(embedding[:, 0], embedding[:, 1], c="#BDBDBD", s=15, alpha=0.5,
               label="All candidates", zorder=1)

    if exploration_idx:
        ax.scatter(embedding[exploration_idx, 0], embedding[exploration_idx, 1],
                   c="#4285F4", s=25, alpha=0.8, label="Exploration", zorder=2)

    if exploitation_idx:
        ax.scatter(embedding[exploitation_idx, 0], embedding[exploitation_idx, 1],
                   c="#EA4335", s=25, alpha=0.8, label="Exploitation", zorder=3)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
        logger.info(f"Chemical UMAP plot saved to {save_path}")
    plt.close(fig)
    return str(save_path) if save_path else None
