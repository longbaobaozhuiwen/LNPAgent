"""主动学习工具: 批量预测 + 探索/利用双板过滤器 (v2.5 — UCB/EI 采集函数)。

v2.1 变更 (基于 v1.7):
- 所有 output_path 通过 path_utils 强制路由到 RESULTS_DIR/working_data/
- 所有 predictions_path 输入路由到 RESULTS_DIR/working_data/
- 彻底屏蔽 LLM 相对路径幻觉

v2.3 变更 (基于 v2.1):
- BatchPredictLNP.execute() 中 batch_predict_formulations 前后添加 GPU 显存清理
- 所有 to_csv 调用替换为 atomic_write_csv，防止写入中断导致文件损坏
- FilterExploitationBatch 和 FilterExplorationBatch 的 top_n 默认值从 10 改为 20 (共 40 候选)

v2.5 变更 (基于 v2.3):
- 新增 UCB (Upper Confidence Bound) 采集函数
- 新增 EI (Expected Improvement) 采集函数
- filter_exploitation_batch 支持 acquisition_method 参数: "performance"/"ucb"/"ei"
- filter_exploration_batch 支持 acquisition_method 参数
- 结果 CSV 中添加 acquisition_score 列
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult
from lnp_agent.path_utils import resolve_round_file

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 采集函数工具方法
# ═══════════════════════════════════════════════════════════

def _normalize(series: pd.Series) -> pd.Series:
    """Min-max 归一化到 [0, 1]。"""
    s_min, s_max = series.min(), series.max()
    if s_max - s_min < 1e-10:
        return pd.Series(0.5, index=series.index)
    return (series - s_min) / (s_max - s_min)


def _compute_ucb_score(df: pd.DataFrame, kappa: float = 2.0) -> pd.Series:
    """UCB (Upper Confidence Bound) 采集函数。

    Score = normalize(μ_tx) - normalize(μ_immune_signal_a) - normalize(μ_immune_signal_b) + κ * normalize(avg_uncertainty)

    Args:
        df: 包含 y_hat_* 和 interval_width_* 列的 DataFrame
        kappa: 探索-利用平衡参数
            - kappa=0: 纯 exploitation
            - kappa=2: 标准 UCB (推荐)
            - kappa>5: 高度探索
    """
    score = pd.Series(0.0, index=df.index)

    if "y_hat_tx_log1p" in df.columns:
        score += _normalize(df["y_hat_tx_log1p"])
    if "y_hat_immune_signal_a" in df.columns:
        score -= _normalize(df["y_hat_immune_signal_a"])
    if "y_hat_immune_signal_b" in df.columns:
        score -= _normalize(df["y_hat_immune_signal_b"])

    # 平均不确定性
    iw_cols = [c for c in df.columns if c.startswith("interval_width_")]
    if iw_cols:
        avg_uncertainty = df[iw_cols].mean(axis=1)
        score += kappa * _normalize(avg_uncertainty)

    return score


def _compute_ei_score(df: pd.DataFrame, f_best: float,
                       endpoint: str = "tx_log1p") -> pd.Series:
    """EI (Expected Improvement) 采集函数。

    EI = σ(x) · [z · Φ(z) + φ(z)]
    其中 z = (μ(x) - f_best) / σ(x)

    Args:
        df: 包含 y_hat_* 和 interval_width_* 列的 DataFrame
        f_best: 历史最优观测值
        endpoint: 优化的目标 endpoint
    """
    mu = df.get(f"y_hat_{endpoint}")
    sigma_col = f"interval_width_{endpoint}"
    sigma = df.get(sigma_col)

    if mu is None or sigma is None:
        # fallback: 用综合性能
        score = pd.Series(0.0, index=df.index)
        if "y_hat_tx_log1p" in df.columns:
            score += _normalize(df["y_hat_tx_log1p"])
        return score

    sigma = sigma / 2  # interval_width ≈ 2σ
    sigma = sigma.clip(lower=1e-8)

    z = (mu - f_best) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))

    return ei


# ═══════════════════════════════════════════════════════════
# BatchPredictLNP (无变更)
# ═══════════════════════════════════════════════════════════

class BatchPredictLNP(BaseTool):
    """批量预测多个 LNP 配方性能。"""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.definition = ToolDefinition(
            name="batch_predict_lnp",
            description=(
                "Batch predict LNP performance for multiple formulations. "
                "Supports file-based input (CSV/JSONL) via formulations_path "
                "or legacy JSON string via formulations."
            ),
            parameters={
                "formulations_path": {
                    "type": "string",
                    "description": "Path to CSV/JSONL file with formulations. "
                                   "Columns: lipid1, lipid2, lipid4, ratio1, ratio4, np_ratio. "
                                   "Prefer this over formulations for large batches.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write prediction results CSV. "
                                   "Default: auto-generated in working_data/.",
                },
                "formulations": {
                    "type": "string",
                    "description": "(Legacy) JSON array of formulation objects.",
                },
                "endpoints": {
                    "type": "string",
                    "description": "(Legacy) JSON array of endpoints. Default: all 3.",
                },
            },
            required=[],
        )

    def execute(
        self,
        formulations_path: str | None = None,
        output_path: str | None = None,
        formulations: str | None = None,
        endpoints: str | None = None,
        **kwargs,
    ) -> ToolResult:
        from lnp_agent.gpu_utils import clear_gpu_memory
        from lnp_agent.path_utils import atomic_write_csv, resolve_round_file, sanitize_output_path

        # v2.3: 路由输入路径
        if formulations_path:
            formulations_path = str(resolve_round_file(
                formulations_path, "virtual_library.csv"))

        # 加载配方数据
        df = None
        if formulations_path:
            df = self._load_from_file(formulations_path)
            if df is None:
                return ToolResult(success=False, output="",
                                  error=f"Failed to load formulations from {formulations_path}")
        elif formulations:
            try:
                formulations_list = json.loads(formulations)
            except json.JSONDecodeError as e:
                return ToolResult(success=False, output="", error=f"Invalid JSON: {e}")
            if not formulations_list:
                return ToolResult(success=True, output='{"predictions": [], "count": 0}')
            df = pd.DataFrame(formulations_list)
        else:
            return ToolResult(success=False, output="",
                              error="Provide formulations_path or formulations")

        # 解析 endpoints
        endpoints_list = None
        if endpoints:
            try:
                endpoints_list = json.loads(endpoints)
            except json.JSONDecodeError:
                endpoints_list = None

        try:
            df = self._auto_map_columns(df)

            lookup = self.dm.get_smiles_lookup()
            for lipid_col in ["lipid1", "lipid2", "lipid4"]:
                smiles_col = f"{lipid_col}_smiles"
                if smiles_col not in df.columns and lipid_col in lookup:
                    df[smiles_col] = df[lipid_col].map(lookup[lipid_col])

            if "lipid3_smiles" not in df.columns:
                df["lipid3_smiles"] = "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C"

            if "ratio3" not in df.columns:
                df["ratio3"] = 100.0 - df["ratio1"] - df.get("ratio2", 10.0) - df["ratio4"]
            if "ratio2" not in df.columns:
                df["ratio2"] = 10.0
            if "aq_org_ratio" not in df.columns:
                df["aq_org_ratio"] = 3.0

            # 批量预测
            clear_gpu_memory("before batch_predict_formulations")
            result_df = self.dm.batch_predict_formulations(df, endpoints_list)
            clear_gpu_memory("after batch_predict_formulations")

            # 输出
            ep_cols = [c for c in result_df.columns
                       if c.startswith(("y_hat_", "pi_low_", "pi_high_", "interval_width_"))]
            smiles_cols = [c for c in result_df.columns if c.endswith("_smiles")]
            output_cols = (["lipid1", "lipid2", "lipid4", "ratio1", "ratio4", "np_ratio"]
                           + smiles_cols + ep_cols)
            output_cols = [c for c in output_cols if c in result_df.columns]

            out = sanitize_output_path(output_path, "predictions.csv")
            atomic_write_csv(result_df[output_cols], out)
            return ToolResult(success=True, output=json.dumps({
                "output_file": str(out),
                "count": len(result_df),
            }, ensure_ascii=False))

        except Exception as e:
            return ToolResult(success=False, output="",
                              error=f"Batch prediction error: {type(e).__name__}: {e}")

    def _auto_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """v2.1: 自动检测 Ugi-3CR 格式并映射到配方格式。"""
        has_lipid1 = "lipid1" in df.columns
        has_combined = "combined_smiles" in df.columns

        if has_lipid1:
            return df

        if has_combined:
            logger.info(f"Auto-mapping Ugi-3CR format ({len(df)} rows) → formulation format")
            df["lipid1"] = df.get("label", df.get("id", pd.Series(range(len(df))))).astype(str)
            df["lipid1_smiles"] = df["combined_smiles"]
            df["lipid2"] = "DSPC"
            df["lipid2_smiles"] = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC(CO)OC(=O)CCCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCCCCCC"
            df["lipid4"] = "PEG2000-DMG"
            df["lipid4_smiles"] = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC(CO)OC"
            df["ratio1"] = 50.0
            df["ratio2"] = 10.0
            df["ratio4"] = 1.5
            df["np_ratio"] = 0.08
        else:
            logger.warning(f"Unknown format, columns: {list(df.columns)}")

        return df

    def _load_from_file(self, path: str) -> pd.DataFrame | None:
        """从 CSV 或 JSONL 文件加载配方数据。"""
        try:
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                return pd.read_csv(path)
            elif ext in (".jsonl", ".json"):
                return pd.read_json(path, lines=(ext == ".jsonl"))
            else:
                logger.error(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None


# ═══════════════════════════════════════════════════════════
# FilterExploitationBatch (v2.5: +UCB/EI)
# ═══════════════════════════════════════════════════════════

class FilterExploitationBatch(BaseTool):
    """利用型过滤器: 返回预测性能最优的 top-N 候选物 (v2.5: 支持 UCB/EI)。"""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.definition = ToolDefinition(
            name="filter_exploitation_batch",
            description=(
                "Filter batch predictions for exploitation: return top-N by predicted performance. "
                "v2.5: Supports acquisition_method parameter for Bayesian acquisition functions."
            ),
            parameters={
                "predictions_path": {
                    "type": "string",
                    "description": "Path to CSV file with batch prediction results.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write filtered results CSV.",
                },
                "predictions_json": {
                    "type": "string",
                    "description": "(Legacy) JSON string of batch prediction results.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top candidates (default 20).",
                },
                "acquisition_method": {
                    "type": "string",
                    "description": 'Acquisition function: "performance" (default), "ucb", or "ei".',
                },
                "kappa": {
                    "type": "number",
                    "description": "UCB exploration parameter (default 2.0). Only used with method=ucb.",
                },
            },
            required=[],
        )

    def execute(
        self,
        predictions_path: str | None = None,
        output_path: str | None = None,
        predictions_json: str | None = None,
        top_n: int = 20,
        acquisition_method: str = "performance",
        kappa: float = 2.0,
        **kwargs,
    ) -> ToolResult:
        # 路由输入路径
        if predictions_path:
            predictions_path = str(resolve_round_file(
                predictions_path, ["predictions_round_*.csv", "*predictions*.csv", "*predict*.csv"]))

        # 加载预测数据
        predictions = self._load_predictions(predictions_path, predictions_json)
        if predictions is None:
            return ToolResult(success=False, output="", error="Provide predictions_path or predictions_json")
        if not predictions:
            return ToolResult(success=True, output='{"top_candidates": [], "count": 0}')

        try:
            df = pd.DataFrame(predictions)

            if acquisition_method == "ucb":
                df["acquisition_score"] = _compute_ucb_score(df, kappa)
                logger.info(f"UCB filtering (kappa={kappa}): score range "
                             f"[{df['acquisition_score'].min():.4f}, {df['acquisition_score'].max():.4f}]")
                top = df.nlargest(min(top_n, len(df)), "acquisition_score")

            elif acquisition_method == "ei":
                f_best = df["y_hat_tx_log1p"].max() if "y_hat_tx_log1p" in df.columns else 0.0
                df["acquisition_score"] = _compute_ei_score(df, f_best, endpoint="tx_log1p")
                logger.info(f"EI filtering (f_best={f_best:.4f}): score range "
                             f"[{df['acquisition_score'].min():.4f}, {df['acquisition_score'].max():.4f}]")
                top = df.nlargest(min(top_n, len(df)), "acquisition_score")

            else:
                # 原有 performance 逻辑
                score = pd.Series(0.0, index=df.index)
                if "y_hat_immune_signal_a" in df.columns:
                    score -= _normalize(df["y_hat_immune_signal_a"])
                if "y_hat_immune_signal_b" in df.columns:
                    score -= _normalize(df["y_hat_immune_signal_b"])
                if "y_hat_tx_log1p" in df.columns:
                    score += _normalize(df["y_hat_tx_log1p"])
                df["acquisition_score"] = score
                top = df.nlargest(min(top_n, len(df)), "acquisition_score")

            return self._output_result(top, "top_candidates", output_path)

        except Exception as e:
            return ToolResult(success=False, output="",
                              error=f"Filter error: {type(e).__name__}: {e}")

    def _load_predictions(self, predictions_path, predictions_json):
        """从文件或 JSON 字符串加载预测数据。"""
        if predictions_path:
            try:
                df = pd.read_csv(predictions_path)
                return df.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Failed to load {predictions_path}: {e}")
                return None
        elif predictions_json:
            try:
                data = json.loads(predictions_json)
                return data.get("predictions", data if isinstance(data, list) else [])
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                return None
        return None

    def _output_result(self, df: pd.DataFrame, key: str, output_path: str | None) -> ToolResult:
        """输出结果到文件或 JSON。"""
        from lnp_agent.path_utils import atomic_write_csv, sanitize_output_path

        records = df.to_dict(orient="records")
        out = sanitize_output_path(output_path, "exploitation_candidates.csv")
        atomic_write_csv(df, out)
        return ToolResult(success=True, output=json.dumps({
            f"{key}": records,
            "output_file": str(out),
            "count": len(records),
        }, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════
# FilterExplorationBatch (v2.5: +UCB/EI)
# ═══════════════════════════════════════════════════════════

class FilterExplorationBatch(BaseTool):
    """探索型过滤器: 返回高不确定性或化学空间多样的 top-N 候选物 (v2.5: 支持 UCB/EI)。"""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.definition = ToolDefinition(
            name="filter_exploration_batch",
            description=(
                "Filter batch predictions for exploration: return top-N by prediction uncertainty, "
                "AGILE embedding clustering diversity, or Bayesian acquisition functions."
            ),
            parameters={
                "predictions_path": {
                    "type": "string",
                    "description": "Path to CSV file with batch prediction results.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to write filtered results CSV.",
                },
                "predictions_json": {
                    "type": "string",
                    "description": "(Legacy) JSON string of batch prediction results.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of candidates (default 20).",
                },
                "endpoint_focus": {
                    "type": "string",
                    "description": "Which endpoint's uncertainty to prioritize (default: tx_log1p).",
                },
                "diversity_method": {
                    "type": "string",
                    "description": 'Selection method: "uncertainty" (default), "agile_cluster", "ucb", or "ei".',
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters for agile_cluster method (default: 5).",
                },
                "kappa": {
                    "type": "number",
                    "description": "UCB exploration parameter (default 3.0 for exploration).",
                },
            },
            required=[],
        )

    def execute(
        self,
        predictions_path: str | None = None,
        output_path: str | None = None,
        predictions_json: str | None = None,
        top_n: int = 20,
        endpoint_focus: str = "tx_log1p",
        diversity_method: str = "uncertainty",
        n_clusters: int = 5,
        kappa: float = 3.0,
        **kwargs,
    ) -> ToolResult:
        from lnp_agent.path_utils import atomic_write_csv, sanitize_output_path

        # 路由输入路径
        if predictions_path:
            predictions_path = str(resolve_round_file(
                predictions_path, ["predictions_round_*.csv", "*predictions*.csv", "*predict*.csv"]))

        # 加载预测数据
        predictions = self._load_predictions(predictions_path, predictions_json)
        if predictions is None:
            return ToolResult(success=False, output="", error="Provide predictions_path or predictions_json")
        if not predictions:
            return ToolResult(success=True, output='{"top_candidates": [], "count": 0}')

        try:
            df = pd.DataFrame(predictions)

            if diversity_method == "agile_cluster":
                top = self._select_by_agile_cluster(df, top_n, n_clusters)

            elif diversity_method == "ucb":
                # UCB with high kappa for exploration
                df["acquisition_score"] = _compute_ucb_score(df, kappa)
                logger.info(f"UCB exploration filtering (kappa={kappa})")
                top = df.nlargest(min(top_n, len(df)), "acquisition_score")

            elif diversity_method == "ei":
                f_best = df["y_hat_tx_log1p"].max() if "y_hat_tx_log1p" in df.columns else 0.0
                df["acquisition_score"] = _compute_ei_score(df, f_best, endpoint=endpoint_focus)
                logger.info(f"EI exploration filtering (f_best={f_best:.4f})")
                top = df.nlargest(min(top_n, len(df)), "acquisition_score")

            else:
                top = self._select_by_uncertainty(df, top_n, endpoint_focus)

            out = sanitize_output_path(output_path, "exploration_candidates.csv")

            records = top.to_dict(orient="records")
            atomic_write_csv(top, out)
            return ToolResult(success=True, output=json.dumps({
                "top_candidates": records,
                "output_file": str(out),
                "count": len(records),
            }, ensure_ascii=False))

        except Exception as e:
            return ToolResult(success=False, output="",
                              error=f"Filter error: {type(e).__name__}: {e}")

    def _load_predictions(self, predictions_path, predictions_json):
        """从文件或 JSON 字符串加载预测数据。"""
        if predictions_path:
            try:
                df = pd.read_csv(predictions_path)
                return df.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Failed to load {predictions_path}: {e}")
                return None
        elif predictions_json:
            try:
                data = json.loads(predictions_json)
                return data.get("predictions", data if isinstance(data, list) else [])
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                return None
        return None

    def _select_by_uncertainty(self, df: pd.DataFrame, top_n: int,
                                endpoint_focus: str) -> pd.DataFrame:
        """基于预测不确定性选择。"""
        col = f"interval_width_{endpoint_focus}"
        if col not in df.columns:
            iw_cols = [c for c in df.columns if c.startswith("interval_width_")]
            if iw_cols:
                df["_uncertainty"] = df[iw_cols].mean(axis=1)
            else:
                raise ValueError("No interval width columns found")
        else:
            df["_uncertainty"] = df[col]

        df["acquisition_score"] = df["_uncertainty"]
        top = df.nlargest(min(top_n, len(df)), "_uncertainty")
        return top.drop(columns=["_uncertainty"])

    def _select_by_agile_cluster(self, df: pd.DataFrame, top_n: int,
                                  n_clusters: int) -> pd.DataFrame:
        """基于 AGILE 嵌入聚类选择骨架多样的配方。"""
        from lnp_agent.fusion_features import (
            compute_agile_embeddings_for_library,
            select_diverse_by_agile_clustering,
        )
        embeddings = compute_agile_embeddings_for_library(df)
        diverse_idx = select_diverse_by_agile_clustering(
            df, n_select=top_n, n_clusters=n_clusters,
            embedding_matrix=embeddings,
        )
        return df.iloc[diverse_idx]
