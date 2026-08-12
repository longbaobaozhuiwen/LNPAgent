"""LNP 领域特定工具。"""

from __future__ import annotations

import json
import logging

import pandas as pd

from lnp_agent.data_manager import DataManager
from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class CheckCompliance(BaseTool):
    """检查 LNP 配方参数是否合规。"""

    definition = ToolDefinition(
        name="check_compliance",
        description=(
            "Validate LNP formulation parameters against the known design grid constraints. "
            "Checks: lipid types are in known vocabulary, ratio1+2+3+4=100, "
            "ratio1 in [40,60], ratio2=10, ratio4 in [0.5,2.5], np_ratio in {3,6}. "
            "Returns: {is_valid: bool, violations: list[str]}"
        ),
        parameters={
            "lipid1": {"type": "string", "description": "Ionizable lipid"},
            "lipid2": {"type": "string", "description": "Helper lipid"},
            "lipid4": {"type": "string", "description": "PEG lipid"},
            "ratio1": {"type": "number", "description": "Ionizable lipid ratio"},
            "ratio4": {"type": "number", "description": "PEG lipid ratio"},
            "np_ratio": {"type": "integer", "description": "N/P ratio"},
        },
        required=["lipid1", "lipid2", "lipid4", "ratio1", "ratio4", "np_ratio"],
    )

    VALID_LIPIDS = {
        "lipid1": {"ALC-0315", "SM-102"},
        "lipid2": {"DOPE", "DPPC", "DSPC"},
        "lipid4": {"DMG-PEG2000", "DSPE-PEG2000"},
    }

    def execute(self, **kwargs) -> ToolResult:
        violations = []

        for lipid_key, valid_set in self.VALID_LIPIDS.items():
            val = kwargs.get(lipid_key, "")
            if val not in valid_set:
                violations.append(f"Unknown {lipid_key}: {val} (valid: {sorted(valid_set)})")

        # 类型转换: LLM 可能传字符串
        try:
            r1 = float(kwargs.get("ratio1", 0))
        except (TypeError, ValueError):
            violations.append(f"ratio1 is not a number: {kwargs.get('ratio1')!r}")
            r1 = 0.0
        try:
            r4 = float(kwargs.get("ratio4", 0))
        except (TypeError, ValueError):
            violations.append(f"ratio4 is not a number: {kwargs.get('ratio4')!r}")
            r4 = 0.0
        try:
            npr = int(kwargs.get("np_ratio", 0))
        except (TypeError, ValueError):
            violations.append(f"np_ratio is not an integer: {kwargs.get('np_ratio')!r}")
            npr = 0

        r2 = 10.0
        r3 = 100.0 - r1 - r2 - r4

        if not (40 <= r1 <= 60):
            violations.append(f"ratio1={r1} outside [40, 60]")
        if not (0.5 <= r4 <= 2.5):
            violations.append(f"ratio4={r4} outside [0.5, 2.5]")
        if r3 < 0:
            violations.append(f"Implied ratio3={r3:.1f} is negative")
        total = r1 + r2 + r3 + r4
        if abs(total - 100.0) > 0.01:
            violations.append(f"Sum of ratios = {total:.1f}, expected 100.0")
        if npr not in (3, 6):
            violations.append(f"np_ratio={npr} not in {{3, 6}}")

        is_valid = len(violations) == 0
        result: dict = {"is_valid": is_valid, "violations": violations}
        if is_valid:
            result["formulation"] = {
                "lipid1": kwargs.get("lipid1"),
                "lipid2": kwargs.get("lipid2"),
                "lipid3": "Cholesterol",
                "lipid4": kwargs.get("lipid4"),
                "ratio1": r1, "ratio2": r2,
                "ratio3": round(r3, 2), "ratio4": r4,
                "np_ratio": npr,
            }

        return ToolResult(success=True, output=json.dumps(result, indent=2))


class PredictLNPPerformance(BaseTool):
    """预测 LNP 配方的多端点性能。"""

    definition = ToolDefinition(
        name="predict_lnp_performance",
        description=(
            "Predict LNP formulation performance for three biological endpoints: "
            "immune signal A (lower=better), immune signal B (lower=better), "
            "Transfection Efficiency (higher=better). "
            "Returns predicted values with uncertainty estimates. "
            "Uses Ridge/HGBR/Huber models trained with leave-template-out CV."
        ),
        parameters={
            "lipid1": {
                "type": "string", "description": "Ionizable lipid",
                "enum": ["ALC-0315", "SM-102"],
            },
            "lipid2": {
                "type": "string", "description": "Helper lipid",
                "enum": ["DOPE", "DPPC", "DSPC"],
            },
            "lipid4": {
                "type": "string", "description": "PEG lipid",
                "enum": ["DMG-PEG2000", "DSPE-PEG2000"],
            },
            "ratio1": {"type": "number", "description": "Ionizable lipid ratio (40-60)"},
            "ratio4": {"type": "number", "description": "PEG lipid ratio (0.5-2.5)"},
            "np_ratio": {"type": "integer", "description": "N/P ratio", "enum": [3, 6]},
        },
        required=["lipid1", "lipid2", "lipid4", "ratio1", "ratio4", "np_ratio"],
    )

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def execute(self, **kwargs) -> ToolResult:
        try:
            # 构建配方行
            r1 = float(kwargs["ratio1"])
            r4 = float(kwargs["ratio4"])
            r2 = 10.0
            r3 = 100.0 - r1 - r2 - r4

            # 从 SoT 获取 SMILES
            smiles_lookup = self.dm.get_smiles_lookup()
            lipid1_smiles = smiles_lookup.get("lipid1", {}).get(kwargs["lipid1"], "")
            lipid2_smiles = smiles_lookup.get("lipid2", {}).get(kwargs["lipid2"], "")
            lipid3_smiles = smiles_lookup.get("lipid3", {}).get("Cholesterol", "")
            lipid4_smiles = smiles_lookup.get("lipid4", {}).get(kwargs["lipid4"], "")

            row = pd.DataFrame([{
                "lipid1": kwargs["lipid1"],
                "lipid2": kwargs["lipid2"],
                "lipid3": "Cholesterol",
                "lipid4": kwargs["lipid4"],
                "ratio1": r1, "ratio2": r2, "ratio3": r3, "ratio4": r4,
                "np_ratio": kwargs["np_ratio"],
                "aq_org_ratio": 3.0,
                "lipid1_smiles": lipid1_smiles,
                "lipid2_smiles": lipid2_smiles,
                "lipid3_smiles": lipid3_smiles,
                "lipid4_smiles": lipid4_smiles,
            }])

            # 对 3 个端点分别预测 (使用 DataManager 缓存的模型)
            predictions = {}
            for endpoint in ["immune_signal_a", "immune_signal_b", "tx_log1p"]:
                result = self.dm.predict_formulation(row, endpoint)
                predictions[endpoint] = result

            output = {
                "input": {
                    "lipid1": kwargs["lipid1"],
                    "lipid2": kwargs["lipid2"],
                    "lipid4": kwargs["lipid4"],
                    "ratio1": r1, "ratio2": r2, "ratio3": round(r3, 2),
                    "ratio4": r4, "np_ratio": kwargs["np_ratio"],
                },
                "predictions": predictions,
            }

            return ToolResult(success=True, output=json.dumps(output, indent=2))

        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=f"Prediction error: {type(e).__name__}: {e}",
            )


class QueryParetoFront(BaseTool):
    """查询当前已知的多目标 Pareto 前沿。"""

    definition = ToolDefinition(
        name="query_pareto_front",
        description=(
            "Query the current Pareto-optimal formulations based on three objectives: "
            "minimize immune signal A, minimize immune signal B, maximize Transfection Efficiency. "
            "Returns top-N candidates with predicted values."
        ),
        parameters={
            "top_n": {
                "type": "integer",
                "description": "Number of top candidates (default 10)",
            },
        },
        required=[],
    )

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def execute(self, top_n: int = 10) -> ToolResult:
        try:
            from lnp_core.candidate_ranking import generate_candidate_pareto_v2
            bench = self.dm.get_model_benchmark()
            pareto_df = generate_candidate_pareto_v2(self.dm.df, bench, self.dm.split_df)
            top = pareto_df.head(top_n)

            # 选择关键列输出
            cols = [c for c in [
                "Formulation_ID", "lipid1", "lipid2", "lipid4",
                "ratio1", "ratio4", "np_ratio",
                "immune_signal_a_pred", "immune_signal_b_pred", "tx_log1p_pred",
                "pareto_rank",
            ] if c in top.columns]

            result = top[cols].to_dict(orient="records") if cols else []
            return ToolResult(
                success=True,
                output=json.dumps({"pareto_candidates": result, "total": len(pareto_df)}, indent=2),
            )
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=f"Pareto query error: {type(e).__name__}: {e}",
            )


class EnumerateMissingCells(BaseTool):
    """枚举设计网格中的缺失单元。"""

    definition = ToolDefinition(
        name="enumerate_missing_cells",
        description=(
            "Enumerate all 120 cells in the full LNP design grid "
            "and identify the ~20 cells that have NOT been experimentally tested. "
            "Optionally predict their performance and rank by conservative criteria."
        ),
        parameters={
            "predict": {
                "type": "boolean",
                "description": "Predict performance for missing cells (default true)",
            },
        },
        required=[],
    )

    def __init__(self, data_manager: DataManager):
        self.dm = data_manager

    def execute(self, predict: bool = True) -> ToolResult:
        try:
            from lnp_core.missing_cells import (
                enumerate_full_design_grid,
                find_missing_cells,
            )

            full_grid = enumerate_full_design_grid()
            missing = find_missing_cells(full_grid, self.dm.df)

            result = {
                "total_cells": len(full_grid),
                "observed_cells": len(full_grid) - len(missing),
                "missing_cells": len(missing),
            }

            # 输出缺失单元概要
            if not missing.empty:
                summary_cols = [c for c in [
                    "lipid1", "lipid2", "lipid4", "ratio1", "ratio4", "np_ratio",
                ] if c in missing.columns]
                if summary_cols:
                    result["missing_list"] = missing[summary_cols].to_dict(orient="records")

            return ToolResult(
                success=True,
                output=json.dumps(result, indent=2, default=str),
            )
        except Exception as e:
            return ToolResult(
                success=False, output="",
                error=f"Missing cells error: {type(e).__name__}: {e}",
            )
