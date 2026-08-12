"""可视化工具 v2.0: Pareto 前沿图 + 化学空间 UMAP。

v2.0 变更:
- 支持 CSV 文件路径输入 (predictions_path, exploitation_path, exploration_path)
- 保留 legacy JSON 输入 (predictions_json, features_json) 作为 fallback
- 自动索引匹配 (用关键列匹配 full_df 与 sub_df 的行)
- PlotChemicalSpaceUMAP 从 CSV 自动计算 AGILE 嵌入
"""

from __future__ import annotations

import json
import logging

from lnp_agent.tools.base import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


def _match_indices(full_df, sub_df) -> list[int] | None:
    """Find indices in full_df that match rows in sub_df."""
    key_cols = ["lipid1", "lipid2", "lipid4", "ratio1", "ratio4", "np_ratio",
                "combined_smiles", "id"]
    key_cols = [c for c in key_cols if c in full_df.columns and c in sub_df.columns]
    if not key_cols:
        return None
    merged = full_df.reset_index().merge(
        sub_df[key_cols].drop_duplicates(),
        on=key_cols, how="inner",
    )
    return sorted(merged["index"].tolist()) if len(merged) > 0 else None


class PlotParetoFront:
    """生成 Pareto 前沿散点图。"""

    definition = ToolDefinition(
        name="plot_pareto_front",
        description=(
            "Generate Pareto front scatter plot showing exploitation/exploration candidates. "
            "x-axis: inflammation score (immune signal A+immune signal B), y-axis: transfection efficiency. "
            "Accepts CSV file paths (preferred) or JSON strings."
        ),
        parameters={
            "predictions_path": {
                "type": "string",
                "description": "Path to CSV file with prediction results. "
                               "Columns should include y_hat_immune_signal_a, y_hat_immune_signal_b, y_hat_tx_log1p.",
            },
            "exploitation_path": {
                "type": "string",
                "description": "Path to CSV file with exploitation candidates (for highlighting).",
            },
            "exploration_path": {
                "type": "string",
                "description": "Path to CSV file with exploration candidates (for highlighting).",
            },
            "predictions_json": {
                "type": "string",
                "description": "(Legacy) JSON with 'predictions' array, 'exploitation_idx', 'exploration_idx'.",
            },
            "save_path": {
                "type": "string",
                "description": "Output PNG file path (default: auto-generated).",
            },
        },
        required=[],
    )

    def execute(
        self,
        predictions_path: str | None = None,
        exploitation_path: str | None = None,
        exploration_path: str | None = None,
        predictions_json: str | None = None,
        save_path: str | None = None,
    ) -> ToolResult:
        try:
            import pandas as pd

            # 加载预测数据
            if predictions_path:
                pred_df = pd.read_csv(predictions_path)
                logger.info(f"Loaded {len(pred_df)} predictions from {predictions_path}")
            elif predictions_json:
                data = json.loads(predictions_json)
                pred_df = pd.DataFrame(data["predictions"])
            else:
                return ToolResult(
                    success=False, output="",
                    error="Provide predictions_path (CSV) or predictions_json.",
                )

            # 加载 exploitation/exploration 索引
            exp_idx, expl_idx = None, None

            if exploitation_path:
                exp_df = pd.read_csv(exploitation_path)
                exp_idx = _match_indices(pred_df, exp_df)
                logger.info(f"Matched {len(exp_idx) if exp_idx else 0} exploitation candidates")

            if exploration_path:
                expl_df = pd.read_csv(exploration_path)
                expl_idx = _match_indices(pred_df, expl_df)
                logger.info(f"Matched {len(expl_idx) if expl_idx else 0} exploration candidates")

            # Legacy JSON 索引 (fallback)
            if predictions_json and not predictions_path:
                data = json.loads(predictions_json)
                exp_idx = exp_idx or data.get("exploitation_idx")
                expl_idx = expl_idx or data.get("exploration_idx")

            if save_path is None:
                from lnp_agent.paths import RESULTS_DIR
                save_path = str(RESULTS_DIR / "figures" / "fig1_pareto_front.png")

            from lnp_agent.visualization import plot_pareto_front
            path = plot_pareto_front(pred_df, exp_idx, expl_idx, save_path)

            return ToolResult(
                success=True,
                output=json.dumps({"plot_path": path, "n_predictions": len(pred_df)}),
            )
        except Exception as e:
            logger.error(f"PlotParetoFront failed: {e}")
            return ToolResult(success=False, output="", error=str(e))


class PlotChemicalSpaceUMAP:
    """生成化学空间 UMAP 图。"""

    definition = ToolDefinition(
        name="plot_chemical_space_umap",
        description=(
            "Generate UMAP chemical space visualization from molecular features. "
            "Accepts CSV file paths (preferred, auto-computes AGILE embeddings) or JSON features."
        ),
        parameters={
            "predictions_path": {
                "type": "string",
                "description": "Path to CSV file with formulation data and SMILES. "
                               "AGILE embeddings will be computed automatically.",
            },
            "exploitation_path": {
                "type": "string",
                "description": "Path to CSV file with exploitation candidates.",
            },
            "exploration_path": {
                "type": "string",
                "description": "Path to CSV file with exploration candidates.",
            },
            "features_json": {
                "type": "string",
                "description": "(Legacy) JSON with 'features' as 2D array, 'exploitation_idx', 'exploration_idx'.",
            },
            "save_path": {
                "type": "string",
                "description": "Output PNG file path (default: auto-generated).",
            },
        },
        required=[],
    )

    def execute(
        self,
        predictions_path: str | None = None,
        exploitation_path: str | None = None,
        exploration_path: str | None = None,
        features_json: str | None = None,
        save_path: str | None = None,
    ) -> ToolResult:
        try:
            import numpy as np
            import pandas as pd

            # 加载特征数据
            if predictions_path:
                df = pd.read_csv(predictions_path)
                logger.info(f"Loading {len(df)} formulations from {predictions_path}")
                try:
                    from lnp_agent.fusion_features import compute_agile_embeddings_for_library
                    features = compute_agile_embeddings_for_library(df)
                    logger.info(f"Computed AGILE embeddings: shape {features.shape}")
                except Exception as e:
                    logger.warning(f"AGILE embedding failed, falling back to Morgan: {e}")
                    features = self._compute_morgan_features(df)
            elif features_json:
                data = json.loads(features_json)
                features = np.array(data["features"])
            else:
                return ToolResult(
                    success=False, output="",
                    error="Provide predictions_path (CSV) or features_json.",
                )

            # 加载索引
            exp_idx, expl_idx = None, None
            if exploitation_path:
                exp_df = pd.read_csv(exploitation_path)
                if predictions_path:
                    full_df = pd.read_csv(predictions_path)
                    exp_idx = _match_indices(full_df, exp_df)
                else:
                    exp_idx = data.get("exploitation_idx") if features_json else None

            if exploration_path:
                expl_df = pd.read_csv(exploration_path)
                if predictions_path:
                    full_df = pd.read_csv(predictions_path)
                    expl_idx = _match_indices(full_df, expl_df)
                else:
                    expl_idx = data.get("exploration_idx") if features_json else None

            # Legacy fallback
            if features_json and not predictions_path:
                data = json.loads(features_json)
                exp_idx = exp_idx or data.get("exploitation_idx")
                expl_idx = expl_idx or data.get("exploration_idx")

            if save_path is None:
                from lnp_agent.paths import RESULTS_DIR
                save_path = str(RESULTS_DIR / "figures" / "fig2_chemical_space_umap.png")

            from lnp_agent.visualization import plot_chemical_umap
            path = plot_chemical_umap(features, exp_idx, expl_idx, save_path)

            return ToolResult(
                success=True,
                output=json.dumps({"plot_path": path, "n_points": len(features)}),
            )
        except Exception as e:
            logger.error(f"PlotChemicalSpaceUMAP failed: {e}")
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def _compute_morgan_features(df) -> "np.ndarray":
        """Fallback: 用 Morgan fingerprint 代替 AGILE 嵌入。"""
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        smiles_col = None
        for col in ["combined_smiles", "lipid1_smiles", "smiles"]:
            if col in df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError("No SMILES column found for feature computation")

        fps = []
        for s in df[smiles_col]:
            mol = Chem.MolFromSmiles(str(s)) if isinstance(s, str) else None
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(2048))
        return np.array(fps)
