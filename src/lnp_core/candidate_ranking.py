"""Candidate ranking via OOF predictions and Pareto front for v5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lnp_core.data_contract import ENDPOINT_DIRECTION
from lnp_core.model_zoo import generate_model_zoo_oof_predictions, MODEL_CONFIGS, FEATURE_SETS


def compute_pareto_front(
    df: pd.DataFrame,
    objective_cols: list[str],
    directions: list[str],
) -> np.ndarray:
    """Compute Pareto front indices.

    Args:
        df: DataFrame with objective columns
        objective_cols: column names for objectives
        directions: "lower_is_better" or "higher_is_better" for each objective

    Returns:
        np.ndarray of bool, True = on Pareto front
    """
    n = len(df)
    if n == 0:
        return np.array([], dtype=bool)

    values = df[objective_cols].values.astype(float).copy()
    valid = np.isfinite(values).all(axis=1)

    # Normalize to "minimize" convention
    for j, direction in enumerate(directions):
        if direction == "higher_is_better":
            values[:, j] = -values[:, j]

    is_pareto = np.zeros(n, dtype=bool)
    is_pareto[valid] = True

    for i in range(n):
        if not valid[i] or not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not valid[j] or not is_pareto[j]:
                continue
            # Check if j dominates i: j is <= i on all, < on at least one
            all_leq = np.all(values[j] <= values[i])
            any_lt = np.any(values[j] < values[i])
            if all_leq and any_lt:
                is_pareto[i] = False
                break

    return is_pareto


def rank_candidates(
    oof_predictions: pd.DataFrame,
    primary_sort_col: str = "pred_immune_signal_a",
    primary_ascending: bool = True,
    secondary_sort_col: str = "pred_tx_log1p",
    secondary_ascending: bool = False,
) -> pd.DataFrame:
    """Rank candidates with fixed sort order.

    Args:
        oof_predictions: DataFrame with pred_immune_signal_a, pred_immune_signal_b, pred_tx_log1p

    Returns:
        DataFrame with added rank and pareto_front columns
    """
    result = oof_predictions.copy()

    # Compute Pareto front
    objective_cols = ["pred_immune_signal_a", "pred_immune_signal_b", "pred_tx_log1p"]
    directions = [
        ENDPOINT_DIRECTION["immune_signal_a"],
        ENDPOINT_DIRECTION["immune_signal_b"],
        ENDPOINT_DIRECTION["tx_log1p"],
    ]
    result["pareto_front"] = compute_pareto_front(result, objective_cols, directions)

    # Fixed sort
    result = result.sort_values(
        by=[primary_sort_col, secondary_sort_col],
        ascending=[primary_ascending, secondary_ascending],
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


def generate_candidate_pareto(
    df: pd.DataFrame,
    model_zoo_bench: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate candidate Pareto/ranking table using OOF predictions.

    Algorithm:
    1. Select best deployable config (highest mean Spearman across endpoints)
    2. Generate OOF predictions with that config
    3. Compute Pareto front
    4. Rank by pred_immune_signal_a asc, pred_tx_log1p desc
    5. Attach original formulation info
    """
    # Select best config by mean Spearman across all endpoints
    config_scores = model_zoo_bench.groupby(["model_name", "feature_set"])["mean_spearman"].mean()
    best_config = config_scores.idxmax()
    best_model, best_fs = best_config

    # Generate OOF predictions for the selected model and a small model ensemble.
    oof_preds = generate_model_zoo_oof_predictions(df, split_df, best_model, best_fs)
    ensemble_preds: dict[str, list[pd.Series]] = {ep: [] for ep in ["immune_signal_a", "immune_signal_b", "tx_log1p"]}
    for model_name, feature_set_name in [(best_model, best_fs), ("ridge", "design_one_hot"), ("hgbr", "design_one_hot")]:
        try:
            preds = generate_model_zoo_oof_predictions(df, split_df, model_name, feature_set_name)
            for ep in ensemble_preds:
                ensemble_preds[ep].append(preds[f"pred_{ep}"])
        except (ValueError, KeyError):
            continue

    # Attach formulation info
    info_cols = [
        "Formulation_ID", "template_key",
        "lipid1", "lipid2", "lipid3", "lipid4",
        "ratio1", "ratio2", "ratio3", "ratio4",
        "np_ratio", "aq_org_ratio",
    ]
    result = df[info_cols].copy()
    for col in ["pred_immune_signal_a", "pred_immune_signal_b", "pred_tx_log1p"]:
        result[col] = oof_preds[col].values
    for ep, series_list in ensemble_preds.items():
        if series_list:
            result[f"pred_uncertainty_{ep}"] = pd.concat(series_list, axis=1).std(axis=1, ddof=0).values

    # Compute uncertainty (fold std as proxy)
    # For simplicity, use the fold-level std from the zoo benchmark
    best_rows = model_zoo_bench[
        (model_zoo_bench["model_name"] == best_model) &
        (model_zoo_bench["feature_set"] == best_fs)
    ]
    for _, row in best_rows.iterrows():
        ep = row["endpoint"]
        pred_col = f"pred_{ep}"
        # Preserve benchmark spread as a fallback only when ensemble dispersion is unavailable.
        col = f"pred_uncertainty_{ep}"
        if col not in result:
            result[col] = row["std_spearman"]

    # Rank and compute Pareto
    result = rank_candidates(result)

    # Reorder columns
    col_order = [
        "rank", "pareto_front", "Formulation_ID", "template_key",
        "lipid1", "lipid2", "lipid3", "lipid4",
        "ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio",
        "pred_immune_signal_a", "pred_immune_signal_b", "pred_tx_log1p",
        "pred_uncertainty_immune_signal_a", "pred_uncertainty_immune_signal_b", "pred_uncertainty_tx_log1p",
    ]
    result = result[[c for c in col_order if c in result.columns]]

    return result


def generate_candidate_summary(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Generate Pareto front summary statistics."""
    n_pareto = candidate_df["pareto_front"].sum()
    n_total = len(candidate_df)
    pareto_fraction = n_pareto / n_total if n_total > 0 else 0.0
    templates_represented = candidate_df[candidate_df["pareto_front"]]["template_key"].nunique()

    risk_statement = (
        "Data-supported priority suggestions for wet-lab discussion. "
        "Not a universal ranking. Predictions based on leave-template-out OOF estimates; "
        "generalization to new templates is uncertain."
    )

    return pd.DataFrame([{
        "n_pareto_candidates": int(n_pareto),
        "n_total": n_total,
        "pareto_fraction": round(pareto_fraction, 4),
        "templates_represented": int(templates_represented),
        "risk_statement": risk_statement,
    }])
