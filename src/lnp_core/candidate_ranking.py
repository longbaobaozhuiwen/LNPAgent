"""Candidate ranking via OOF predictions and Pareto front for v5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lnp_core.data_contract import ENDPOINT_DIRECTION
from lnp_core.model_zoo import generate_model_zoo_oof_predictions, MODEL_CONFIGS, FEATURE_SETS


DEFAULT_EXPERIMENT_VALUE_WEIGHTS = {
    "exploitation": 0.55,
    "exploration": 0.30,
    "diversity": 0.15,
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    "tx_log1p": 0.50,
    "immune_signal_a": 0.25,
    "immune_signal_b": 0.25,
}

DEFAULT_OBJECTIVE_UNCERTAINTY_WEIGHT = 0.20

DEFAULT_BATCH_COMPLEMENTARITY_WEIGHT = 0.20


def _finite_minmax_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale a numeric column to [0, 1] while preserving invalid values as 0."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = np.isfinite(numeric)
    if not valid.any():
        return pd.Series(0.0, index=values.index)

    finite = numeric[valid]
    span = finite.max() - finite.min()
    if span <= 1e-12:
        scaled = pd.Series(0.5, index=values.index)
    else:
        scaled = (numeric - finite.min()) / span
    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.where(valid, 0.0).clip(0.0, 1.0)


def compute_experiment_value_scores(
    candidates: pd.DataFrame,
    weights: dict[str, float] | None = None,
    objective_weights: dict[str, float] | None = None,
    objective_uncertainty_weight: float = DEFAULT_OBJECTIVE_UNCERTAINTY_WEIGHT,
) -> pd.DataFrame:
    """Score candidates by expected experiment value.

    The score intentionally blends three algorithmic motives:
    exploitation of predicted biological objectives, exploration of uncertain
    candidates, and diversity across formulation/design regions.  It is a
    decision-policy score, not a biological efficacy claim.
    """
    result = candidates.copy()
    weights = {**DEFAULT_EXPERIMENT_VALUE_WEIGHTS, **(weights or {})}
    objective_weights = {**DEFAULT_OBJECTIVE_WEIGHTS, **(objective_weights or {})}
    uncertainty_weight = max(float(objective_uncertainty_weight), 0.0)

    exploitation_parts: list[pd.Series] = []
    exploitation_mean_parts: list[pd.Series] = []
    exploitation_uncertainty_parts: list[pd.Series] = []
    objective_specs = [
        ("pred_tx_log1p", True),
        ("pred_immune_signal_a", False),
        ("pred_immune_signal_b", False),
    ]
    exploitation_weights: list[float] = []
    for col, higher_is_better in objective_specs:
        if col in result:
            mean_score = _finite_minmax_score(result[col], higher_is_better)
            uncertainty_col = f"pred_uncertainty_{col.removeprefix('pred_')}"
            uncertainty_score = (
                _finite_minmax_score(result[uncertainty_col], True)
                if uncertainty_col in result
                else pd.Series(0.0, index=result.index)
            )
            exploitation_mean_parts.append(mean_score)
            exploitation_uncertainty_parts.append(uncertainty_score)
            uncertainty_direction = 1.0 if higher_is_better else -1.0
            exploitation_parts.append(
                mean_score + uncertainty_direction * uncertainty_weight * uncertainty_score
            )
            exploitation_weights.append(
                max(float(objective_weights.get(col.removeprefix("pred_"), 0.0)), 0.0)
            )
    if exploitation_parts:
        matrix = pd.concat(exploitation_parts, axis=1)
        mean_matrix = pd.concat(exploitation_mean_parts, axis=1)
        uncertainty_matrix = pd.concat(exploitation_uncertainty_parts, axis=1)
        weight_array = np.asarray(exploitation_weights, dtype=float)
        if weight_array.sum() <= 1e-12:
            weight_array = np.ones(len(exploitation_parts), dtype=float)
        normalized_weights = weight_array / weight_array.sum()
        result["mean_exploitation_score"] = mean_matrix.mul(
            normalized_weights, axis=1
        ).sum(axis=1)
        direction_array = np.asarray(
            [
                1.0 if higher else -1.0
                for col, higher in objective_specs
                if col in result
            ],
            dtype=float,
        )
        result["objective_uncertainty_bonus"] = uncertainty_matrix.mul(
            normalized_weights * uncertainty_weight * direction_array, axis=1
        ).sum(axis=1)
        result["exploitation_score"] = matrix.mul(normalized_weights, axis=1).sum(axis=1)
    else:
        result["exploitation_score"] = 0.0

    uncertainty_cols = [c for c in result.columns if c.startswith("pred_uncertainty_")]
    if uncertainty_cols:
        uncertainty = result[uncertainty_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        result["exploration_score"] = _finite_minmax_score(uncertainty, higher_is_better=True)
    else:
        result["exploration_score"] = 0.0

    diversity_parts: list[pd.Series] = []
    ratio_cols = [c for c in ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"] if c in result]
    if ratio_cols and len(result) > 1:
        ratio_frame = result[ratio_cols].apply(pd.to_numeric, errors="coerce")
        center = ratio_frame.median(axis=0, skipna=True)
        distance = (ratio_frame - center).abs().mean(axis=1)
        diversity_parts.append(_finite_minmax_score(distance, higher_is_better=True))
    for col in ["template_key", "lipid1", "lipid2", "lipid4"]:
        if col in result:
            frequency = result[col].map(result[col].value_counts(dropna=False))
            diversity_parts.append(_finite_minmax_score(frequency, higher_is_better=False))
    if diversity_parts:
        result["diversity_score"] = pd.concat(diversity_parts, axis=1).mean(axis=1)
    else:
        result["diversity_score"] = 0.0

    result["experiment_value_score"] = (
        weights["exploitation"] * result["exploitation_score"]
        + weights["exploration"] * result["exploration_score"]
        + weights["diversity"] * result["diversity_score"]
    )
    result["selection_rationale"] = np.select(
        [
            result["exploration_score"] >= result["exploitation_score"] + 0.15,
            result["diversity_score"] >= result["exploitation_score"] + 0.15,
        ],
        ["explore uncertain candidate", "cover under-sampled design region"],
        default="exploit predicted multi-objective trade-off",
    )
    return result


def _batch_similarity_to_selected(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.Series:
    """Return each candidate's maximum similarity to the selected batch."""
    if selected.empty:
        return pd.Series(0.0, index=candidates.index)

    similarity_parts: list[pd.Series] = []
    ratio_cols = [
        c
        for c in ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"]
        if c in candidates and c in selected
    ]
    if ratio_cols:
        pool_ratios = candidates[ratio_cols].apply(pd.to_numeric, errors="coerce")
        selected_ratios = selected[ratio_cols].apply(pd.to_numeric, errors="coerce")
        spans = pool_ratios.max(axis=0) - pool_ratios.min(axis=0)
        spans = spans.where(spans > 1e-12, 1.0)
        nearest = []
        for _, selected_row in selected_ratios.iterrows():
            distance = ((pool_ratios - selected_row).abs() / spans).mean(axis=1)
            nearest.append((1.0 - distance.clip(0.0, 1.0)).fillna(0.0))
        similarity_parts.append(pd.concat(nearest, axis=1).max(axis=1))

    uncertainty_cols = [
        c
        for c in [
            "pred_uncertainty_tx_log1p",
            "pred_uncertainty_immune_signal_a",
            "pred_uncertainty_immune_signal_b",
        ]
        if c in candidates and c in selected
    ]
    if uncertainty_cols:
        pool_uncertainty = candidates[uncertainty_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        selected_uncertainty = selected[uncertainty_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        spans = pool_uncertainty.max(axis=0) - pool_uncertainty.min(axis=0)
        spans = spans.where(spans > 1e-12, 1.0)
        nearest_uncertainty = []
        for _, selected_row in selected_uncertainty.iterrows():
            distance = (
                (pool_uncertainty - selected_row).abs() / spans
            ).mean(axis=1)
            nearest_uncertainty.append(
                (1.0 - distance.clip(0.0, 1.0)).fillna(0.0)
            )
        similarity_parts.append(
            pd.concat(nearest_uncertainty, axis=1).max(axis=1)
        )

    for col in ["template_key", "lipid1", "lipid2", "lipid4"]:
        if col in candidates and col in selected:
            selected_values = set(selected[col].astype(str))
            similarity_parts.append(candidates[col].astype(str).isin(selected_values).astype(float))

    if not similarity_parts:
        return pd.Series(0.0, index=candidates.index)
    return pd.concat(similarity_parts, axis=1).mean(axis=1).clip(0.0, 1.0)


def select_experiment_value_batch(
    candidates: pd.DataFrame,
    batch_size: int = 8,
    weights: dict[str, float] | None = None,
    batch_complementarity_weight: float = DEFAULT_BATCH_COMPLEMENTARITY_WEIGHT,
    objective_weights: dict[str, float] | None = None,
    objective_uncertainty_weight: float = DEFAULT_OBJECTIVE_UNCERTAINTY_WEIGHT,
) -> pd.DataFrame:
    """Select a small batch by score while discouraging within-batch redundancy."""
    if batch_size <= 0:
        return compute_experiment_value_scores(
            candidates,
            weights,
            objective_weights,
            objective_uncertainty_weight,
        ).head(0)
    scored = compute_experiment_value_scores(
        candidates,
        weights,
        objective_weights,
        objective_uncertainty_weight,
    )
    if "pareto_front" not in scored:
        scored["pareto_front"] = False

    remaining = scored.copy()
    selected_parts: list[pd.DataFrame] = []
    target_size = min(batch_size, len(remaining))
    complementarity_weight = max(float(batch_complementarity_weight), 0.0)

    while len(selected_parts) < target_size and not remaining.empty:
        selected = (
            pd.concat(selected_parts, axis=0)
            if selected_parts
            else remaining.head(0)
        )
        similarity = _batch_similarity_to_selected(remaining, selected)
        remaining = remaining.assign(
            batch_redundancy_score=similarity,
            batch_complementarity_score=1.0 - similarity,
        )
        remaining["batch_selection_score"] = remaining["experiment_value_score"]
        if not selected.empty:
            remaining["batch_selection_score"] = (
                remaining["batch_selection_score"]
                - complementarity_weight * remaining["batch_redundancy_score"]
            )
        next_row = remaining.sort_values(
            by=["batch_selection_score", "experiment_value_score", "pareto_front"],
            ascending=[False, False, False],
        ).head(1)
        selected_parts.append(next_row)
        remaining = remaining.drop(index=next_row.index)

    return pd.concat(selected_parts, axis=0).reset_index(drop=True)


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
    result = compute_experiment_value_scores(result)

    # Reorder columns
    col_order = [
        "rank", "pareto_front", "Formulation_ID", "template_key",
        "lipid1", "lipid2", "lipid3", "lipid4",
        "ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio",
        "pred_immune_signal_a", "pred_immune_signal_b", "pred_tx_log1p",
        "pred_uncertainty_immune_signal_a", "pred_uncertainty_immune_signal_b", "pred_uncertainty_tx_log1p",
        "exploitation_score", "exploration_score", "diversity_score",
        "experiment_value_score", "selection_rationale",
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
