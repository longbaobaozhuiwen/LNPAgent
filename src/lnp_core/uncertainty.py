"""Bootstrap CI and paired-delta uncertainty quantification for v5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_CI_LEVEL = 0.90


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    random_state: int = 42,
) -> dict:
    """Compute bootstrap CI for a set of fold scores.

    Args:
        values: fold-wise scores, shape (n_folds,)
        n_bootstrap: resample count (default 10000)
        ci_level: confidence level (default 0.90)
        random_state: random seed

    Returns:
        dict with mean, std, ci_lower, ci_upper, n_samples
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_samples": 0}
    if n == 1:
        return {"mean": float(values[0]), "std": 0.0, "ci_lower": float(values[0]), "ci_upper": float(values[0]), "n_samples": 1}

    rng = np.random.RandomState(random_state)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": n,
    }


def paired_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    random_state: int = 42,
) -> dict:
    """Compute paired delta CI between two sets of fold scores.

    Computes delta_i = a_i - b_i for each fold, then bootstraps mean delta.

    Returns:
        dict with delta_mean, delta_std, ci_lower, ci_upper, significant, n_pairs
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    mask = np.isfinite(values_a) & np.isfinite(values_b)
    a = values_a[mask]
    b = values_b[mask]
    n = len(a)

    if n == 0:
        return {"delta_mean": np.nan, "delta_std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "significant": False, "n_pairs": 0}
    if n == 1:
        d = float(a[0] - b[0])
        return {"delta_mean": d, "delta_std": 0.0, "ci_lower": d, "ci_upper": d, "significant": False, "n_pairs": 1}

    deltas = a - b
    rng = np.random.RandomState(random_state)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    significant = (ci_lower > 0) or (ci_upper < 0)

    return {
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "n_pairs": n,
    }


def compute_model_zoo_uncertainty(
    model_zoo_bench: pd.DataFrame,
) -> pd.DataFrame:
    """Compute CI for all model zoo configurations.

    Args:
        model_zoo_bench: benchmark DataFrame from M1 (must have fold_spearmans column)

    Returns:
        DataFrame with CI columns for each metric
    """
    rows = []
    for _, row in model_zoo_bench.iterrows():
        ci_sp = bootstrap_ci(row["fold_spearmans"])
        ci_mae = bootstrap_ci(row["fold_maes"])
        ci_rmse = bootstrap_ci(row["fold_rmses"])

        rows.append({
            "endpoint": row["endpoint"],
            "model_name": row["model_name"],
            "feature_set": row["feature_set"],
            "spearman_mean": ci_sp["mean"],
            "spearman_ci_lower": ci_sp["ci_lower"],
            "spearman_ci_upper": ci_sp["ci_upper"],
            "mae_mean": ci_mae["mean"],
            "mae_ci_lower": ci_mae["ci_lower"],
            "mae_ci_upper": ci_mae["ci_upper"],
            "rmse_mean": ci_rmse["mean"],
            "rmse_ci_lower": ci_rmse["ci_lower"],
            "rmse_ci_upper": ci_rmse["ci_upper"],
        })

    return pd.DataFrame(rows)


def compute_pairwise_deltas(
    model_zoo_bench: pd.DataFrame,
    baseline_model: str = "ridge",
    baseline_feature_set: str = "design_one_hot",
) -> pd.DataFrame:
    """Compute paired delta CI for all configs vs baseline.

    Args:
        model_zoo_bench: benchmark DataFrame from M1
        baseline_model: baseline model name
        baseline_feature_set: baseline feature set name

    Returns:
        DataFrame with delta CI for each (endpoint, model, feature_set) vs baseline
    """
    rows = []
    for endpoint in model_zoo_bench["endpoint"].unique():
        ep_data = model_zoo_bench[model_zoo_bench["endpoint"] == endpoint]
        baseline_row = ep_data[
            (ep_data["model_name"] == baseline_model) &
            (ep_data["feature_set"] == baseline_feature_set)
        ]
        if len(baseline_row) == 0:
            continue
        baseline_spearmans = baseline_row.iloc[0]["fold_spearmans"]

        for _, row in ep_data.iterrows():
            result = paired_delta_ci(row["fold_spearmans"], baseline_spearmans)
            rows.append({
                "endpoint": endpoint,
                "model_name": row["model_name"],
                "feature_set": row["feature_set"],
                "delta_mean": result["delta_mean"],
                "delta_ci_lower": result["ci_lower"],
                "delta_ci_upper": result["ci_upper"],
                "significant": result["significant"],
                "n_pairs": result["n_pairs"],
                "comparison_baseline": f"{baseline_model}+{baseline_feature_set}",
            })

    return pd.DataFrame(rows)


def run_uncertainty_analysis(
    model_zoo_bench: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full uncertainty analysis.

    Returns:
        (ci_df, delta_df) tuple
    """
    ci_df = compute_model_zoo_uncertainty(model_zoo_bench)
    delta_df = compute_pairwise_deltas(model_zoo_bench)
    return ci_df, delta_df
