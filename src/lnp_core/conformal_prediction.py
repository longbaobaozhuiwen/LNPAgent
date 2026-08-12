"""Conformal CV+ prediction intervals for v5.3.

Implements per-sample prediction intervals using leave-template-out
cross-validation residuals as calibration set (CV+ method, Barber et al. 2021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lnp_core.model_evaluation import (
    V53_MODEL_CONFIGS,
    build_feature_set_v5_3,
)

ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]


def compute_cv_plus_intervals(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    endpoint: str,
    model_name: str,
    feature_set_name: str,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Compute CV+ conformal prediction intervals for one endpoint.

    Algorithm:
    1. For each fold k: train on train set, predict on test set, compute residuals
    2. For each sample i in fold k(i):
       - Collect residuals from other K-1 folds as calibration set
       - Compute conformal quantile
       - pi_low = y_hat - q, pi_high = y_hat + q

    Returns:
        DataFrame with per-sample intervals.
    """
    config = V53_MODEL_CONFIGS[model_name]
    features = build_feature_set_v5_3(df, feature_set_name)
    all_folds = sorted(split_df["fold_id"].unique())

    # Step 1: Compute OOF predictions and residuals per fold
    fold_data = {}
    for fold_id in all_folds:
        fold_split = split_df[split_df["fold_id"] == fold_id]
        train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
        test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values
        if len(test_idx) == 0:
            continue

        X_train = features.loc[train_idx].values
        X_test = features.loc[test_idx].values
        y_train = df.loc[train_idx, endpoint].values.astype(float)
        y_test = df.loc[test_idx, endpoint].values.astype(float)

        if config["needs_scaling"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = config["model_class"](**config["fixed_params"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = np.abs(y_test - y_pred)

        fold_data[fold_id] = {
            "indices": test_idx,
            "y_true": y_test,
            "y_pred": y_pred,
            "residuals": residuals,
        }

    # Step 2: Build per-sample prediction intervals
    results = []
    for fold_id in all_folds:
        if fold_id not in fold_data:
            continue
        fd = fold_data[fold_id]

        # Calibration residuals from OTHER folds
        calibration_residuals = []
        for other_fold_id in all_folds:
            if other_fold_id == fold_id or other_fold_id not in fold_data:
                continue
            calibration_residuals.extend(fold_data[other_fold_id]["residuals"].tolist())

        n_calib = len(calibration_residuals)
        if n_calib == 0:
            continue

        calibration_residuals = np.sort(calibration_residuals)
        q_index = int(np.ceil((1 - alpha) * (n_calib + 1))) - 1
        q_index = min(q_index, n_calib - 1)
        q_value = calibration_residuals[q_index]

        for i, idx in enumerate(fd["indices"]):
            source_template = df.loc[idx, "template_key"]
            results.append({
                "sample_index": int(idx),
                "endpoint": endpoint,
                "y_true": float(fd["y_true"][i]),
                "y_hat": float(fd["y_pred"][i]),
                "pi_low": float(fd["y_pred"][i] - q_value),
                "pi_high": float(fd["y_pred"][i] + q_value),
                "interval_width": float(2 * q_value),
                "fold_id": fold_id,
                "source_template": source_template,
                "model_name": model_name,
                "feature_set": feature_set_name,
                "n_calibration": n_calib,
            })

    return pd.DataFrame(results)


def compute_calibration_diagnostics(
    conformal_df: pd.DataFrame,
    endpoint: str,
    alpha: float = 0.10,
) -> dict:
    """Compute coverage and width statistics for conformal intervals."""
    ep_data = conformal_df[conformal_df["endpoint"] == endpoint]
    n = len(ep_data)
    if n == 0:
        return {"endpoint": endpoint, "nominal_coverage": 1 - alpha, "empirical_coverage": np.nan,
                "mean_width": np.nan, "median_width": np.nan, "n_samples": 0}

    covered = ((ep_data["y_true"] >= ep_data["pi_low"]) & (ep_data["y_true"] <= ep_data["pi_high"])).sum()
    return {
        "endpoint": endpoint,
        "nominal_coverage": round(1 - alpha, 2),
        "empirical_coverage": round(float(covered / n), 4),
        "mean_width": round(float(ep_data["interval_width"].mean()), 4),
        "median_width": round(float(ep_data["interval_width"].median()), 4),
        "std_width": round(float(ep_data["interval_width"].std()), 4),
        "min_width": round(float(ep_data["interval_width"].min()), 4),
        "max_width": round(float(ep_data["interval_width"].max()), 4),
        "n_samples": n,
    }


def run_conformal_all_endpoints(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    best_configs: dict[str, tuple[str | None, str | None]],
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run CV+ for all endpoints using their best configs.

    Returns:
        (conformal_df_wide, diagnostics_df)
        conformal_df_wide: one row per sample, columns for all 3 endpoints
        diagnostics_df: one row per endpoint
    """
    all_long = []
    diagnostics_rows = []

    for endpoint in ENDPOINT_COLS:
        best_model, best_fs = best_configs.get(endpoint, (None, None))
        if best_model is None:
            continue

        ep_conformal = compute_cv_plus_intervals(
            df, split_df, endpoint, best_model, best_fs, alpha,
        )
        all_long.append(ep_conformal)
        diagnostics_rows.append(compute_calibration_diagnostics(ep_conformal, endpoint, alpha))

    diagnostics_df = pd.DataFrame(diagnostics_rows)

    # Convert long format to wide format (one row per sample)
    if not all_long:
        return pd.DataFrame(), diagnostics_df

    long_df = pd.concat(all_long, ignore_index=True)

    # Pivot to wide
    wide_parts = []
    for endpoint in ENDPOINT_COLS:
        ep_data = long_df[long_df["endpoint"] == endpoint].set_index("sample_index")
        if len(ep_data) == 0:
            continue
        rename_map = {
            "y_true": f"y_true_{endpoint}",
            "y_hat": f"y_hat_{endpoint}",
            "pi_low": f"pi_low_{endpoint}",
            "pi_high": f"pi_high_{endpoint}",
            "interval_width": f"interval_width_{endpoint}",
            "fold_id": f"fold_{endpoint}",
            "source_template": f"template_{endpoint}",
        }
        ep_wide = ep_data[rename_map.keys()].rename(columns=rename_map)
        wide_parts.append(ep_wide)

    if not wide_parts:
        return pd.DataFrame(), diagnostics_df

    # Merge all endpoints on sample_index
    wide_df = wide_parts[0]
    for part in wide_parts[1:]:
        wide_df = wide_df.join(part, how="outer")

    wide_df = wide_df.sort_index().reset_index()
    return wide_df, diagnostics_df
