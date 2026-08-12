"""Unified model evaluation for v5.3: leave-template-out with HuberRegressor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from lnp_core.feature_engineering import (
    design_one_hot_v5_3,
    design_morgan_fp_weighted_v5_3,
)
from lnp_core.model_zoo import _compute_spearman_with_status

ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]

V53_MODEL_CONFIGS = {
    "ridge": {
        "model_class": Ridge,
        "fixed_params": {"alpha": 1.0, "random_state": 42},
        "tune_params": None,
        "needs_scaling": True,
    },
    "hgbr": {
        "model_class": HistGradientBoostingRegressor,
        "fixed_params": {
            "max_depth": 3, "learning_rate": 0.05,
            "max_iter": 500, "random_state": 42,
        },
        "tune_params": None,
        "needs_scaling": False,
    },
    "huber": {
        "model_class": HuberRegressor,
        "fixed_params": {"max_iter": 200},
        "tune_params": None,
        "needs_scaling": True,
    },
}

V53_FEATURE_SETS = {
    "design_one_hot": design_one_hot_v5_3,
    "design_morgan_fp_weighted": design_morgan_fp_weighted_v5_3,
}


def build_feature_set_v5_3(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """Build feature matrix using v5.3 feature sets."""
    if feature_set_name not in V53_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    return V53_FEATURE_SETS[feature_set_name](df)


def _run_single_config(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    endpoint: str,
    model_name: str,
    feature_set_name: str,
) -> dict:
    """Run single model config with validity-aware metrics."""
    config = V53_MODEL_CONFIGS[model_name]
    features = build_feature_set_v5_3(df, feature_set_name)
    all_folds = sorted(split_df["fold_id"].unique())

    fold_spearmans = []
    fold_spearman_status = []
    fold_maes = []
    fold_rmses = []
    n_folds_completed = 0

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

        corr, status = _compute_spearman_with_status(y_test, y_pred)
        fold_spearmans.append(corr)
        fold_spearman_status.append(status)
        fold_maes.append(float(np.mean(np.abs(y_test - y_pred))))
        fold_rmses.append(float(np.sqrt(np.mean((y_test - y_pred) ** 2))))
        n_folds_completed += 1

    n_valid = sum(1 for s in fold_spearman_status if s == "ok")
    valid_fraction = n_valid / n_folds_completed if n_folds_completed > 0 else 0.0

    return {
        "endpoint": endpoint,
        "model_name": model_name,
        "feature_set": feature_set_name,
        "deployable": True,
        "n_folds_completed": n_folds_completed,
        "n_valid_folds": n_valid,
        "valid_fraction": valid_fraction,
        "mean_spearman": float(np.nanmean(fold_spearmans)) if fold_spearmans else np.nan,
        "std_spearman": float(np.nanstd(fold_spearmans)) if fold_spearmans else np.nan,
        "mean_spearman_nanmean": float(np.nanmean(fold_spearmans)) if fold_spearmans else np.nan,
        "std_spearman_nanstd": float(np.nanstd(fold_spearmans)) if fold_spearmans else np.nan,
        "mean_mae": float(np.mean(fold_maes)) if fold_maes else np.nan,
        "mean_rmse": float(np.mean(fold_rmses)) if fold_rmses else np.nan,
        "fold_spearmans": fold_spearmans,
        "fold_spearman_status": fold_spearman_status,
        "fold_maes": fold_maes,
        "fold_rmses": fold_rmses,
        "invalid_for_selection": n_valid < 3,
    }


def run_leave_template_out_evaluation(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Unified evaluation: 3 endpoints x 3 models x 2 feature sets = 18 rows."""
    rows = []
    for endpoint in ENDPOINT_COLS:
        for feature_set_name in V53_FEATURE_SETS:
            for model_name in V53_MODEL_CONFIGS:
                result = _run_single_config(df, split_df, endpoint, model_name, feature_set_name)
                rows.append(result)
    return pd.DataFrame(rows)


def generate_oof_predictions_v5_3(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
) -> pd.DataFrame:
    """Generate OOF predictions for all 3 endpoints using specified config."""
    config = V53_MODEL_CONFIGS[model_name]
    features = build_feature_set_v5_3(df, feature_set_name)
    all_folds = sorted(split_df["fold_id"].unique())

    predictions = pd.DataFrame(index=df.index)
    for endpoint in ENDPOINT_COLS:
        oof_pred = np.full(len(df), np.nan)
        for fold_id in all_folds:
            fold_split = split_df[split_df["fold_id"] == fold_id]
            train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
            test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values
            if len(test_idx) == 0:
                continue

            X_train = features.loc[train_idx].values
            X_test = features.loc[test_idx].values
            y_train = df.loc[train_idx, endpoint].values.astype(float)

            if config["needs_scaling"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model = config["model_class"](**config["fixed_params"])
            model.fit(X_train, y_train)
            oof_pred[test_idx] = model.predict(X_test)

        predictions[f"pred_{endpoint}"] = oof_pred

    return predictions
