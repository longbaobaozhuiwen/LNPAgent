"""Deployable baseline: design-only features + leave-template-out CV for v5.0."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats as sp_stats

DEPLOYABLE_ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]


def build_deployable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build design-only feature matrix.

    Features:
    - lipid1 one-hot (2 cols), lipid2 one-hot (3 cols), lipid4 one-hot (2 cols)
    - ratio1-4, np_ratio, aq_org_ratio (numeric)
    - ratio1^2, ratio1^3 (polynomial)
    No lipid3 (always Chol), no template_key, no ladder_step_index.
    """
    parts = []

    # One-hot for lipid1, lipid2, lipid4
    for col in ["lipid1", "lipid2", "lipid4"]:
        dummies = pd.get_dummies(df[col], prefix=f"feat_comp_{col}", dtype=float)
        parts.append(dummies)

    # Numeric features
    for col in ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"]:
        parts.append(pd.DataFrame({f"feat_{col}": df[col].astype(float).values}, index=df.index))

    # Polynomial features
    parts.append(pd.DataFrame({
        "feat_ratio1_sq": (df["ratio1"].astype(float) ** 2).values,
        "feat_ratio1_cb": (df["ratio1"].astype(float) ** 3).values,
    }, index=df.index))

    result = pd.concat(parts, axis=1)
    result.index = df.index
    return result


def _compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation. NaN if < 3 valid pairs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    corr, _ = sp_stats.spearmanr(y_true[mask], y_pred[mask])
    return float(corr) if np.isfinite(corr) else np.nan


def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def run_deployable_baseline(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    alpha: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run deployable baseline: design-only features for 3 endpoints."""
    features = build_deployable_features(df)
    all_folds = sorted(split_df["fold_id"].unique())

    rows = []
    for endpoint in DEPLOYABLE_ENDPOINT_COLS:
        fold_metrics = []
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

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = Ridge(alpha=alpha, random_state=random_state)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            fold_metrics.append({
                "spearman": _compute_spearman(y_test, y_pred),
                "mae": _compute_mae(y_test, y_pred),
                "rmse": _compute_rmse(y_test, y_pred),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })

        if not fold_metrics:
            continue

        df_metrics = pd.DataFrame(fold_metrics)
        rows.append({
            "endpoint": endpoint,
            "model_name": "design_only",
            "deployable": True,
            "n_folds_completed": len(fold_metrics),
            "mean_spearman": df_metrics["spearman"].mean(),
            "std_spearman": df_metrics["spearman"].std(),
            "mean_mae": df_metrics["mae"].mean(),
            "mean_rmse": df_metrics["rmse"].mean(),
            "n_train_mean": df_metrics["n_train"].mean(),
            "n_test_mean": df_metrics["n_test"].mean(),
        })

    return pd.DataFrame(rows)


def generate_oof_predictions(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    endpoint: str,
    alpha: float = 1.0,
    random_state: int = 42,
) -> pd.Series:
    """Generate OOF endpoint predictions for a single endpoint."""
    features = build_deployable_features(df)
    all_folds = sorted(split_df["fold_id"].unique())

    predictions = pd.Series(np.nan, index=df.index, name=f"pred_{endpoint}")

    for fold_id in all_folds:
        fold_split = split_df[split_df["fold_id"] == fold_id]
        train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
        test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values

        if len(test_idx) == 0:
            continue

        X_train = features.loc[train_idx].values
        X_test = features.loc[test_idx].values
        y_train = df.loc[train_idx, endpoint].values.astype(float)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train_s, y_train)
        predictions.loc[test_idx] = model.predict(X_test_s)

    return predictions
