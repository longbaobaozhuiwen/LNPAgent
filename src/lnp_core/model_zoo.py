"""Deployable model zoo: Ridge + ElasticNet + HGBR × design_one_hot + design_morgan_fp for v5.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy import stats as sp_stats

from lnp_core.deployable_baseline import build_deployable_features
from lnp_core.chemical_features import build_morgan_fp_features

DEPLOYABLE_ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]

MODEL_CONFIGS = {
    "ridge": {
        "model_class": Ridge,
        "fixed_params": {"alpha": 1.0, "random_state": 42},
        "tune_params": None,
        "needs_scaling": True,
    },
    "elasticnet": {
        "model_class": ElasticNet,
        "fixed_params": {"random_state": 42, "max_iter": 10000},
        "tune_params": {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.1, 0.5]},
        "needs_scaling": True,
    },
    "hgbr": {
        "model_class": HistGradientBoostingRegressor,
        "fixed_params": {"max_depth": 3, "learning_rate": 0.05, "max_iter": 500, "random_state": 42},
        "tune_params": None,
        "needs_scaling": False,
    },
}

FEATURE_SETS = {
    "design_one_hot": build_deployable_features,
    "design_morgan_fp_weighted": build_morgan_fp_features,
}


def _compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


def build_feature_set(df: pd.DataFrame, feature_set_name: str) -> pd.DataFrame:
    """Build feature matrix by name."""
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    return FEATURE_SETS[feature_set_name](df)


def run_single_model(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    endpoint: str,
    model_name: str,
    feature_set_name: str,
) -> dict:
    """Run a single model config under leave-template-out CV."""
    config = MODEL_CONFIGS[model_name]
    features = build_feature_set(df, feature_set_name)
    all_folds = sorted(split_df["fold_id"].unique())

    fold_spearmans = []
    fold_maes = []
    fold_rmses = []
    n_train_list = []
    n_test_list = []

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

        # Scaling
        if config["needs_scaling"]:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
        else:
            X_train_s = X_train
            X_test_s = X_test

        # Model fitting (with optional hyperparameter tuning)
        if config["tune_params"] is not None:
            base_params = config["fixed_params"].copy()
            grid = GridSearchCV(
                config["model_class"](**base_params),
                config["tune_params"],
                cv=3,
                scoring="neg_mean_squared_error",
            )
            grid.fit(X_train_s, y_train)
            model = config["model_class"](**{**base_params, **grid.best_params_})
            model.fit(X_train_s, y_train)
        else:
            model = config["model_class"](**config["fixed_params"])
            model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        fold_spearmans.append(_compute_spearman(y_test, y_pred))
        fold_maes.append(_compute_mae(y_test, y_pred))
        fold_rmses.append(_compute_rmse(y_test, y_pred))
        n_train_list.append(len(train_idx))
        n_test_list.append(len(test_idx))

    return {
        "endpoint": endpoint,
        "model_name": model_name,
        "feature_set": feature_set_name,
        "deployable": True,
        "n_folds_completed": len(fold_spearmans),
        "mean_spearman": np.mean(fold_spearmans) if fold_spearmans else np.nan,
        "std_spearman": np.std(fold_spearmans) if fold_spearmans else np.nan,
        "mean_mae": np.mean(fold_maes) if fold_maes else np.nan,
        "mean_rmse": np.mean(fold_rmses) if fold_rmses else np.nan,
        "fold_spearmans": fold_spearmans,
        "fold_maes": fold_maes,
        "fold_rmses": fold_rmses,
        "n_train_mean": np.mean(n_train_list) if n_train_list else np.nan,
        "n_test_mean": np.mean(n_test_list) if n_test_list else np.nan,
    }


def run_model_zoo(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run full model zoo: 3 endpoints × 3 models × 2 feature sets = 18 rows."""
    rows = []
    for endpoint in DEPLOYABLE_ENDPOINT_COLS:
        for feature_set_name in FEATURE_SETS:
            for model_name in MODEL_CONFIGS:
                result = run_single_model(df, split_df, endpoint, model_name, feature_set_name)
                rows.append(result)
    return pd.DataFrame(rows)


def generate_model_zoo_oof_predictions(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    model_name: str,
    feature_set_name: str,
) -> pd.DataFrame:
    """Generate OOF endpoint predictions for a specific model/feature_set config."""
    config = MODEL_CONFIGS[model_name]
    features = build_feature_set(df, feature_set_name)
    all_folds = sorted(split_df["fold_id"].unique())

    predictions = pd.DataFrame(index=df.index)
    for endpoint in DEPLOYABLE_ENDPOINT_COLS:
        preds = pd.Series(np.nan, index=df.index, name=f"pred_{endpoint}")
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
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
            else:
                X_train_s = X_train
                X_test_s = X_test

            if config["tune_params"] is not None:
                base_params = config["fixed_params"].copy()
                grid = GridSearchCV(
                    config["model_class"](**base_params),
                    config["tune_params"],
                    cv=3,
                    scoring="neg_mean_squared_error",
                )
                grid.fit(X_train_s, y_train)
                model = config["model_class"](**{**base_params, **grid.best_params_})
                model.fit(X_train_s, y_train)
            else:
                model = config["model_class"](**config["fixed_params"])
                model.fit(X_train_s, y_train)

            preds.loc[test_idx] = model.predict(X_test_s)

        predictions[f"pred_{endpoint}"] = preds

    return predictions
