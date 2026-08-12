"""Missing design cells enumeration and conservative prioritization for v5.3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lnp_core.feature_engineering import (
    KNOWN_LIPID1, KNOWN_LIPID2, KNOWN_LIPID4,
    RATIO1_LADDER, RATIO4_LADDER, RATIO2_FIXED,
    design_one_hot_v5_3, design_morgan_fp_weighted_v5_3,
)
from lnp_core.model_evaluation import V53_MODEL_CONFIGS, build_feature_set_v5_3

LIPID3_FIXED = "Chol"
NP_RATIO_OPTIONS = [3, 6]
ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]


def enumerate_full_design_grid() -> pd.DataFrame:
    """Enumerate all cells in the full design grid.

    Grid: 2 lipid1 x 3 lipid2 x 2 lipid4 x 2 np_ratio x 5 steps = 120 rows.
    Each step: (ratio1, ratio4) paired, ratio3 = 100 - ratio1 - 10 - ratio4.

    Returns:
        DataFrame with all 120 grid cells.
    """
    rows = []
    for l1 in KNOWN_LIPID1:
        for l2 in KNOWN_LIPID2:
            for l4 in KNOWN_LIPID4:
                for npr in NP_RATIO_OPTIONS:
                    for r1, r4 in zip(RATIO1_LADDER, RATIO4_LADDER):
                        r3 = 100.0 - r1 - RATIO2_FIXED - r4
                        rows.append({
                            "lipid1": l1, "lipid2": l2,
                            "lipid3": LIPID3_FIXED, "lipid4": l4,
                            "ratio1": r1, "ratio2": RATIO2_FIXED,
                            "ratio3": r3, "ratio4": r4,
                            "np_ratio": float(npr), "aq_org_ratio": 3.0,
                            "template_key": f"{l1}_{l2}_{l4}",
                            "design_cell_key": f"{l1}_{l2}_{l4}_np{npr}",
                        })
    return pd.DataFrame(rows)


def find_missing_cells(
    full_grid: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify cells present in full_grid but absent in observed_df.

    Uses design_cell_key + ratio1 as composite key.

    Returns:
        DataFrame of missing cells.
    """
    observed_keys = set(
        zip(observed_df["design_cell_key"], observed_df["ratio1"])
    )
    full_keys = list(zip(full_grid["design_cell_key"], full_grid["ratio1"]))
    mask = [key not in observed_keys for key in full_keys]
    return full_grid[mask].reset_index(drop=True)


def prioritize_missing_cells(
    missing_cells: pd.DataFrame,
    conformal_df: pd.DataFrame,
    best_configs: dict[str, tuple[str | None, str | None]],
    df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank missing cells by conservative strategy.

    For each missing cell:
      1. Build feature vector from design parameters
      2. Predict using best model (trained on full observed data)
      3. Apply conformal quantile from calibration residuals

    Conservative ranking: pi_high immune_signal_a/immune_signal_b ASC, pi_low tx DESC
    """
    results = []

    for _, cell in missing_cells.iterrows():
        row_result = {
            "template_key": cell["template_key"],
            "lipid1": cell["lipid1"], "lipid2": cell["lipid2"],
            "lipid3": cell["lipid3"], "lipid4": cell["lipid4"],
            "ratio1": cell["ratio1"], "ratio2": cell["ratio2"],
            "ratio3": cell["ratio3"], "ratio4": cell["ratio4"],
            "np_ratio": cell["np_ratio"], "aq_org_ratio": cell["aq_org_ratio"],
        }

        for endpoint in ENDPOINT_COLS:
            best_model, best_fs = best_configs.get(endpoint, (None, None))
            if best_model is None:
                row_result[f"y_hat_{endpoint}"] = np.nan
                row_result[f"pi_low_{endpoint}"] = np.nan
                row_result[f"pi_high_{endpoint}"] = np.nan
                row_result[f"config_{endpoint}"] = "none"
                continue

            config = V53_MODEL_CONFIGS[best_model]
            features = build_feature_set_v5_3(df, best_fs)

            # Train on full observed data
            X_all = features.values
            y_all = df[endpoint].values.astype(float)

            if config["needs_scaling"]:
                scaler = StandardScaler()
                X_all_s = scaler.fit_transform(X_all)
            else:
                X_all_s = X_all
                scaler = None

            model = config["model_class"](**config["fixed_params"])
            model.fit(X_all_s, y_all)

            # Build feature vector for the missing cell
            cell_df = pd.DataFrame([cell])
            # Add SMILES columns from df (use first matching template)
            template_match = df[df["template_key"] == cell["template_key"]]
            if len(template_match) > 0:
                for sm_col in ["lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles"]:
                    cell_df[sm_col] = template_match.iloc[0][sm_col]
            else:
                for sm_col in ["lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles"]:
                    cell_df[sm_col] = ""

            cell_features = build_feature_set_v5_3(cell_df, best_fs)
            # Ensure same columns
            missing_cols = set(features.columns) - set(cell_features.columns)
            for c in missing_cols:
                cell_features[c] = 0.0
            cell_features = cell_features[features.columns]

            X_cell = cell_features.values
            if scaler is not None:
                X_cell = scaler.transform(X_cell)

            y_hat = float(model.predict(X_cell)[0])

            # Compute calibration residual quantile from OOF
            all_folds = sorted(split_df["fold_id"].unique())
            oof_residuals = []
            for fold_id in all_folds:
                fold_split = split_df[split_df["fold_id"] == fold_id]
                train_idx = fold_split[fold_split["split_role"] == "train"]["row_index"].values
                test_idx = fold_split[fold_split["split_role"] == "test"]["row_index"].values
                if len(test_idx) == 0:
                    continue
                X_tr = features.loc[train_idx].values
                X_te = features.loc[test_idx].values
                y_tr = df.loc[train_idx, endpoint].values.astype(float)
                y_te = df.loc[test_idx, endpoint].values.astype(float)

                if config["needs_scaling"]:
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_tr)
                    X_te = sc.transform(X_te)

                mdl = config["model_class"](**config["fixed_params"])
                mdl.fit(X_tr, y_tr)
                y_pred = mdl.predict(X_te)
                oof_residuals.extend(np.abs(y_te - y_pred).tolist())

            if oof_residuals:
                q_value = float(np.percentile(oof_residuals, 90))
            else:
                q_value = np.nan

            row_result[f"y_hat_{endpoint}"] = y_hat
            row_result[f"pi_low_{endpoint}"] = y_hat - q_value
            row_result[f"pi_high_{endpoint}"] = y_hat + q_value
            row_result[f"config_{endpoint}"] = f"{best_model}+{best_fs}"

        results.append(row_result)

    result_df = pd.DataFrame(results)

    # Conservative sort: pi_high_immune_signal_a ASC, pi_high_immune_signal_b ASC, pi_low_tx_log1p DESC
    if len(result_df) > 0:
        result_df = result_df.sort_values(
            by=["pi_high_immune_signal_a", "pi_high_immune_signal_b", "pi_low_tx_log1p"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        result_df["rank"] = range(1, len(result_df) + 1)

    return result_df
