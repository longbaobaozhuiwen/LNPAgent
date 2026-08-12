"""Data cleaning for v5.3: explicit cleaning log, ratio validation, outlier flagging."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    "Formulation_ID", "LNP_ID",
    "lipid1", "lipid2", "lipid3", "lipid4",
    "lipid1_smiles", "lipid2_smiles", "lipid3_smiles", "lipid4_smiles",
    "ratio1", "ratio2", "ratio3", "ratio4",
    "np_ratio", "aq_org_ratio",
    "size", "pdi", "zeta_potential", "encapsulation_efficiency",
    "transfection_efficiency", "immune_signal_a", "immune_signal_b",
]

DESIGN_NUMERIC_COLS = ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"]
PROPERTY_COLS = ["size", "pdi", "zeta_potential", "encapsulation_efficiency"]
ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b"]
TARGETING_COLS = ["apoe", "apoa1"]


def load_and_clean_v5_3(source_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SoT CSV, clean, validate, compute structural keys. v5.3 version.

    Returns:
        (cleaned_df, cleaning_log_df) where cleaning_log has columns:
            check_name, affected_rows, action_taken, details
    """
    source_path = Path(source_path)
    df = pd.read_csv(source_path)
    log_rows = []

    # Drop unnamed / empty trailing columns
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Validate expected columns
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Coerce numeric columns to float
    numeric_cols = DESIGN_NUMERIC_COLS + PROPERTY_COLS + [
        "transfection_efficiency", "immune_signal_a", "immune_signal_b",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Missing value audit: design columns
    for col in DESIGN_NUMERIC_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            log_rows.append({
                "check_name": "missing_values",
                "affected_rows": int(n_missing),
                "action_taken": "raise",
                "details": f"Design column {col} has {n_missing} NaN values",
            })

    # Missing value audit: endpoint columns
    for col in ENDPOINT_COLS:
        n_missing = df[col].isna().sum()
        log_rows.append({
            "check_name": "missing_values",
            "affected_rows": int(n_missing),
            "action_taken": "logged_not_dropped",
            "details": f"Endpoint column {col} has {n_missing} NaN values",
        })

    # Ratio constraint validation
    ratio_sum = df["ratio1"] + df["ratio2"] + df["ratio3"] + df["ratio4"]
    violations = (ratio_sum - 100.0).abs() > 1.0
    n_violations = int(violations.sum())
    log_rows.append({
        "check_name": "ratio_constraint",
        "affected_rows": n_violations,
        "action_taken": "logged_flagged",
        "details": f"ratio1+2+3+4 deviation > 1.0: {n_violations} rows",
    })

    # Outlier flagging (IQR 1.5x rule for endpoints + tx)
    outlier_cols = ENDPOINT_COLS + ["transfection_efficiency"]
    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        log_rows.append({
            "check_name": "outlier_flag",
            "affected_rows": n_outliers,
            "action_taken": "logged_flagged",
            "details": f"{col}: {n_outliers} outliers (IQR 1.5x rule, range [{lower:.2f}, {upper:.2f}])",
        })

    # Structural keys
    df["template_key"] = df["lipid1"] + "_" + df["lipid2"] + "_" + df["lipid4"]
    df["design_cell_key"] = (
        df["template_key"] + "_np" + df["np_ratio"].astype(int).astype(str)
    )
    df["ladder_step_index"] = (
        df.groupby("design_cell_key")["ratio1"]
        .rank(method="dense", ascending=True)
        .astype(int) - 1
    )

    invalid_tx = df["transfection_efficiency"].notna() & (df["transfection_efficiency"] <= -1.0)
    if invalid_tx.any():
        bad_rows = df.index[invalid_tx].tolist()[:5]
        raise ValueError(
            "transfection_efficiency must be greater than -1 before log1p; "
            f"found {int(invalid_tx.sum())} invalid rows, first indices: {bad_rows}"
        )

    # Derived endpoint columns
    df["tx_raw"] = df["transfection_efficiency"]
    df["tx_log1p"] = np.log1p(df["transfection_efficiency"])

    # Targeting placeholders
    for col in TARGETING_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log_rows.append({
        "check_name": "dataset_profile",
        "affected_rows": int(len(df)),
        "action_taken": "logged_not_asserted",
        "details": (
            f"rows={len(df)}; templates={df['template_key'].nunique()}; "
            f"design_cells={df['design_cell_key'].nunique()}"
        ),
    })

    cleaning_log = pd.DataFrame(log_rows)
    return df, cleaning_log
