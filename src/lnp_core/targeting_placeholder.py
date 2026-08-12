"""Targeting data placeholder for v5.3.

Creates assay template for mouse blood measurements (ApoE, ApoA-I)
and optionally joins targeting data if provided.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGETING_ASSAY_COLUMNS = [
    "Formulation_ID", "apoe", "apoa1",
    "time_point", "batch", "replicate",
]


def create_targeting_assay_template(df: pd.DataFrame) -> pd.DataFrame:
    """Create targeting assay template for mouse blood measurements.

    Returns:
        DataFrame with Formulation_ID column populated, all other columns NaN.
    """
    template = pd.DataFrame({"Formulation_ID": df["Formulation_ID"].values})
    for col in TARGETING_ASSAY_COLUMNS:
        if col != "Formulation_ID":
            template[col] = float("nan")
    return template


def join_targeting_data(
    df: pd.DataFrame,
    targeting_path: Path | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Optionally join targeting data if it exists.

    Args:
        df: Main DataFrame with Formulation_ID
        targeting_path: Path to targeting data CSV (optional)

    Returns:
        (targeting_joined_df or None, status_dict)
        status_dict: {"status": "skipped"|"joined", "n_rows": int, "n_missing": int}
    """
    if targeting_path is None or not Path(targeting_path).exists():
        return None, {"status": "skipped", "n_rows": 0, "n_missing": 0}

    targeting_df = pd.read_csv(targeting_path)
    joined = df.merge(targeting_df, on="Formulation_ID", how="left")

    n_missing = 0
    for col in ["apoe", "apoa1"]:
        if col in joined.columns:
            n_missing += int(joined[col].isna().sum())

    return joined, {
        "status": "joined",
        "n_rows": len(targeting_df),
        "n_missing": n_missing,
    }
