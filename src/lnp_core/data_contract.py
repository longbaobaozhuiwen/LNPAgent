"""Data contract: loading, cleaning, structural keys, and split matrix for v5.1.

Extends v5.0 with endpoint direction constants and targeting placeholder columns.
"""

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

PROPERTY_COLS = ["size", "pdi", "zeta_potential", "encapsulation_efficiency"]
ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b"]
ENDPOINT_TX_COL = "transfection_efficiency"
DESIGN_NUMERIC_COLS = ["ratio1", "ratio2", "ratio3", "ratio4", "np_ratio", "aq_org_ratio"]
ALL_ENDPOINT_COLS = ["immune_signal_a", "immune_signal_b", "tx_log1p"]

# v5.1 new: Endpoint direction constants
ENDPOINT_DIRECTION: dict[str, str] = {
    "immune_signal_a": "lower_is_better",
    "immune_signal_b": "lower_is_better",
    "tx_log1p": "higher_is_better",
    # Targeting (pre-embedded, not trained)
    "apoe": "lower_is_better",
    "apoa1": "higher_is_better",
}

# v5.1 new: Targeting placeholder columns
TARGETING_COLS = ["apoe", "apoa1"]


def load_and_clean(source_path: Path | str) -> pd.DataFrame:
    """Load SoT CSV, clean, validate, compute structural keys.

    v5.1 changes from v5.0:
    - Adds targeting placeholder columns (apoe, apoa1) if not present, filled with NaN.
    """
    source_path = Path(source_path)
    df = pd.read_csv(source_path)

    # Drop unnamed / empty trailing columns
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
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

    # Structural keys
    df["template_key"] = df["lipid1"] + "_" + df["lipid2"] + "_" + df["lipid4"]
    df["design_cell_key"] = (
        df["template_key"] + "_np" + df["np_ratio"].astype(int).astype(str)
    )

    # ladder_step_index: 0-based rank of ratio1 within each design_cell_key
    df["ladder_step_index"] = (
        df.groupby("design_cell_key")["ratio1"]
        .rank(method="dense", ascending=True)
        .astype(int)
        - 1
    )

    # Derived endpoint columns
    df["tx_raw"] = df["transfection_efficiency"]
    df["tx_log1p"] = np.log1p(df["transfection_efficiency"])

    # v5.1 new: Targeting placeholder columns
    for col in TARGETING_COLS:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Structural assertions
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert df["template_key"].nunique() == 12, (
        f"Expected 12 templates, got {df['template_key'].nunique()}"
    )
    assert df["design_cell_key"].nunique() == 20, (
        f"Expected 20 design cells, got {df['design_cell_key'].nunique()}"
    )

    return df


def build_split_matrix(
    df: pd.DataFrame,
    n_outer_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build leave-template-out cross-validation split matrix.

    Assigns each of the 12 templates to one of n_outer_folds folds
    in round-robin fashion after shuffling with seed=random_state.
    For each fold, rows belonging to that fold's templates are 'test',
    all others are 'train'.
    """
    rng = np.random.RandomState(random_state)
    templates = sorted(df["template_key"].unique())
    rng.shuffle(templates)

    # Assign templates to folds in round-robin
    template_to_fold = {}
    for i, tk in enumerate(templates):
        template_to_fold[tk] = i % n_outer_folds

    # Build per-fold split rows
    rows = []
    for fold_id in range(n_outer_folds):
        for _, row in df.iterrows():
            tk = row["template_key"]
            assigned_fold = template_to_fold[tk]
            split_role = "test" if assigned_fold == fold_id else "train"
            rows.append({
                "row_index": row.name,
                "fold_id": fold_id,
                "template_key": tk,
                "design_cell_key": row["design_cell_key"],
                "split_role": split_role,
            })

    return pd.DataFrame(rows)


def build_nested_inner_splits(
    train_df: pd.DataFrame,
    outer_fold_id: int,
    n_inner_folds: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build inner fold assignments within a single outer fold's train set."""
    inner_seed = random_state + outer_fold_id + 1
    rng = np.random.RandomState(inner_seed)

    templates = sorted(train_df["template_key"].unique())
    rng.shuffle(templates)

    template_to_inner = {}
    for i, tk in enumerate(templates):
        template_to_inner[tk] = i % n_inner_folds

    rows = []
    for idx, row in train_df.iterrows():
        tk = row["template_key"]
        inner_fold = template_to_inner[tk]
        rows.append({
            "row_index": idx,
            "inner_fold_id": inner_fold,
            "outer_fold_id": outer_fold_id,
        })

    return pd.DataFrame(rows)


def validate_no_leakage(
    split_df: pd.DataFrame,
    df: pd.DataFrame,
    fold_id: int,
    group_col: str = "template_key",
) -> bool:
    """Assert no same-group appears in both train and test for given fold."""
    fold_split = split_df[split_df["fold_id"] == fold_id]
    train_groups = set(
        fold_split[fold_split["split_role"] == "train"]["template_key"].unique()
    )
    test_groups = set(
        fold_split[fold_split["split_role"] == "test"]["template_key"].unique()
    )
    return len(train_groups & test_groups) == 0
