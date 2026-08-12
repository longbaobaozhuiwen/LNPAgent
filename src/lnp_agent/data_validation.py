"""Validation and schema detection for publicly supported input tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LNPDB_REQUIRED_COLUMNS = frozenset(
    {
        "LNP_ID",
        "IL_name",
        "IL_SMILES",
        "Model",
        "Experiment_method",
        "Experiment_value",
        "Publication_link",
    }
)
NATIVE_REQUIRED_COLUMNS = frozenset(
    {
        "lipid1",
        "lipid2",
        "lipid3",
        "lipid4",
        "ratio1",
        "ratio2",
        "ratio3",
        "ratio4",
        "transfection_efficiency",
        "immune_signal_a",
        "immune_signal_b",
    }
)


@dataclass(frozen=True)
class DatasetSummary:
    schema: str
    rows: int
    columns: int


def detect_schema(columns: set[str]) -> str:
    """Return the supported table schema or raise a focused validation error."""
    if LNPDB_REQUIRED_COLUMNS.issubset(columns):
        return "lnpdb"
    if NATIVE_REQUIRED_COLUMNS.issubset(columns):
        return "lnpagent-native"

    missing_lnpdb = sorted(LNPDB_REQUIRED_COLUMNS - columns)
    missing_native = sorted(NATIVE_REQUIRED_COLUMNS - columns)
    raise ValueError(
        "Unsupported CSV schema. Expected either the public LNPDB schema "
        f"(missing: {', '.join(missing_lnpdb)}) or the LNPAgent-native schema "
        f"(missing: {', '.join(missing_native)})."
    )


def validate_csv(path: Path | str) -> DatasetSummary:
    """Read a CSV, identify its supported schema, and apply safe invariants."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {source}")

    data = pd.read_csv(source)
    if data.empty:
        raise ValueError(f"CSV file contains no rows: {source}")

    schema = detect_schema(set(data.columns))
    if schema == "lnpdb":
        if data["LNP_ID"].isna().all():
            raise ValueError("LNPDB input must contain at least one LNP_ID.")
        if data["Experiment_value"].notna().sum() == 0:
            raise ValueError("LNPDB input must contain at least one numeric Experiment_value.")
    else:
        ratios = data[["ratio1", "ratio2", "ratio3", "ratio4"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if ratios.isna().any().any():
            raise ValueError("LNPAgent-native ratio columns must be numeric.")

    return DatasetSummary(schema=schema, rows=len(data), columns=len(data.columns))
