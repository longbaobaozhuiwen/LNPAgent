"""Create the versioned public LNPDB example from an official LNPDB CSV.

This utility selects 100 rows and a compact set of formulation, assay, and
provenance fields. It never reads or transforms an LNPAgent internal dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "LNP_ID",
    "Experiment_ID",
    "Formulation_ID",
    "IL_name",
    "IL_SMILES",
    "IL_molratio",
    "HL_name",
    "HL_molratio",
    "CHL_name",
    "CHL_molratio",
    "PEG_name",
    "PEG_molratio",
    "Model",
    "Model_type",
    "Cargo",
    "Cargo_type",
    "Experiment_method",
    "Experiment_value",
    "Publication_link",
    "Publication_PMID",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Official LNPDB.csv download")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "lnpdb_public_example.csv",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.source, usecols=COLUMNS)
    example = data.loc[
        (data["Model"] == "in_vitro")
        & (data["Experiment_method"] == "luminescence_normalized")
        & data["Experiment_value"].notna()
    ].head(100)
    if len(example) != 100:
        raise ValueError("Expected at least 100 public in-vitro luminescence rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
