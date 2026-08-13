"""Validation and schema detection for publicly supported input tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def summarize_public_lnpdb(path: Path | str) -> dict[str, Any]:
    """Return a provenance-preserving, endpoint-agnostic summary of public LNPDB data.

    LNPDB assay values are intentionally not renamed to LNPAgent's native
    ``immune_signal``/``transfection`` endpoints: those measurements are not
    scientifically interchangeable. This summary is therefore suitable for
    smoke tests and dataset inspection, not model benchmarking.
    """
    source = Path(path)
    summary = validate_csv(source)
    if summary.schema != "lnpdb":
        raise ValueError("Public LNPDB summary requires an LNPDB-schema CSV.")
    data = pd.read_csv(source)
    values = pd.to_numeric(data["Experiment_value"], errors="coerce")
    numeric = values.dropna()
    return {
        "schema": summary.schema,
        "rows": summary.rows,
        "columns": summary.columns,
        "unique_lnp_ids": int(data["LNP_ID"].nunique()),
        "unique_experiments": int(data["Experiment_ID"].nunique())
        if "Experiment_ID" in data.columns else 0,
        "experiment_methods": sorted(data["Experiment_method"].dropna().astype(str).unique()),
        "model_types": sorted(data["Model_type"].dropna().astype(str).unique()),
        "cargo_types": sorted(data["Cargo_type"].dropna().astype(str).unique()),
        "numeric_experiment_values": int(numeric.size),
        "experiment_value_min": float(numeric.min()) if not numeric.empty else None,
        "experiment_value_max": float(numeric.max()) if not numeric.empty else None,
        "scientific_use": "public schema inspection only; not an endpoint benchmark",
    }


def benchmark_public_lnpdb(path: Path | str, seed: int = 42) -> dict[str, Any]:
    """Build a reproducible public-data benchmark artifact.

    This is a data-access and assay-coverage benchmark, not a predictive model
    benchmark. It exists so public users can verify the package can load and
    summarize the bundled LNPDB subset without private data.
    """
    summary = summarize_public_lnpdb(path)
    from lnp_agent import __version__

    return {
        "benchmark_name": "public_lnpdb_schema_coverage",
        "benchmark_version": 1,
        "random_seed": int(seed),
        "source_path": str(Path(path)),
        "provenance": {
            "generated_by": "lnp-agent --benchmark-public",
            "package": "lnp-agent",
            "package_version": __version__,
            "artifact_schema": "public_lnpdb_benchmark.v1",
            "source_dataset": "LNPDB public example subset",
            "source_license": "MIT",
            "private_data_included": False,
        },
        "dataset": summary,
        "metrics": {
            "rows_loaded": summary["rows"],
            "unique_lnp_ids": summary["unique_lnp_ids"],
            "numeric_experiment_values": summary["numeric_experiment_values"],
            "unique_experiments": summary["unique_experiments"],
        },
        "scientific_use": (
            "reproducible public-data software benchmark only; not a model "
            "performance, biological efficacy, or candidate recommendation claim"
        ),
    }


def build_public_demo_round(
    path: Path | str,
    seed: int = 42,
    library_size: int = 24,
    batch_size: int = 6,
) -> dict[str, Any]:
    """Run a public-safe one-round acquisition demo from the LNPDB example.

    The generated prediction columns are synthetic software fixtures derived
    from public table structure. They are not endpoint relabels and not
    biological performance estimates.
    """
    source = Path(path)
    summary = validate_csv(source)
    if summary.schema != "lnpdb":
        raise ValueError("Public demo requires an LNPDB-schema CSV.")

    data = pd.read_csv(source)
    values = pd.to_numeric(data["Experiment_value"], errors="coerce")
    data = data.loc[values.notna()].copy()
    data["_experiment_value"] = values.loc[data.index].astype(float)
    n = min(max(int(library_size), 1), len(data))
    sampled = data.sample(n=n, random_state=int(seed)).reset_index(drop=True)

    def _scale(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").astype(float)
        span = numeric.max() - numeric.min()
        if span <= 1e-12:
            return pd.Series(0.5, index=series.index)
        return (numeric - numeric.min()) / span

    public_value_score = _scale(sampled["_experiment_value"])
    ratio_frame = sampled[
        ["IL_molratio", "HL_molratio", "CHL_molratio", "PEG_molratio"]
    ].apply(pd.to_numeric, errors="coerce")
    ratio_distance = (ratio_frame - ratio_frame.median(axis=0)).abs().mean(axis=1)
    novelty_proxy = _scale(ratio_distance)
    method_rarity = sampled["Experiment_method"].map(
        sampled["Experiment_method"].value_counts(dropna=False)
    )
    method_rarity = 1.0 - _scale(method_rarity)
    uncertainty = (0.65 * novelty_proxy + 0.35 * method_rarity).clip(0.05, 1.0)

    candidates = pd.DataFrame(
        {
            "Formulation_ID": sampled["Formulation_ID"].astype(str),
            "LNP_ID": sampled["LNP_ID"].astype(str),
            "template_key": (
                sampled["Experiment_method"].astype(str)
                + "|"
                + sampled["Model_type"].astype(str)
                + "|"
                + sampled["Cargo_type"].astype(str)
            ),
            "lipid1": sampled["IL_name"].astype(str),
            "lipid2": sampled["HL_name"].astype(str),
            "lipid3": sampled["CHL_name"].astype(str),
            "lipid4": sampled["PEG_name"].astype(str),
            "ratio1": ratio_frame["IL_molratio"],
            "ratio2": ratio_frame["HL_molratio"],
            "ratio3": ratio_frame["CHL_molratio"],
            "ratio4": ratio_frame["PEG_molratio"],
            "np_ratio": 0.0,
            "aq_org_ratio": 0.0,
            "public_assay_value": sampled["_experiment_value"],
            "pred_tx_log1p": public_value_score,
            "pred_immune_signal_a": 1.0 - public_value_score,
            "pred_immune_signal_b": (1.0 - public_value_score + novelty_proxy) / 2.0,
            "pred_uncertainty_tx_log1p": uncertainty,
            "pred_uncertainty_immune_signal_a": uncertainty,
            "pred_uncertainty_immune_signal_b": uncertainty,
        }
    )

    from lnp_core.candidate_ranking import (
        compute_experiment_value_scores,
        rank_candidates,
        select_experiment_value_batch,
    )

    ranked = rank_candidates(candidates)
    objective_weights = {
        "tx_log1p": 0.50,
        "immune_signal_a": 0.25,
        "immune_signal_b": 0.25,
    }
    scored_pool = compute_experiment_value_scores(
        ranked, objective_weights=objective_weights
    )
    selected = select_experiment_value_batch(
        ranked,
        batch_size=int(batch_size),
        objective_weights=objective_weights,
        objective_uncertainty_weight=0.20,
    )
    selected_public = pd.to_numeric(selected["public_assay_value"], errors="coerce")
    pool_public = pd.to_numeric(ranked["public_assay_value"], errors="coerce")
    score_public = pd.to_numeric(scored_pool["experiment_value_score"], errors="coerce")
    assay_rank_corr = ranked["public_assay_value"].corr(
        score_public, method="spearman"
    )
    rationale_counts = selected["selection_rationale"].value_counts().to_dict()
    selected_cols = [
        "rank",
        "Formulation_ID",
        "LNP_ID",
        "lipid1",
        "lipid2",
        "lipid3",
        "lipid4",
        "public_assay_value",
        "experiment_value_score",
        "mean_exploitation_score",
        "objective_uncertainty_bonus",
        "batch_selection_score",
        "exploitation_score",
        "exploration_score",
        "diversity_score",
        "batch_complementarity_score",
        "batch_redundancy_score",
        "batch_uncertainty_complementarity_score",
        "batch_uncertainty_redundancy_score",
        "selection_rationale",
    ]
    selected_cols = [c for c in selected_cols if c in selected.columns]

    return {
        "demo_name": "public_lnpdb_one_round_acquisition_demo",
        "demo_version": 1,
        "random_seed": int(seed),
        "provenance": {
            "generated_by": "lnp-agent --demo-public",
            "source_dataset": "LNPDB public example subset",
            "source_license": "MIT",
            "measurement_type": "synthetic_public_demo",
            "private_data_included": False,
        },
        "candidate_pool_size": int(len(ranked)),
        "selected_batch_size": int(len(selected)),
        "acquisition_policy": {
            "name": "greedy_experiment_value_with_batch_complementarity",
            "experiment_value_terms": [
                "exploitation",
                "exploration",
                "design_space_diversity",
            ],
            "objective_weights": objective_weights,
            "objective_uncertainty_weight": 0.20,
            "batch_level_term": (
                "balanced_formulation_and_uncertainty_profile_complementarity"
            ),
        },
        "selected_candidates": selected[selected_cols].to_dict(orient="records"),
        "diagnostics": {
            "diagnostic_type": "retrospective_public_demo_mechanics",
            "score_public_assay_spearman": None
            if pd.isna(assay_rank_corr) else float(assay_rank_corr),
            "selected_public_assay_mean": float(selected_public.mean()),
            "pool_public_assay_mean": float(pool_public.mean()),
            "selected_minus_pool_public_assay_mean": float(
                selected_public.mean() - pool_public.mean()
            ),
            "selected_unique_lipid1": int(selected["lipid1"].nunique()),
            "selected_mean_batch_complementarity": float(
                pd.to_numeric(
                    selected["batch_complementarity_score"], errors="coerce"
                ).mean()
            ),
            "selection_rationale_counts": {
                str(key): int(value) for key, value in rationale_counts.items()
            },
        },
        "scientific_use": (
            "public software demo for acquisition-policy mechanics only; "
            "synthetic predictions are not endpoint relabels, model performance, "
            "biological efficacy, or candidate recommendations"
        ),
    }
