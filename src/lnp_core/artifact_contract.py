"""Artifact contract enforcement for v5.3.

Ensures that all required output artifacts exist and are non-empty
after the pipeline has run.
"""

from __future__ import annotations

from pathlib import Path

from lnp_core.paths import (
    VERSION_ROOT,
    DATA_DERIVED_DIR,
    BENCHMARK_DIR,
    FIGURES_DIR,
    DATA_MANIFEST_DIR,
    RESULTS_CANDIDATES_DIR,
    DERIVED_INTERIM_DATASET,
    BENCHMARK_LEAVE_TEMPLATE_OUT,
    DERIVED_CONFORMAL_PREDICTIONS,
    MISSING_NP3_CELLS,
    OUT_MANIFEST,
)

REQUIRED_DIRS: list[tuple[str, Path]] = [
    ("Data/derived", DATA_DERIVED_DIR),
    ("Results/benchmarks", BENCHMARK_DIR),
    ("Results/figures", FIGURES_DIR),
    ("Data/manifests", DATA_MANIFEST_DIR),
    ("Results/candidates", RESULTS_CANDIDATES_DIR),
]

REQUIRED_FILES: list[tuple[str, Path]] = [
    ("interim_dataset_v5_3.csv", DERIVED_INTERIM_DATASET),
    ("benchmark_leave_template_out_v5_3.csv", BENCHMARK_LEAVE_TEMPLATE_OUT),
    ("conformal_predictions_v5_3.csv", DERIVED_CONFORMAL_PREDICTIONS),
    ("missing_np3_design_cells_prioritized_v5_3.csv", MISSING_NP3_CELLS),
    ("run_manifest_v5_3.json", OUT_MANIFEST),
]


def check_artifact_contract() -> dict[str, str]:
    """Check all required artifacts exist and are non-empty.

    Returns:
        dict mapping artifact_name to "pass" or "fail: <reason>"
    """
    results: dict[str, str] = {}

    for name, dir_path in REQUIRED_DIRS:
        if not dir_path.exists():
            results[name] = f"fail: directory missing ({dir_path})"
        else:
            files = list(dir_path.iterdir())
            if not files:
                results[name] = f"fail: directory empty ({dir_path})"
            else:
                results[name] = "pass"

    for name, file_path in REQUIRED_FILES:
        if not file_path.exists():
            results[name] = f"fail: file missing ({file_path})"
        elif file_path.stat().st_size == 0:
            results[name] = f"fail: file empty ({file_path})"
        elif file_path.suffix == ".csv":
            with open(file_path) as f:
                lines = f.readlines()
            if len(lines) < 2:
                results[name] = f"fail: CSV has no data rows ({file_path})"
            else:
                results[name] = "pass"
        else:
            results[name] = "pass"

    return results


def enforce_artifact_contract() -> None:
    """Run artifact contract check and raise on any failure."""
    results = check_artifact_contract()
    failures = {k: v for k, v in results.items() if v != "pass"}
    if failures:
        msg = "\n".join(f"  {k}: {v}" for k, v in failures.items())
        raise RuntimeError(f"Artifact contract violations:\n{msg}")
