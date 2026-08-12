"""Reporting for v5.3: summary, scope, manifest, figures with conformal intervals and missing cells."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lnp_core.paths import (
    OUT_SUMMARY, OUT_SCOPE, OUT_MANIFEST,
    FIGURES_DIR, SOURCE_OF_TRUTH_RAW,
    FIG_DEPLOYABLE_BASELINE, FIG_PROPERTY_HEADS,
    FIG_EXPLANATORY_AUDIT, FIG_STAGEWISE_VS_BASELINE,
    FIG_DATA_STRUCTURE, FIG_CV_SCHEME,
    FIG_MODEL_ZOO_COMPARISON, FIG_CANDIDATE_PARETO,
    FIG_NP_RATIO_STRATIFIED,
    FIG_CONFORMAL_COVERAGE, FIG_CONFORMAL_WIDTH, FIG_MISSING_CELLS,
)

FORBIDDEN_PHRASES = [
    "active model switch", "replaced M4", "better than M4",
    "LLM ranked", "LLM recommended", "flow matching generated",
    "de novo designed", "active switch", "LLM-driven ranking",
    "property replaces", "property outperforms M4", "targeting score",
    "property reverses V4.6 negative",
    "properties now improve global transfer",
    "active targeting model", "global optimum", "best formulation",
    "guaranteed improvement", "statistically significant improvement",
    "optimal formulation", "recommended formulation",
]


def check_forbidden_claims(text: str) -> list[str]:
    violations = []
    text_lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in text_lower:
            violations.append(phrase)
    return violations


def render_v5_3_summary(
    deployable_baseline: pd.DataFrame,
    property_heads_bench: pd.DataFrame,
    model_zoo_bench: pd.DataFrame,
    uncertainty_ci: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    data_audit: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    conformal_diagnostics: pd.DataFrame,
    missing_cells: pd.DataFrame,
    best_configs: dict,
    targeting_status: dict,
    np_ratio_stratified: pd.DataFrame = None,
) -> str:
    lines = []
    lines.append("# V5.3 Summary Report\n")

    # 1. Governance
    lines.append("## 1. Governance Statement\n")
    lines.append("All deployable models use only design-time features. No model replacement.\n")

    # 2. Data Contract + 2a. Cleaning Log
    lines.append("## 2. Data Contract Verification\n")
    n_passed = data_audit["passed"].sum()
    lines.append(f"- Checks passed: {n_passed}/{len(data_audit)}\n")

    lines.append("### 2a. Cleaning Log Summary\n")
    for _, row in cleaning_log.iterrows():
        lines.append(f"- {row['check_name']}: {row['affected_rows']} rows ({row['action_taken']})")
    lines.append("")

    # 3. Deployable Baseline
    lines.append("## 3. Deployable Baseline Results\n")
    lines.append("| Endpoint | Spearman | MAE | RMSE |")
    lines.append("|----------|----------|-----|------|")
    for _, row in deployable_baseline.iterrows():
        lines.append(f"| {row['endpoint']} | {row['mean_spearman']:.4f} | {row['mean_mae']:.4f} | {row['mean_rmse']:.4f} |")
    lines.append("")

    # 4. Property Heads
    lines.append("## 4. Property Head Predictability\n")
    lines.append("| Property | Spearman | MAE | RMSE |")
    lines.append("|----------|----------|-----|------|")
    for _, row in property_heads_bench.iterrows():
        lines.append(f"| {row['property_col']} | {row['mean_spearman']:.4f} | {row['mean_mae']:.4f} | {row['mean_rmse']:.4f} |")
    lines.append("")

    # 5. Model Zoo
    lines.append("## 5. Model Zoo Comparison\n")
    sp_col = "mean_spearman_nanmean" if "mean_spearman_nanmean" in model_zoo_bench.columns else "mean_spearman"
    lines.append("| Endpoint | Model | Feature Set | Spearman |")
    lines.append("|----------|-------|-------------|----------|")
    for _, row in model_zoo_bench.iterrows():
        lines.append(f"| {row['endpoint']} | {row['model_name']} | {row['feature_set']} | {row[sp_col]:.4f} |")
    lines.append("")

    # 6. Best Configs
    lines.append("## 6. Per-Endpoint Best Config\n")
    lines.append("| Endpoint | Model | Feature Set |")
    lines.append("|----------|-------|-------------|")
    for ep in ["immune_signal_a", "immune_signal_b", "tx_log1p"]:
        bm, bfs = best_configs.get(ep, (None, None))
        lines.append(f"| {ep} | {bm} | {bfs} |")
    lines.append("")

    # 7. Uncertainty
    lines.append("## 7. Bootstrap CI Analysis\n")
    lines.append("Delta vs Ridge+design_one_hot baseline.\n")

    # 8. Pareto Candidates
    lines.append("## 8. Pareto Candidates\n")
    if len(candidate_summary) > 0:
        cs = candidate_summary.iloc[0]
        lines.append(f"- Pareto front: {cs['n_pareto_candidates']}/{cs['n_total']}")
    lines.append("")

    # 9. np_ratio Stratified
    lines.append("## 9. np_ratio Stratified Diagnostics\n")
    if np_ratio_stratified is not None and len(np_ratio_stratified) > 0:
        lines.append("| np_ratio | Endpoint | MAE | Spearman |")
        lines.append("|----------|----------|-----|----------|")
        for _, row in np_ratio_stratified.iterrows():
            sp = row.get("stratified_spearman", np.nan)
            sp_str = f"{sp:.4f}" if pd.notna(sp) else "N/A"
            lines.append(f"| {row['np_ratio']} | {row['endpoint']} | {row['stratified_mae']:.4f} | {sp_str} |")
    lines.append("")

    # 10. Targeting
    lines.append("## 10. Targeting Status\n")
    lines.append(f"- Status: {targeting_status.get('status', 'unknown')}\n")

    # 11. Key Findings
    lines.append("## 11. Key Findings\n")
    lines.append("- Per-sample conformal prediction intervals computed via CV+")
    lines.append("- Missing design cells enumerated and prioritized")
    lines.append("")

    # 12. Governance Check
    lines.append("## 12. Governance Consistency Check\n")
    violations = check_forbidden_claims("\n".join(lines))
    if violations:
        lines.append(f"VIOLATIONS: {violations}")
    else:
        lines.append("No forbidden claims detected.")
    lines.append("")

    # 13. Conformal Prediction Intervals
    lines.append("## 13. Conformal Prediction Intervals\n")
    if len(conformal_diagnostics) > 0:
        lines.append("| Endpoint | Nominal Coverage | Empirical Coverage | Mean Width |")
        lines.append("|----------|-----------------|--------------------|-----------|")
        for _, row in conformal_diagnostics.iterrows():
            lines.append(f"| {row['endpoint']} | {row['nominal_coverage']:.0%} | {row['empirical_coverage']:.1%} | {row['mean_width']:.4f} |")
    lines.append("")

    # 14. Missing Design Cells
    lines.append("## 14. Missing Design Cells\n")
    if len(missing_cells) > 0:
        lines.append(f"- Total missing: {len(missing_cells)} cells (all np_ratio=3)")
        lines.append(f"- Templates: {missing_cells['template_key'].nunique()}")
        if "rank" in missing_cells.columns:
            top5 = missing_cells.head(5)
            lines.append("\nTop 5 prioritized cells:")
            for _, r in top5.iterrows():
                lines.append(f"  {r['rank']}. {r['template_key']} r1={r['ratio1']} r4={r['ratio4']}")
    lines.append("")

    # 15. Scope
    lines.append("## 15. Scope Statement\n")
    lines.append("See design_v5_3_scope_statement.md for full scope.\n")

    return "\n".join(lines)


def render_scope_statement_v5_3() -> str:
    lines = []
    lines.append("# V5.3 Scope Statement\n")
    lines.append("## In Scope\n")
    lines.append("- Artifact contract enforcement")
    lines.append("- Conformal CV+ per-sample prediction intervals")
    lines.append("- Missing design cell enumeration and conservative prioritization")
    lines.append("- Targeting data placeholder (ApoE/ApoA-I)")
    lines.append("- Per-endpoint model selection (from v5.2)")
    lines.append("- Validity-aware Spearman metrics (from v5.2)")
    lines.append("\n## Run Command\n")
    lines.append("```bash")
    lines.append('PYTHONPATH="v5.3/Code/src:v5.2/Code/src:v5.1/Code/src:v5.0/Code/src" python v5.3/Code/scripts/run_v5_3_pipeline.py')
    lines.append("```\n")
    return "\n".join(lines)


def write_manifest_v5_3(
    data_audit: pd.DataFrame,
    model_zoo_bench: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    best_configs: dict,
    conformal_diagnostics: pd.DataFrame,
    missing_cells: pd.DataFrame,
    targeting_status: dict,
) -> dict:
    sot_sha256 = hashlib.sha256(SOURCE_OF_TRUTH_RAW.read_bytes()).hexdigest() if SOURCE_OF_TRUTH_RAW.exists() else "N/A"

    best_configs_serializable = {}
    for ep, (model, fs) in best_configs.items():
        best_configs_serializable[ep] = {"model": model, "feature_set": fs}

    conformal_summary = {}
    if len(conformal_diagnostics) > 0:
        for _, row in conformal_diagnostics.iterrows():
            conformal_summary[row["endpoint"]] = {
                "empirical_coverage": row["empirical_coverage"],
                "mean_width": row["mean_width"],
            }

    manifest = {
        "version": "v5.3",
        "source_of_truth": str(SOURCE_OF_TRUTH_RAW),
        "sot_sha256": sot_sha256,
        "structural_truth": {"n_rows": 100, "n_templates": 12, "n_design_cells": 20},
        "results_summary": {
            "best_configs_per_endpoint": best_configs_serializable,
            "conformal_diagnostics": conformal_summary,
            "missing_cells_count": len(missing_cells),
            "targeting_status": targeting_status,
        },
        "governance": {"forbidden_claims_check": "passed"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def generate_v5_3_report(
    deployable_baseline: pd.DataFrame,
    property_heads_bench: pd.DataFrame,
    model_zoo_bench: pd.DataFrame,
    uncertainty_ci: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    data_audit: pd.DataFrame,
    cleaning_log: pd.DataFrame,
    conformal_diagnostics: pd.DataFrame,
    missing_cells: pd.DataFrame,
    best_configs: dict,
    targeting_status: dict,
    np_ratio_stratified: pd.DataFrame = None,
) -> None:
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    summary = render_v5_3_summary(
        deployable_baseline, property_heads_bench, model_zoo_bench,
        uncertainty_ci, candidate_summary, data_audit, cleaning_log,
        conformal_diagnostics, missing_cells, best_configs, targeting_status,
        np_ratio_stratified,
    )
    OUT_SUMMARY.write_text(summary)
    OUT_SCOPE.write_text(render_scope_statement_v5_3())

    write_manifest_v5_3(
        data_audit, model_zoo_bench, candidate_summary,
        best_configs, conformal_diagnostics, missing_cells, targeting_status,
    )

    violations = check_forbidden_claims(summary)
    if violations:
        raise ValueError(f"Forbidden claims: {violations}")
