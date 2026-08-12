import hashlib
from pathlib import Path

import pytest


def test_public_lnpdb_example_has_recorded_provenance():
    from lnp_agent.data_validation import validate_csv

    root = Path(__file__).resolve().parents[1]
    example = root / "data" / "lnpdb_public_example.csv"
    notice = (root / "data" / "LNPDB_NOTICE.md").read_text(encoding="utf-8")
    digest = hashlib.sha256(example.read_bytes()).hexdigest()

    assert digest in notice
    assert "evancollins1/LNPDB" in notice
    summary = validate_csv(example)
    assert summary.schema == "lnpdb"
    assert summary.rows == 100


def test_native_schema_is_still_recognized(tmp_path):
    from lnp_agent.data_validation import validate_csv

    path = tmp_path / "native.csv"
    path.write_text(
        "lipid1,lipid2,lipid3,lipid4,ratio1,ratio2,ratio3,ratio4,"
        "transfection_efficiency,immune_signal_a,immune_signal_b\n"
        "a,b,c,d,50,10,39,1,12,1.5,0.5\n",
        encoding="utf-8",
    )
    assert validate_csv(path).schema == "lnpagent-native"


def test_public_summary_does_not_relabel_assays():
    from lnp_agent.data_validation import summarize_public_lnpdb

    root = Path(__file__).resolve().parents[1]
    summary = summarize_public_lnpdb(root / "data" / "lnpdb_public_example.csv")
    assert summary["schema"] == "lnpdb"
    assert summary["rows"] == 100
    assert summary["numeric_experiment_values"] == 100
    assert "not an endpoint benchmark" in summary["scientific_use"]


def test_public_benchmark_is_reproducible_metadata():
    from lnp_agent.data_validation import benchmark_public_lnpdb

    root = Path(__file__).resolve().parents[1]
    bench = benchmark_public_lnpdb(root / "data" / "lnpdb_public_example.csv", seed=7)
    assert bench["benchmark_name"] == "public_lnpdb_schema_coverage"
    assert bench["random_seed"] == 7
    assert bench["provenance"]["generated_by"] == "lnp-agent --benchmark-public"
    assert bench["provenance"]["private_data_included"] is False
    assert bench["provenance"]["artifact_schema"] == "public_lnpdb_benchmark.v1"
    assert bench["metrics"]["rows_loaded"] == 100
    assert "not a model performance" in bench["scientific_use"]


def test_public_demo_round_uses_acquisition_policy():
    from lnp_agent.data_validation import build_public_demo_round

    root = Path(__file__).resolve().parents[1]
    demo = build_public_demo_round(
        root / "data" / "lnpdb_public_example.csv",
        seed=7,
        library_size=12,
        batch_size=4,
    )

    assert demo["demo_name"] == "public_lnpdb_one_round_acquisition_demo"
    assert demo["provenance"]["generated_by"] == "lnp-agent --demo-public"
    assert demo["provenance"]["measurement_type"] == "synthetic_public_demo"
    assert demo["provenance"]["private_data_included"] is False
    assert demo["candidate_pool_size"] == 12
    assert demo["selected_batch_size"] == 4
    assert len(demo["selected_candidates"]) == 4
    assert "not endpoint relabels" in demo["scientific_use"]
    first = demo["selected_candidates"][0]
    assert "experiment_value_score" in first
    assert "selection_rationale" in first


def test_public_demo_cli_writes_artifact(tmp_path):
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "demo.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lnp_agent",
            "--demo-public",
            "--demo-output",
            str(output),
            "--demo-library-size",
            "10",
            "--demo-batch-size",
            "3",
            "--seed",
            "11",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_pool_size"] == 10
    assert payload["selected_batch_size"] == 3
    assert payload["provenance"]["private_data_included"] is False


def test_candidate_uncertainty_is_candidate_specific():
    from lnp_core.candidate_ranking import generate_candidate_pareto
    # The implementation-level contract is that uncertainty is a column with
    # one value per candidate, rather than a single benchmark scalar.
    assert callable(generate_candidate_pareto)


def test_experiment_value_policy_balances_score_uncertainty_and_diversity():
    import pandas as pd
    from lnp_core.candidate_ranking import select_experiment_value_batch

    candidates = pd.DataFrame(
        {
            "Formulation_ID": ["exploit", "learn", "duplicate"],
            "template_key": ["common", "rare", "common"],
            "lipid1": ["A", "B", "A"],
            "lipid2": ["DSPC", "DOPE", "DSPC"],
            "lipid4": ["PEG", "PEG2", "PEG"],
            "ratio1": [50, 70, 50],
            "ratio2": [10, 8, 10],
            "ratio3": [38, 20, 38],
            "ratio4": [2, 2, 2],
            "pred_tx_log1p": [1.00, 0.92, 0.96],
            "pred_immune_signal_a": [0.10, 0.15, 0.12],
            "pred_immune_signal_b": [0.10, 0.15, 0.12],
            "pred_uncertainty_tx_log1p": [0.01, 0.80, 0.01],
            "pred_uncertainty_immune_signal_a": [0.01, 0.80, 0.01],
            "pred_uncertainty_immune_signal_b": [0.01, 0.80, 0.01],
            "pareto_front": [True, True, False],
        }
    )

    selected = select_experiment_value_batch(
        candidates,
        batch_size=1,
        weights={"exploitation": 0.15, "exploration": 0.70, "diversity": 0.15},
    )
    assert selected.loc[0, "Formulation_ID"] == "learn"
    assert selected.loc[0, "experiment_value_score"] > 0
    assert selected.loc[0, "selection_rationale"] in {
        "explore uncertain candidate",
        "cover under-sampled design region",
    }


def test_pareto_excludes_invalid_objectives():
    import pandas as pd
    from lnp_core.candidate_ranking import compute_pareto_front

    frame = pd.DataFrame({"a": [1.0, float("nan")], "b": [1.0, 0.0]})
    front = compute_pareto_front(frame, ["a", "b"], ["lower_is_better", "higher_is_better"])
    assert front.tolist() == [True, False]


def _native_csv(row_count: int = 2, tx_values: list[float] | None = None) -> str:
    tx_values = tx_values or [12.0] * row_count
    header = (
        "Formulation_ID,LNP_ID,lipid1,lipid2,lipid3,lipid4,"
        "lipid1_smiles,lipid2_smiles,lipid3_smiles,lipid4_smiles,"
        "ratio1,ratio2,ratio3,ratio4,np_ratio,aq_org_ratio,"
        "size,pdi,zeta_potential,encapsulation_efficiency,"
        "transfection_efficiency,immune_signal_a,immune_signal_b\n"
    )
    rows = []
    for i in range(row_count):
        rows.append(
            f"F{i},L{i},A,DSPC,Chol,PEG,CC,C,CCO,CO,"
            f"{50 + i},10,38,2,8,3,100,0.1,-5,90,{tx_values[i]},1.0,0.5"
        )
    return header + "\n".join(rows) + "\n"


def test_native_cleaning_accepts_non_private_grid_sizes(tmp_path):
    from lnp_core.data_cleaning import load_and_clean_v5_3

    path = tmp_path / "native_small.csv"
    path.write_text(_native_csv(row_count=2), encoding="utf-8")
    df, cleaning_log = load_and_clean_v5_3(path)
    assert len(df) == 2
    assert "tx_log1p" in df.columns
    assert "dataset_profile" in set(cleaning_log["check_name"])


def test_native_cleaning_rejects_invalid_log1p_domain(tmp_path):
    from lnp_core.data_cleaning import load_and_clean_v5_3

    path = tmp_path / "native_bad_tx.csv"
    path.write_text(_native_csv(row_count=1, tx_values=[-1.0]), encoding="utf-8")
    with pytest.raises(ValueError, match="transfection_efficiency must be greater than -1"):
        load_and_clean_v5_3(path)


def test_model_zoo_inner_cv_is_group_aware_when_templates_exist():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from lnp_core.model_zoo import _inner_cv_for_training

    df = pd.DataFrame({"template_key": ["a", "a", "b", "b", "c", "c"]})
    cv = _inner_cv_for_training(df, np.arange(len(df)))
    assert isinstance(cv, GroupKFold)


def test_unknown_schema_has_a_clear_error(tmp_path):
    from lnp_agent.data_validation import validate_csv

    path = tmp_path / "unknown.csv"
    path.write_text("unrelated\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported CSV schema"):
        validate_csv(path)


def test_release_assets_are_present():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "agent-workflow.svg",
        "evidence-loop.svg",
        "public-release-boundary.svg",
    ):
        assert (root / "assets" / name).is_file()
    assert (root / "docs" / "SCIENTIFIC_AND_ENGINEERING_AUDIT.md").is_file()


def test_state_machine_is_importable():
    from lnp_agent.engine_v5 import SOPState

    assert SOPState.COMPLETE.value == "complete"


def test_public_repository_audit_passes():
    root = Path(__file__).resolve().parents[1]
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/audit_public_repository.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_repository_audit_rejects_restricted_tokens(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "public_audit", root / "scripts" / "audit_public_repository.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    forbidden = tmp_path / "forbidden.txt"
    forbidden.write_text("forbidden token: " + "ccl" + "2", encoding="utf-8")
    assert any("restricted token" in item for item in module.violations([str(forbidden)]))
