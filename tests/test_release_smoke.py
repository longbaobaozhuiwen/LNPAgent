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


def test_unknown_schema_has_a_clear_error(tmp_path):
    from lnp_agent.data_validation import validate_csv

    path = tmp_path / "unknown.csv"
    path.write_text("unrelated\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported CSV schema"):
        validate_csv(path)


def test_release_assets_are_present():
    root = Path(__file__).resolve().parents[1]
    for name in ("agent-workflow.svg",):
        assert (root / "assets" / name).is_file()


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
