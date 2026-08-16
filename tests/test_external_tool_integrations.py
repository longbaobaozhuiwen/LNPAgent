from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _comet_root(tmp_path: Path) -> Path:
    root = tmp_path / "COMET"
    (root / "unimol").mkdir(parents=True)
    (root / "experiments" / "task_schemas").mkdir(parents=True)
    (root / "unimol" / "infer_np.py").write_text("# external runner\n", encoding="utf-8")
    (root / "experiments" / "dict.txt").write_text("[PAD]\n", encoding="utf-8")
    for schema in (
        "in_house_lnp_master_schema_NPratio_AOvolratio.json",
        "in_house_lnp_master_schema_NPratio_AOvolratio_PBAE.json",
        "in_house_lnp_master_schema_NPratio_AOvolratio_PBAE_SSLNP.json",
    ):
        (root / "experiments" / "task_schemas" / schema).write_text("{}\n", encoding="utf-8")
    return root


def _lamgen_root(tmp_path: Path) -> Path:
    root = tmp_path / "LaMGen"
    for relative in ("model/lamgen_model.py", "scripts/train_triple.py", "utils/bert_tokenizer.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# external runner\n", encoding="utf-8")
    (root / "Pretrained_model").mkdir()
    (root / "data").mkdir()
    (root / "data" / "torsion_voc.csv").write_text("token\n", encoding="utf-8")
    return root


def test_external_tool_status_reports_configured_roots(tmp_path):
    from lnp_agent.external_tools import external_tools_status

    payload = external_tools_status(_comet_root(tmp_path), _lamgen_root(tmp_path))

    assert payload["artifact_schema"] == "lnp_agent.external_tools_status.v1"
    assert payload["bundled_external_assets"] is False
    assert [tool["runner_available"] for tool in payload["tools"]] == [True, True]


def test_comet_adapter_constructs_stability_command_and_manifest(tmp_path, monkeypatch):
    from lnp_agent import external_tools

    root = _comet_root(tmp_path)
    lmdb = tmp_path / "input.lmdb"
    checkpoint = tmp_path / "checkpoint.pt"
    lmdb.mkdir()
    checkpoint.write_text("weights stay local", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(command, *, cwd):
        seen["command"] = command
        seen["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(external_tools, "_run", fake_run)
    output = tmp_path / "result"
    payload = external_tools.run_comet_inference(
        input_lmdb=lmdb,
        checkpoint=checkpoint,
        output_dir=output,
        task="stability",
        comet_root=root,
    )

    assert payload["status"] == "completed"
    assert payload["private_data_included"] is False
    assert Path(payload["manifest_path"]).is_file()
    command = seen["command"]
    assert "in_house_lnp_master_schema_NPratio_AOvolratio_PBAE_SSLNP.json" in " ".join(command)
    assert command[command.index("--user-dir") + 2] == "."
    assert "--valid-subset" in command
    assert "test" in command


def test_lamgen_adapter_constructs_protein_conditioned_request(tmp_path, monkeypatch):
    from lnp_agent import external_tools

    root = _lamgen_root(tmp_path)
    checkpoint = tmp_path / "dual_target_ckpt"
    embedding_a = tmp_path / "target_a.npy"
    embedding_b = tmp_path / "target_b.npy"
    checkpoint.write_text("weights stay local", encoding="utf-8")
    embedding_a.write_text("embedding", encoding="utf-8")
    embedding_b.write_text("embedding", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(command, *, cwd):
        seen["command"] = command
        seen["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr(external_tools, "_run", fake_run)
    output = tmp_path / "molecules.csv"
    payload = external_tools.run_lamgen_generation(
        targets=["TARGET_A", "TARGET_B"],
        embeddings=[embedding_a, embedding_b],
        model_path=checkpoint,
        output_path=output,
        mode="dual",
        lamgen_root=root,
        samples=4,
    )

    assert payload["status"] == "completed"
    assert payload["targets"] == ["TARGET_A", "TARGET_B"]
    assert Path(payload["manifest_path"]).is_file()
    command = seen["command"]
    assert command[1].endswith("lamgen_runner.py")
    assert command.count("--target") == 2
    assert command.count("--embedding") == 2


def test_external_tools_are_registered():
    from lnp_agent.sandbox import Sandbox
    from lnp_agent.tools import create_all_tools

    tools = create_all_tools(Sandbox(Path.cwd()))

    assert "run_comet_inference" in tools
    assert "generate_lamgen_molecules" in tools


def test_external_tools_status_cli_is_dependency_light(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lnp_agent",
            "--external-tools-status",
            "--comet-root",
            str(tmp_path / "missing-comet"),
            "--lamgen-root",
            str(tmp_path / "missing-lamgen"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"COMET"' in result.stdout
    assert '"LaMGen"' in result.stdout
