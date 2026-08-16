"""Adapters for optional, locally installed LNP research tools.

The package deliberately does not vendor COMET, LaMGen, their data, or model
weights.  These adapters provide a stable LNPAgent request/result boundary for
users who have installed the respective upstream projects locally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


COMET_SCHEMAS = {
    "lipid": "experiments/task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio.json",
    "pbae": "experiments/task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio_PBAE.json",
    "stability": "experiments/task_schemas/in_house_lnp_master_schema_NPratio_AOvolratio_PBAE_SSLNP.json",
}

COMET_SUBSETS = {"lipid": "infer", "pbae": "test", "stability": "test"}


class ExternalToolError(RuntimeError):
    """Raised when an optional upstream tool is unavailable or rejects a request."""


@dataclass(frozen=True)
class ToolStatus:
    name: str
    configured_root: str | None
    runner_available: bool
    required_paths: dict[str, bool]
    setup_hint: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "configured_root": self.configured_root,
            "runner_available": self.runner_available,
            "required_paths": self.required_paths,
            "setup_hint": self.setup_hint,
        }


def _optional_path(value: str | Path | None, env_name: str) -> Path | None:
    configured = value or os.environ.get(env_name)
    return Path(configured).expanduser().resolve() if configured else None


def comet_status(comet_root: str | Path | None = None) -> ToolStatus:
    root = _optional_path(comet_root, "LNP_AGENT_COMET_ROOT")
    required = {
        "unimol/infer_np.py": bool(root and (root / "unimol" / "infer_np.py").is_file()),
        "experiments/task_schemas": bool(root and (root / "experiments" / "task_schemas").is_dir()),
    }
    return ToolStatus(
        name="COMET",
        configured_root=str(root) if root else None,
        runner_available=all(required.values()),
        required_paths=required,
        setup_hint=(
            "Clone https://github.com/alvinchangw/COMET, install its documented "
            "CUDA/Uni-Core environment, then set LNP_AGENT_COMET_ROOT. "
            "Pass a COMET-preprocessed LMDB and a compatible checkpoint per run."
        ),
    )


def lamgen_status(lamgen_root: str | Path | None = None) -> ToolStatus:
    root = _optional_path(lamgen_root, "LNP_AGENT_LAMGEN_ROOT")
    required = {
        "model/lamgen_model.py": bool(root and (root / "model" / "lamgen_model.py").is_file()),
        "scripts/train_triple.py": bool(root and (root / "scripts" / "train_triple.py").is_file()),
        "utils/bert_tokenizer.py": bool(root and (root / "utils" / "bert_tokenizer.py").is_file()),
    }
    return ToolStatus(
        name="LaMGen",
        configured_root=str(root) if root else None,
        runner_available=all(required.values()),
        required_paths=required,
        setup_hint=(
            "Install a complete LaMGen checkout with its checkpoint and ESM-C "
            "embeddings, then set LNP_AGENT_LAMGEN_ROOT. Protein conditioning uses "
            "the upstream ESM-C embedding interface."
        ),
    )


def external_tools_status(
    comet_root: str | Path | None = None, lamgen_root: str | Path | None = None
) -> dict:
    """Return a serializable readiness report for both optional integrations."""
    return {
        "artifact_schema": "lnp_agent.external_tools_status.v1",
        "tools": [comet_status(comet_root).as_dict(), lamgen_status(lamgen_root).as_dict()],
        "bundled_external_assets": False,
    }


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ExternalToolError(f"Could not start external tool: {error}") from error


def run_comet_inference(
    *,
    input_lmdb: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    task: Literal["lipid", "pbae", "stability"] = "lipid",
    comet_root: str | Path | None = None,
    python_executable: str | None = None,
    schema_path: str | Path | None = None,
    batch_size: int = 256,
) -> dict:
    """Run the upstream COMET inference entry point and write a manifest.

    ``input_lmdb`` must already follow COMET's documented preprocessed LMDB
    schema.  The stability task uses COMET's lyophilized-LNP schema; prediction
    semantics remain defined by the checkpoint supplied by the operator.
    """
    if task not in COMET_SCHEMAS:
        raise ExternalToolError(f"Unsupported COMET task '{task}'. Choose from {sorted(COMET_SCHEMAS)}.")
    status = comet_status(comet_root)
    if not status.runner_available or status.configured_root is None:
        raise ExternalToolError(status.setup_hint)

    root = Path(status.configured_root)
    input_path = Path(input_lmdb).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    selected_schema = Path(schema_path).expanduser().resolve() if schema_path else root / COMET_SCHEMAS[task]
    for label, path in {"input_lmdb": input_path, "checkpoint": checkpoint_path, "schema": selected_schema}.items():
        if not path.exists():
            raise ExternalToolError(f"COMET {label} does not exist: {path}")
    if batch_size < 1:
        raise ExternalToolError("COMET batch_size must be positive.")

    output_path.mkdir(parents=True, exist_ok=True)
    executable = python_executable or os.environ.get("LNP_AGENT_COMET_PYTHON") or sys.executable
    subset = COMET_SUBSETS[task]
    command = [
        executable,
        str(root / "unimol" / "infer_np.py"),
        "--user-dir",
        str(root / "unimol"),
        ".",
        "--task-name",
        str(input_path),
        "--valid-subset",
        subset,
        "--num-workers",
        "8",
        "--ddp-backend",
        "c10d",
        "--batch-size",
        str(batch_size),
        "--task",
        "mol_np_finetune",
        "--loss",
        "np_finetune_contrastive",
        "--arch",
        "np_unimol",
        "--classification-head-name",
        str(input_path),
        "--num-classes",
        "1",
        "--dict-name",
        "dict.txt",
        "--conf-size",
        "11",
        "--only-polar",
        "0",
        "--path",
        str(checkpoint_path),
        "--fp16",
        "--fp16-init-scale",
        "4",
        "--fp16-scale-window",
        "256",
        "--log-interval",
        "50",
        "--log-format",
        "simple",
        "--results-path",
        str(output_path),
        "--lnp-encoder-layers",
        "8",
        "--lnp-encoder-embed-dim",
        "256",
        "--lnp-encoder-ffn-embed-dim",
        "256",
        "--lnp-encoder-attention-heads",
        "8",
        "--full-dataset-task-schema-path",
        str(selected_schema),
        "--load-full-np-model",
        "--concat-datasets",
        "--output-cls-rep",
    ]
    completed = _run(command, cwd=root / "experiments")
    result_files = sorted(str(path) for path in output_path.glob("*.out.pkl"))
    manifest = {
        "artifact_schema": "lnp_agent.comet_inference.v1",
        "tool": "COMET",
        "task": task,
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "input_lmdb": str(input_path),
        "checkpoint": str(checkpoint_path),
        "schema": str(selected_schema),
        "result_files": result_files,
        "private_data_included": False,
        "note": "Input, checkpoint, and result files remain local to the operator.",
    }
    manifest_path = output_path / "comet_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    if completed.returncode != 0:
        raise ExternalToolError(f"COMET failed; see {manifest_path}")
    return manifest


def run_lamgen_generation(
    *,
    targets: list[str],
    embeddings: list[str | Path],
    model_path: str | Path,
    output_path: str | Path,
    mode: Literal["dual", "triple"] = "dual",
    lamgen_root: str | Path | None = None,
    python_executable: str | None = None,
    pretrained_model_dir: str | Path | None = None,
    vocab_path: str | Path | None = None,
    batch_size: int = 50,
    samples: int = 100,
) -> dict:
    """Generate protein-conditioned molecules through a local LaMGen runtime.

    LaMGen's published runtime conditions on ESM-C embedding arrays, so callers
    provide one ``.npy`` embedding per target.  They may originate from amino
    acid sequences through the upstream ESM-C workflow.
    """
    expected_targets = 2 if mode == "dual" else 3
    if mode not in {"dual", "triple"}:
        raise ExternalToolError("LaMGen mode must be 'dual' or 'triple'.")
    if len(targets) != expected_targets or len(embeddings) != expected_targets:
        raise ExternalToolError(f"LaMGen {mode} generation requires exactly {expected_targets} targets and embeddings.")
    if batch_size < 1 or samples < 1:
        raise ExternalToolError("LaMGen batch_size and samples must be positive.")

    status = lamgen_status(lamgen_root)
    if not status.runner_available or status.configured_root is None:
        raise ExternalToolError(status.setup_hint)
    root = Path(status.configured_root)
    model = Path(model_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    embedding_paths = [Path(value).expanduser().resolve() for value in embeddings]
    if not model.is_file():
        raise ExternalToolError(f"LaMGen model checkpoint does not exist: {model}")
    missing_embeddings = [str(path) for path in embedding_paths if not path.is_file()]
    if missing_embeddings:
        raise ExternalToolError(f"LaMGen embedding files do not exist: {', '.join(missing_embeddings)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pretrained = Path(pretrained_model_dir).expanduser().resolve() if pretrained_model_dir else root / "Pretrained_model"
    vocab = Path(vocab_path).expanduser().resolve() if vocab_path else root / "data" / "torsion_voc.csv"
    if not pretrained.is_dir() or not vocab.is_file():
        raise ExternalToolError("LaMGen pretrained model directory or vocabulary is missing.")

    executable = python_executable or os.environ.get("LNP_AGENT_LAMGEN_PYTHON") or sys.executable
    runner = Path(__file__).with_name("lamgen_runner.py").resolve()
    command = [
        executable,
        str(runner),
        "--lamgen-root",
        str(root),
        "--mode",
        mode,
        "--model-path",
        str(model),
        "--pretrained-model-dir",
        str(pretrained),
        "--vocab-path",
        str(vocab),
        "--output-path",
        str(output),
        "--batch-size",
        str(batch_size),
        "--samples",
        str(samples),
    ]
    for target, embedding in zip(targets, embedding_paths, strict=True):
        command.extend(["--target", target, "--embedding", str(embedding)])
    completed = _run(command, cwd=root)
    manifest = {
        "artifact_schema": "lnp_agent.lamgen_generation.v1",
        "tool": "LaMGen",
        "mode": mode,
        "targets": targets,
        "embeddings": [str(path) for path in embedding_paths],
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_file": str(output),
        "private_data_included": False,
        "note": "Protein conditioning uses user-supplied ESM-C embeddings; generated molecules require downstream validation.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    if completed.returncode != 0:
        raise ExternalToolError(f"LaMGen failed; see {manifest_path}")
    return manifest
