"""Agent-callable adapters for optional COMET and LaMGen runtimes."""

from __future__ import annotations

import json

from lnp_agent.external_tools import ExternalToolError, run_comet_inference, run_lamgen_generation
from lnp_agent.tools.base import BaseTool, ToolDefinition, ToolResult


class RunCOMETInference(BaseTool):
    def __init__(self) -> None:
        self.definition = ToolDefinition(
            name="run_comet_inference",
            description="Run a local COMET checkpoint on a COMET-preprocessed LNP LMDB dataset.",
            parameters={
                "input_lmdb": {"type": "string", "description": "COMET-preprocessed LMDB input path."},
                "checkpoint": {"type": "string", "description": "Compatible local COMET checkpoint path."},
                "output_dir": {"type": "string", "description": "Directory for local COMET results and manifest."},
                "task": {"type": "string", "description": "lipid, pbae, or stability."},
                "comet_root": {"type": "string", "description": "Optional local COMET checkout."},
            },
            required=["input_lmdb", "checkpoint", "output_dir"],
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            payload = run_comet_inference(**kwargs)
        except (ExternalToolError, TypeError, ValueError) as error:
            return ToolResult(success=False, output="", error=str(error))
        return ToolResult(success=True, output=json.dumps(payload, indent=2))


class GenerateLaMGenMolecules(BaseTool):
    def __init__(self) -> None:
        self.definition = ToolDefinition(
            name="generate_lamgen_molecules",
            description=(
                "Generate dual- or triple-target molecular token sequences with a local LaMGen "
                "runtime using one ESM-C embedding array per target."
            ),
            parameters={
                "targets": {"type": "array", "description": "Two or three target identifiers."},
                "embeddings": {"type": "array", "description": "Matching ESM-C .npy embedding paths."},
                "model_path": {"type": "string", "description": "Local LaMGen checkpoint path."},
                "output_path": {"type": "string", "description": "CSV path for generated molecular tokens."},
                "mode": {"type": "string", "description": "dual or triple."},
                "lamgen_root": {"type": "string", "description": "Optional local LaMGen checkout."},
            },
            required=["targets", "embeddings", "model_path", "output_path"],
        )

    def execute(self, **kwargs) -> ToolResult:
        try:
            payload = run_lamgen_generation(**kwargs)
        except (ExternalToolError, TypeError, ValueError) as error:
            return ToolResult(success=False, output="", error=str(error))
        return ToolResult(success=True, output=json.dumps(payload, indent=2))
