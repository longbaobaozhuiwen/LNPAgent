"""Portable paths used by LNPAgent.

Set ``LNP_AGENT_HOME`` to point to a checkout and ``LNP_AGENT_ARTIFACTS`` to
store generated files elsewhere.  No source module assumes a developer-local
absolute path.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("LNP_AGENT_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_OF_TRUTH = Path(
    os.environ.get("LNP_AGENT_DATA", DATA_DIR / "lnpdb_public_example.csv")
)
RESULTS_DIR = Path(os.environ.get("LNP_AGENT_ARTIFACTS", PROJECT_ROOT / "artifacts"))
WORKING_DATA_DIR = RESULTS_DIR / "working_data"
FIGURES_DIR = RESULTS_DIR / "figures"
WET_LAB_DIR = RESULTS_DIR / "wet_lab"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Optional assets, deliberately not bundled in the repository.
AGILE_CHECKPOINT = Path(os.environ.get("LNP_AGENT_AGILE_CHECKPOINT", "agile_model.pth"))
GEMMA_MODEL_DIR = os.environ.get("LNP_AGENT_GEMMA_MODEL")
