"""Fail when Git-tracked paths violate LNPAgent's public-release policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath


RESTRICTED_PREFIXES = (
    "Data/",
    "artifacts/",
    "agent_output/",
    "working_data/",
    "v1/",
    "v1.",
    "v2.",
)
RESTRICTED_FILENAMES = {
    "example_training_data.csv",
    "generate_example_data.py",
}
RESTRICTED_TOKENS = ("jin" + "hao", "il" + "1b", "ccl" + "2")
RESTRICTED_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pt",
    ".pth",
    ".ckpt",
}
ALLOWED_DATA_FILES = {
    "data/README.md",
    "data/LNPDB_NOTICE.md",
    "data/lnpdb_public_example.csv",
}


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=False
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def violations(paths: list[str]) -> list[str]:
    invalid = []
    for path in paths:
        name = PurePosixPath(path).name
        if path.startswith("data/") and path not in ALLOWED_DATA_FILES:
            invalid.append(f"unapproved public data path: {path}")
        elif path.startswith(RESTRICTED_PREFIXES):
            invalid.append(f"restricted path: {path}")
        elif name in RESTRICTED_FILENAMES:
            invalid.append(f"removed private-example artifact: {path}")
        elif PurePosixPath(path).suffix.lower() in RESTRICTED_SUFFIXES:
            invalid.append(f"model artifact: {path}")
        else:
            try:
                content = Path(path).read_text(encoding="utf-8").lower()
            except (OSError, UnicodeDecodeError):
                continue
            for token in RESTRICTED_TOKENS:
                if token in content:
                    invalid.append(f"restricted token '{token}' in: {path}")
    return invalid


def main() -> int:
    invalid = violations(tracked_paths())
    if invalid:
        print("Public repository audit failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in invalid), file=sys.stderr)
        return 1
    print("Public repository audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
