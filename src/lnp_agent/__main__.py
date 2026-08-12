"""Small, dependency-light command line entry point."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Utilities for the LNPAgent research package.")
    parser.add_argument("--check-data", action="store_true", help="Validate the configured input CSV.")
    args = parser.parse_args()

    if args.check_data:
        from lnp_agent.data_validation import validate_csv
        from lnp_agent.paths import SOURCE_OF_TRUTH

        summary = validate_csv(SOURCE_OF_TRUTH)
        print(
            f"Validated {summary.rows} rows and {summary.columns} columns "
            f"as {summary.schema} schema."
        )
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
