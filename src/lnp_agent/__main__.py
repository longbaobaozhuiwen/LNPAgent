"""Small, dependency-light command line entry point."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Utilities for the LNPAgent research package.")
    parser.add_argument("--check-data", action="store_true", help="Validate the configured input CSV.")
    parser.add_argument(
        "--public-summary", action="store_true",
        help="Summarize a public LNPDB CSV without mapping assays to native endpoints.",
    )
    args = parser.parse_args()

    if args.check_data or args.public_summary:
        from lnp_agent.data_validation import summarize_public_lnpdb, validate_csv
        from lnp_agent.paths import SOURCE_OF_TRUTH

        if args.public_summary:
            import json
            print(json.dumps(summarize_public_lnpdb(SOURCE_OF_TRUTH), indent=2))
        else:
            summary = validate_csv(SOURCE_OF_TRUTH)
            print(f"Validated {summary.rows} rows and {summary.columns} columns as {summary.schema} schema.")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
