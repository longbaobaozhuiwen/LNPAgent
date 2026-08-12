"""Small, dependency-light command line entry point."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Utilities for the LNPAgent research package.")
    parser.add_argument("--check-data", action="store_true", help="Validate the configured input CSV.")
    parser.add_argument(
        "--public-summary", action="store_true",
        help="Summarize a public LNPDB CSV without mapping assays to native endpoints.",
    )
    parser.add_argument(
        "--benchmark-public", action="store_true",
        help="Write a reproducible public LNPDB schema/coverage benchmark JSON.",
    )
    parser.add_argument(
        "--demo-public", action="store_true",
        help="Write a public-safe one-round acquisition demo JSON.",
    )
    parser.add_argument(
        "--benchmark-output",
        help="Output JSON path for --benchmark-public; defaults under LNP_AGENT_ARTIFACTS.",
    )
    parser.add_argument(
        "--demo-output",
        help="Output JSON path for --demo-public; defaults under LNP_AGENT_ARTIFACTS.",
    )
    parser.add_argument("--demo-library-size", type=int, default=24)
    parser.add_argument("--demo-batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed metadata.")
    args = parser.parse_args()

    if args.check_data or args.public_summary or args.benchmark_public or args.demo_public:
        from lnp_agent.data_validation import (
            benchmark_public_lnpdb,
            build_public_demo_round,
            summarize_public_lnpdb,
            validate_csv,
        )
        from lnp_agent.paths import RESULTS_DIR, SOURCE_OF_TRUTH

        if args.public_summary:
            print(json.dumps(summarize_public_lnpdb(SOURCE_OF_TRUTH), indent=2))
        elif args.benchmark_public:
            output = args.benchmark_output
            if output is None:
                output = RESULTS_DIR / "benchmark_public_lnpdb.json"
            else:
                from pathlib import Path
                output = Path(output)
            payload = benchmark_public_lnpdb(SOURCE_OF_TRUTH, seed=args.seed)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote public benchmark: {output}")
        elif args.demo_public:
            output = args.demo_output
            if output is None:
                output = RESULTS_DIR / "public_demo_round.json"
            else:
                from pathlib import Path
                output = Path(output)
            payload = build_public_demo_round(
                SOURCE_OF_TRUTH,
                seed=args.seed,
                library_size=args.demo_library_size,
                batch_size=args.demo_batch_size,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Wrote public demo: {output}")
        else:
            summary = validate_csv(SOURCE_OF_TRUTH)
            print(f"Validated {summary.rows} rows and {summary.columns} columns as {summary.schema} schema.")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
