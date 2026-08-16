"""Small, dependency-light command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser.add_argument(
        "--external-tools-status",
        action="store_true",
        help="Report whether locally configured COMET and LaMGen runtimes are callable.",
    )
    parser.add_argument("--comet-predict", action="store_true", help="Run local COMET inference.")
    parser.add_argument("--comet-root", help="Path to an installed COMET checkout.")
    parser.add_argument("--comet-python", help="Python executable from the COMET environment.")
    parser.add_argument("--comet-input-lmdb", help="COMET-preprocessed input LMDB directory.")
    parser.add_argument("--comet-checkpoint", help="Compatible local COMET checkpoint.")
    parser.add_argument("--comet-output", help="Output directory for COMET results and manifest.")
    parser.add_argument("--comet-task", choices=("lipid", "pbae", "stability"), default="lipid")
    parser.add_argument("--comet-schema", help="Optional COMET task-schema JSON path.")
    parser.add_argument("--comet-batch-size", type=int, default=256)
    parser.add_argument("--lamgen-generate", action="store_true", help="Run local LaMGen generation.")
    parser.add_argument("--lamgen-root", help="Path to an installed LaMGen checkout.")
    parser.add_argument("--lamgen-python", help="Python executable from the LaMGen environment.")
    parser.add_argument("--lamgen-mode", choices=("dual", "triple"), default="dual")
    parser.add_argument(
        "--lamgen-target",
        action="append",
        default=[],
        metavar="TARGET=EMBEDDING.npy",
        help="Repeat once per protein target; values are target identifier and ESM-C embedding path.",
    )
    parser.add_argument("--lamgen-model", help="Local LaMGen checkpoint.")
    parser.add_argument("--lamgen-output", help="CSV path for generated molecular tokens.")
    parser.add_argument("--lamgen-pretrained-model-dir", help="Optional LaMGen pretrained-model directory.")
    parser.add_argument("--lamgen-vocab", help="Optional LaMGen vocabulary CSV path.")
    parser.add_argument("--lamgen-batch-size", type=int, default=50)
    parser.add_argument("--lamgen-samples", type=int, default=100)
    args = parser.parse_args()

    if args.external_tools_status or args.comet_predict or args.lamgen_generate:
        from lnp_agent.external_tools import (
            ExternalToolError,
            external_tools_status,
            run_comet_inference,
            run_lamgen_generation,
        )
        from lnp_agent.paths import RESULTS_DIR

        try:
            if args.external_tools_status:
                print(json.dumps(external_tools_status(args.comet_root, args.lamgen_root), indent=2))
            elif args.comet_predict:
                if not args.comet_input_lmdb or not args.comet_checkpoint:
                    parser.error("--comet-predict requires --comet-input-lmdb and --comet-checkpoint")
                output = Path(args.comet_output) if args.comet_output else RESULTS_DIR / "comet"
                payload = run_comet_inference(
                    input_lmdb=args.comet_input_lmdb,
                    checkpoint=args.comet_checkpoint,
                    output_dir=output,
                    task=args.comet_task,
                    comet_root=args.comet_root,
                    python_executable=args.comet_python,
                    schema_path=args.comet_schema,
                    batch_size=args.comet_batch_size,
                )
                print(json.dumps(payload, indent=2))
            else:
                if not args.lamgen_model:
                    parser.error("--lamgen-generate requires --lamgen-model")
                parsed_targets: list[str] = []
                embeddings: list[str] = []
                for item in args.lamgen_target:
                    if "=" not in item:
                        parser.error("--lamgen-target must use TARGET=EMBEDDING.npy")
                    target, embedding = item.split("=", 1)
                    if not target or not embedding:
                        parser.error("--lamgen-target must use TARGET=EMBEDDING.npy")
                    parsed_targets.append(target)
                    embeddings.append(embedding)
                output = Path(args.lamgen_output) if args.lamgen_output else RESULTS_DIR / "lamgen_generated.csv"
                payload = run_lamgen_generation(
                    targets=parsed_targets,
                    embeddings=embeddings,
                    model_path=args.lamgen_model,
                    output_path=output,
                    mode=args.lamgen_mode,
                    lamgen_root=args.lamgen_root,
                    python_executable=args.lamgen_python,
                    pretrained_model_dir=args.lamgen_pretrained_model_dir,
                    vocab_path=args.lamgen_vocab,
                    batch_size=args.lamgen_batch_size,
                    samples=args.lamgen_samples,
                )
                print(json.dumps(payload, indent=2))
        except ExternalToolError as error:
            parser.error(str(error))
        return 0

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
