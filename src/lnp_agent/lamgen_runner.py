"""Execution module used inside an operator's LaMGen environment."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNPAgent adapter for local LaMGen generation.")
    parser.add_argument("--lamgen-root", required=True)
    parser.add_argument("--mode", choices=("dual", "triple"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pretrained-model-dir", required=True)
    parser.add_argument("--vocab-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--embedding", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    return parser.parse_args()


def _decode(tokens: list[str]) -> str:
    return " ".join(token for token in tokens if token != "<|endofmask|>")


def main() -> int:
    args = _parse_args()
    expected = 2 if args.mode == "dual" else 3
    if len(args.target) != expected or len(args.embedding) != expected:
        raise ValueError(f"{args.mode} generation needs exactly {expected} targets and embeddings.")
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive.")

    root = Path(args.lamgen_root).expanduser().resolve()
    sys.path[:0] = [str(root), str(root / "scripts")]
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as functional
    from model.lamgen_model import LaMGen_dual, LaMGen_triple
    from train_triple import Ada_config
    from utils.bert_tokenizer import ExpressionBertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ExpressionBertTokenizer.from_pretrained(args.vocab_path)
    model_class = LaMGen_dual if args.mode == "dual" else LaMGen_triple
    model = model_class(pretrain_path=args.pretrained_model_dir, config=Ada_config)
    checkpoint = torch.load(args.model_path, map_location=device)
    state = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.load_state_dict(state, strict=args.mode == "triple")
    model.to(device).eval()

    proteins = [np.load(path, allow_pickle=True) for path in args.embedding]
    max_length = max(protein.shape[0] for protein in proteins)
    proteins = [np.pad(protein, ((0, max_length - protein.shape[0]), (0, 0))) for protein in proteins]
    proteins = [torch.tensor(protein, dtype=torch.float32, device=device).unsqueeze(0) for protein in proteins]
    prompt = tokenizer.encode("<|beginoftext|> <|mask:0|> <|mask:0|>", add_special_tokens=False)
    outputs: list[dict[str, str]] = []

    for sample_index in range(args.samples):
        random.seed(sample_index)
        np.random.seed(sample_index)
        torch.manual_seed(sample_index)
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        finished = False
        generated: list[str] = []
        with torch.no_grad():
            for _ in range(195):
                repeated = [protein.repeat(1, 1, 1) for protein in proteins]
                logits = model(input_ids, *repeated).logits[:, -1, :]
                token_id = torch.multinomial(functional.softmax(logits, dim=-1), 1)
                token = tokenizer.convert_ids_to_tokens(token_id)[0][0]
                if token == "<|endofmask|>":
                    finished = True
                    break
                generated.append(token)
                input_ids = torch.cat((input_ids, token_id), dim=1)
        outputs.append(
            {
                "molecule_tokens": _decode(generated),
                "terminated": str(finished).lower(),
                **{f"target_{index + 1}": target for index, target in enumerate(args.target)},
            }
        )

    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(output, index=False)
    print(f"Wrote {len(outputs)} LaMGen samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
