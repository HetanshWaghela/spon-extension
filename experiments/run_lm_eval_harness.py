#!/usr/bin/env python3
"""
CLI wrapper for `run_lm_eval_harness` in `src.evaluation`.

Usage example:

python -m experiments.run_lm_eval_harness \\
    --model meta-llama/Llama-3.2-1B \\
    --output_dir results/lm_eval/llama-3.2-1b \\
    --tasks mmlu,arc_easy,hellaswag \\
    --batch_size auto \\
    --device cuda:0
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import run_lm_eval_harness
from src.model_provider import (
    add_model_provider_args,
    log_provider_resolution,
    resolve_model_provider,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval-harness via Python wrapper.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path (passed to lm-eval as `pretrained=`).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store lm-eval results.json and logs.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu,arc_easy,hellaswag",
        help="Comma-separated list of lm-eval tasks.",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help='Batch size for lm-eval (e.g. "auto", "16").',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cuda/mps/cpu).",
    )
    add_model_provider_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        from src import get_device
        args.device = str(get_device())

    resolution = resolve_model_provider(
        model_name=args.model,
        provider=args.model_provider,
        ollama_model=args.ollama_model,
        pull_ollama=args.pull_ollama,
        require_transformers_model=True,
    )
    log_provider_resolution(resolution)
    if resolution.used_fallback and resolution.reason:
        logger.warning("lm-eval using Hugging Face runtime: %s", resolution.reason)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    metrics = run_lm_eval_harness(
        model_path=resolution.hf_model_name,
        output_dir=args.output_dir,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
    )

    if not metrics:
        print("lm-eval finished, but no metrics were parsed (check logs).")
    else:
        print("lm-eval metrics:")
        for task, acc in metrics.items():
            print(f"{task}: {acc:.4f}")


if __name__ == "__main__":
    main()
