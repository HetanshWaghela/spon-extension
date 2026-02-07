#!/usr/bin/env python3
"""
Generate data-driven SPON allocation configurations.

This script:
1. Loads a model and tokenizer.
2. Builds a small calibration dataloader.
3. Runs layer-importance analysis (gradient / ILA).
4. Uses importance scores to select layers for:
   - CRITICAL-ONLY   (above-median sensitivity)
   - SHAPLEY-GUIDED  (top-k by importance)
   - CORNERSTONE     (layers above a degradation threshold)
5. Writes a YAML file with concrete layer indices that can be plugged into
   the existing allocation builder.

Typical usage:

python -m experiments.exp1_allocation.generate_data_driven_configs \\
    --model meta-llama/Llama-3.2-1B \\
    --output configs/data_driven_allocation_overrides.yaml \\
    --method gradient
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp1_allocation.layer_importance import analyze_layer_importance
from src.allocation import create_layer_mask_from_importance
from src.model_provider import (
    add_model_provider_args,
    load_causal_lm,
    load_tokenizer,
    log_provider_resolution,
    resolve_model_provider,
)
from src.spon_trainer import create_calibration_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data-driven SPON allocation configs.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID to analyze (e.g., meta-llama/Llama-3.2-1B).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/data_driven_allocation_overrides.yaml",
        help="Path to YAML file where data-driven configs will be written.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gradient",
        choices=["gradient", "ila", "shapley"],
        help="Layer-importance method to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cuda/mps/cpu).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Token block size for calibration data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for calibration data.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Approximate number of calibration chunks.",
    )
    add_model_provider_args(parser)
    return parser.parse_args()


def build_data_driven_entries(importance_scores) -> Dict[str, Any]:
    """
    Build concrete layer lists for CRITICAL-ONLY, SHAPLEY-GUIDED, CORNERSTONE
    using a single importance vector.
    """
    entries: Dict[str, Any] = {}

    # CRITICAL-ONLY: above-median sensitivity
    critical_layers = create_layer_mask_from_importance(
        importance_scores,
        fraction=0.5,
        method="threshold",
        threshold=float(torch.median(torch.tensor(importance_scores)).item()),
    )

    # SHAPLEY-GUIDED: top 50% by importance (even if method != shapley, same idea)
    shapley_layers = create_layer_mask_from_importance(
        importance_scores,
        fraction=0.5,
        method="top_k",
    )

    # CORNERSTONE: layers above a degradation threshold; here interpreted
    # as importance > 0.5 * max_score
    max_score = float(max(importance_scores)) if len(importance_scores) > 0 else 0.0
    cornerstone_threshold = 0.5 * max_score if max_score > 0 else 0.0
    cornerstone_layers = create_layer_mask_from_importance(
        importance_scores,
        fraction=0.5,
        method="cornerstone",
        threshold=cornerstone_threshold,
    )

    entries["CRITICAL-ONLY"] = {
        "layer_indices": critical_layers,
        "notes": "Above-median importance according to selected method.",
    }
    entries["SHAPLEY-GUIDED"] = {
        "layer_indices": shapley_layers,
        "notes": "Top-50% layers by importance (method-agnostic).",
    }
    entries["CORNERSTONE"] = {
        "layer_indices": cornerstone_layers,
        "threshold": cornerstone_threshold,
        "notes": "Layers above relative-importance threshold, proxy for cornerstone.",
    }

    return entries


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        from src import get_device
        args.device = str(get_device())

    device = torch.device(args.device)
    provider_resolution = resolve_model_provider(
        model_name=args.model,
        provider=args.model_provider,
        ollama_model=args.ollama_model,
        pull_ollama=args.pull_ollama,
        require_transformers_model=True,
    )
    log_provider_resolution(provider_resolution)

    tokenizer = load_tokenizer(provider_resolution.hf_model_name)
    model = load_causal_lm(provider_resolution.hf_model_name, device)

    dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        dataset_name="wikitext",
        dataset_subset="wikitext-2-raw-v1",
    )

    importance_dict = analyze_layer_importance(
        model=model,
        dataloader=dataloader,
        device=device,
        methods=[args.method],
    )

    importance_scores = importance_dict[args.method]
    overrides = build_data_driven_entries(importance_scores)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(overrides, f, sort_keys=False)

    print(f"Wrote data-driven allocation overrides to {output_path}")


if __name__ == "__main__":
    main()
