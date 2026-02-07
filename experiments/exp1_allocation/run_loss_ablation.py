#!/usr/bin/env python3
"""
Run a small loss-ablation experiment for SPON training.

This script compares different distillation losses:
  - KL (paper default)
  - CE (soft cross-entropy to dense teacher)
  - KL+logit_l2 (KL + logit-level MSE penalty)

Usage:
    python -m experiments.exp1_allocation.run_loss_ablation --model meta-llama/Llama-3.2-1B
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.allocation import AllocationBuilder
from src.model_provider import (
    add_model_provider_args,
    load_causal_lm,
    load_tokenizer,
    log_provider_resolution,
    resolve_model_provider,
)
from src.spon_trainer import SPONTrainer, TrainingArgs, create_calibration_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPON loss ablation experiment.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="TOP-50",
        help="Allocation config name (e.g., TOP-50, UNIF-ALL).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Activation sparsity level.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs per loss variant (keep small for ablation).",
    )
    parser.add_argument(
        "--losses",
        type=str,
        nargs="+",
        default=["kl", "ce", "kl+logit_l2"],
        choices=["kl", "ce", "kl+logit_l2"],
        help="Distillation losses to compare.",
    )
    parser.add_argument(
        "--logit_l2_weight",
        type=float,
        default=0.1,
        help="L2 weight used for the kl+logit_l2 variant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/loss_ablation",
        help="Directory to store metrics.",
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

    provider_resolution = resolve_model_provider(
        model_name=args.model,
        provider=args.model_provider,
        ollama_model=args.ollama_model,
        pull_ollama=args.pull_ollama,
        require_transformers_model=True,
    )
    log_provider_resolution(provider_resolution)

    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    tokenizer = load_tokenizer(provider_resolution.hf_model_name)

    dataloader = create_calibration_dataloader(
        tokenizer=tokenizer,
        block_size=128,
        batch_size=8,
        num_samples=512,
        dataset_name="wikitext",
        dataset_subset="wikitext-2-raw-v1",
    )

    config_path = Path(__file__).resolve().parents[2] / "configs" / "allocation_configs.yaml"
    builder = AllocationBuilder(str(config_path))
    allocation_cfg = builder.build_config(args.config, args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    from src import clear_memory

    for loss_name in args.losses:
        print(f"=== Training with loss={loss_name} ===")
        model = load_causal_lm(provider_resolution.hf_model_name, device)

        train_args = TrainingArgs(
            epochs=args.epochs,
            device=args.device,
            distillation_loss=loss_name,
            logit_l2_weight=args.logit_l2_weight if loss_name == "kl+logit_l2" else 0.0,
        )
        trainer = SPONTrainer(
            model=model,
            config=allocation_cfg,
            sparsity=args.sparsity,
            args=train_args,
            tokenizer=tokenizer,
        )
        metrics = trainer.train(dataloader)
        final_loss = float(metrics["final_loss"])
        print(f"[{loss_name}] final_loss = {final_loss:.6f}")
        results[loss_name] = {
            "final_loss": final_loss,
            "logit_l2_weight": train_args.logit_l2_weight,
        }

        del trainer
        del model
        clear_memory(device)

    summary_path = output_dir / "loss_ablation_summary.txt"
    with open(summary_path, "w") as f:
        for loss_name in args.losses:
            f.write(f"{loss_name}: {results[loss_name]['final_loss']:.6f}\n")

    json_path = output_dir / "loss_ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "config": args.config,
                "sparsity": args.sparsity,
                "epochs": args.epochs,
                "losses": results,
            },
            f,
            indent=2,
        )

    print(f"Wrote loss ablation summary to {summary_path}")
    print(f"Wrote loss ablation JSON to {json_path}")


if __name__ == "__main__":
    main()
