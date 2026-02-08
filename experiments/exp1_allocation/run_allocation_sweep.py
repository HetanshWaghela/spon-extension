#!/usr/bin/env python3
"""
=============================================================================
Experiment 1: Layer-wise & Module-wise SPON Allocation Sweep
=============================================================================

This script systematically tests different SPON bias allocation strategies
to find the Pareto-optimal trade-off between parameter count and performance.

Research Questions:
1. Can we achieve equivalent PPL improvement with fewer SPON parameters?
2. Are early layers ("top") more important than later layers ("bottom")?
3. What is the minimum number of layers needed for effective SPON?

Usage:
    # Basic run with default settings
    python experiments/exp1_allocation/run_allocation_sweep.py
    
    # Custom configuration
    python experiments/exp1_allocation/run_allocation_sweep.py \
        --model meta-llama/Llama-3.2-1B \
        --configs BASELINE-TEAL UNIF-ALL TOP-25 TOP-50 \
        --sparsity 0.5 0.6 \
        --epochs 10 \
        --output_dir results

Output Structure:
    results/<experiment_name>/
    ├── runs/
    │   └── run_YYYYMMDD_HHMMSS/
    │       ├── summary.json        # Complete run summary
    │       ├── results.json        # Per-config results
    │       ├── metrics.csv         # Training metrics over time
    │       ├── config.json         # Run configuration
    │       ├── checkpoints/        # Saved SPON biases
    │       └── figures/            # Generated plots
    └── aggregated/
        ├── summary.csv             # All results in tabular form
        ├── pareto.json             # Pareto-optimal configurations
        └── latex_tables/           # Publication-ready tables

Authors: [Your Name]
Date: 2024
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.allocation import AllocationBuilder, SPONConfig
from src.model_provider import (
    add_model_provider_args,
    load_causal_lm,
    load_tokenizer,
    log_provider_resolution,
    resolve_model_provider,
)
from src.spon_trainer import SPONTrainer, TrainingArgs, create_calibration_dataloader
from src.evaluation import compute_perplexity
from src.result_manager import (
    ExperimentManager, 
    RunConfig, 
    ConfigResult
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SPON allocation sweep experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID or path"
    )
    
    # SPON configurations to test
    parser.add_argument(
        "--configs", 
        type=str, 
        nargs="+",
        default=["BASELINE-TEAL", "UNIF-ALL", "TOP-25", "TOP-50", "TOP-75", "BOTTOM-50"],
        help="Configuration names to test (from allocation_configs.yaml)"
    )
    
    # Sparsity levels
    parser.add_argument(
        "--sparsity", 
        type=float, 
        nargs="+",
        default=[0.5, 0.6],
        help="Sparsity levels to test (fraction of activations zeroed)"
    )
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per config")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length")
    parser.add_argument("--num_samples", type=int, default=2048, help="Calibration samples")
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=0,
        help="Evaluation samples (0 = use num_samples // 4, -1 = full test set)"
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset subset for calibration/eval (wikitext-103-raw-v1 matches paper)"
    )
    
    # Output settings
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Base results root (results are written under <output_dir>/<experiment_name>/...)"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="allocation_sweep",
        help="Experiment name for result organization"
    )
    
    # Execution settings
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip SPON training and load existing checkpoints for evaluation"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing SPON checkpoints when --skip_training is set "
             "(default: current run's checkpoints directory)"
    )
    parser.add_argument(
        "--default_sparse_modules",
        type=str,
        nargs="+",
        default=["down_proj"],
        help="Modules to sparsify when a config has no explicit modules "
             "(e.g. BASELINE-TEAL)"
    )
    parser.add_argument(
        "--data_driven_overrides",
        type=str,
        default=None,
        help="YAML file containing concrete layer indices for data-driven configs "
             "(e.g. CRITICAL-ONLY/SHAPLEY-GUIDED/CORNERSTONE)"
    )
    parser.add_argument(
        "--dense_baseline",
        action="store_true",
        help="Also compute dense (no sparsity) baseline"
    )
    add_model_provider_args(parser)
    
    return parser.parse_args()


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, device: str):
    """
    Load a fresh copy of the model.
    
    IMPORTANT: We reload for each config because training modifies model state.
    This ensures clean, reproducible results for each configuration.
    
    Args:
        model_name: HuggingFace model ID
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_name}")
    
    model = load_causal_lm(model_name, torch.device(device))
    
    # Log model info
    num_layers = len(model.model.layers)
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Model loaded: {num_layers} layers, {num_params:.2f}B parameters")
    
    return model


# =============================================================================
# Dense Baseline Computation
# =============================================================================

def compute_dense_baseline(
    model_name: str,
    tokenizer,
    eval_dataloader,
    device: str
) -> float:
    """
    Compute perplexity of dense (non-sparse) model as reference.
    
    This is the "gold standard" - the best possible performance.
    All sparse configurations should be compared against this.
    
    Args:
        model_name: Model to evaluate
        tokenizer: Tokenizer (unused but kept for consistency)
        eval_dataloader: Evaluation data
        device: Device to run on
    
    Returns:
        Dense model perplexity
    """
    logger.info("Computing dense baseline...")
    
    model = load_model(model_name, device)
    
    result = compute_perplexity(
        model,
        eval_dataloader,
        torch.device(device),
        use_sparse=False  # No sparsification
    )
    
    logger.info(f"Dense baseline PPL: {result.perplexity:.2f}")
    
    # Clean up
    del model
    from src import clear_memory
    clear_memory(torch.device(device))
    
    return result.perplexity


# =============================================================================
# Sparse Baseline / Utility Helpers
# =============================================================================

def resolve_target_modules(config: SPONConfig, default_modules: List[str]) -> List[str]:
    """Resolve which modules should be sparsified for this configuration."""
    modules = config.modules if config.modules else default_modules
    if not modules:
        raise ValueError(
            f"Configuration '{config.name}' does not define modules and no "
            "--default_sparse_modules were provided."
        )
    return modules


def compute_teal_baseline(
    model_name: str,
    eval_dataloader,
    device: str,
    sparsity: float,
    target_modules: List[str]
) -> float:
    """Compute sparse baseline perplexity (TEAL only, no SPON biases)."""
    logger.info(
        "Computing sparse baseline (TEAL-only) for sparsity=%s modules=%s",
        sparsity,
        target_modules,
    )

    model = load_model(model_name, device)
    eval_result = compute_perplexity(
        model,
        eval_dataloader,
        torch.device(device),
        use_sparse=True,
        sparsity=sparsity,
        spon_biases=None,
        target_modules=target_modules,
    )

    del model
    from src import clear_memory

    clear_memory(torch.device(device))
    return eval_result.perplexity


def compute_relative_params(model, num_spon_params: int) -> Optional[float]:
    """
    Compute relative SPON params against UNIF-ALL down_proj baseline for this model.
    """
    if num_spon_params < 0:
        return None

    try:
        ref_params = 0
        for layer in model.model.layers:
            ref_params += int(layer.mlp.down_proj.out_features)
        if ref_params <= 0:
            return None
        return float(num_spon_params) / float(ref_params)
    except Exception:
        return None


# =============================================================================
# Single Configuration Training & Evaluation
# =============================================================================

def train_and_evaluate_config(
    model_name: str,
    tokenizer,
    config: SPONConfig,
    sparsity: float,
    train_dataloader,
    eval_dataloader,
    args,
    run_manager,
    dense_ppl: Optional[float] = None,
    teal_ppl: Optional[float] = None
) -> ConfigResult:
    """
    Train SPON biases for a single configuration and evaluate.
    
    This is the core function that:
    1. Loads a fresh model
    2. Trains SPON biases (if config has any)
    3. Evaluates perplexity
    4. Computes comparison metrics
    5. Saves checkpoint
    
    Args:
        model_name: Model to use
        tokenizer: Tokenizer
        config: SPON configuration
        sparsity: Sparsity level
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        args: Command line arguments
        run_manager: RunManager for saving results
        dense_ppl: Dense baseline PPL (for comparison)
        teal_ppl: TEAL-only baseline PPL (for comparison)
    
    Returns:
        ConfigResult with all metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Configuration: {config.name} @ {sparsity:.0%} sparsity")
    target_modules = resolve_target_modules(config, args.default_sparse_modules)
    logger.info(f"Layers: {config.layer_mask}")
    logger.info(f"Modules: {target_modules}")
    logger.info(f"{'='*60}")
    
    # Load fresh model
    model = load_model(model_name, args.device)
    device = torch.device(args.device)
    
    start_time = time.time()
    training_loss_final = 0.0
    spon_biases = None
    num_spon_params = 0
    
    # Train SPON (if this config has biases)
    if config.name != "BASELINE-TEAL" and len(config.layer_mask) > 0:
        checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_manager.run_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"spon_{config.name}_s{sparsity:.2f}.pt"

        if args.skip_training:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"--skip_training set but checkpoint not found: {checkpoint_path}"
                )
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            spon_biases = {k: v.to(device) for k, v in checkpoint["biases"].items()}
            num_spon_params = int(sum(v.numel() for v in spon_biases.values()))
            training_loss_final = float(checkpoint.get("extra", {}).get("training_loss", 0.0))
            ckpt_modules = checkpoint.get("extra", {}).get("modules")
            if isinstance(ckpt_modules, list) and sorted(ckpt_modules) != sorted(target_modules):
                logger.warning(
                    "Checkpoint modules %s do not match current config modules %s",
                    ckpt_modules,
                    target_modules,
                )
                target_modules = ckpt_modules
        else:
            training_args = TrainingArgs(
                epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=args.device
            )
            
            # Ensure trainer and evaluator sparsify the same module set
            config_for_training = SPONConfig(
                name=config.name,
                layer_mask=config.layer_mask,
                modules=target_modules,
                capacity_multiplier=config.capacity_multiplier,
                description=config.description,
            )
            trainer = SPONTrainer(model, config_for_training, sparsity, training_args, tokenizer)
            
            # Log SPON parameter count
            num_spon_params = sum(p.numel() for p in trainer.spon_params)
            logger.info(f"Training {num_spon_params:,} SPON parameters...")
            
            # Train
            train_metrics = trainer.train(train_dataloader)
            training_loss_final = train_metrics['final_loss']
            
            # Log training metrics
            for step, loss in enumerate(train_metrics['training_loss']):
                run_manager.log_metric("training_loss", loss, step=step, config_name=config.name)
            
            # Get trained biases
            spon_biases = trainer.get_spon_biases()
            
            # Save checkpoint
            run_manager.save_checkpoint(
                spon_biases, 
                config.name, 
                sparsity,
                extra_info={
                    "training_loss": training_loss_final,
                    "modules": target_modules,
                    "layer_mask": config.layer_mask,
                }
            )
            
            logger.info(f"Training complete. Final loss: {training_loss_final:.4f}")
    else:
        logger.info("Baseline config - no SPON training needed")
    
    training_time = time.time() - start_time
    
    # Evaluate
    logger.info("Evaluating...")
    eval_result = compute_perplexity(
        model,
        eval_dataloader,
        device,
        use_sparse=True,
        sparsity=sparsity,
        spon_biases=spon_biases,
        target_modules=target_modules
    )
    
    logger.info(f"Perplexity: {eval_result.perplexity:.2f}")
    
    # Compute comparison metrics
    ppl_vs_dense = eval_result.perplexity / dense_ppl if dense_ppl else 0.0
    ppl_vs_teal = eval_result.perplexity / teal_ppl if teal_ppl else 0.0
    ppl_improvement = ((teal_ppl - eval_result.perplexity) / teal_ppl * 100) if teal_ppl else 0.0
    
    relative_params = compute_relative_params(model, num_spon_params)
    
    # Create result object
    result = ConfigResult(
        config_name=config.name,
        sparsity=sparsity,
        perplexity=eval_result.perplexity,
        loss=eval_result.loss,
        num_tokens=eval_result.num_tokens,
        num_spon_layers=len(config.layer_mask),
        num_spon_params=num_spon_params,
        relative_params=relative_params,
        training_loss_final=training_loss_final,
        training_time_seconds=training_time,
        ppl_vs_dense=ppl_vs_dense,
        ppl_vs_teal=ppl_vs_teal,
        ppl_improvement=ppl_improvement
    )
    
    # Log result
    run_manager.log_config_result(result)
    run_manager.log_metric("perplexity", eval_result.perplexity, config_name=config.name)
    
    # Clean up
    del model
    from src import clear_memory
    clear_memory(device)
    
    return result


# =============================================================================
# Main Experiment Loop
# =============================================================================

def main():
    """Main experiment function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Resolve auto device
    if args.device == "auto":
        from src import get_device
        args.device = str(get_device())
    logger.info(f"Using device: {args.device}")

    provider_resolution = resolve_model_provider(
        model_name=args.model,
        provider=args.model_provider,
        ollama_model=args.ollama_model,
        pull_ollama=args.pull_ollama,
        require_transformers_model=True,
    )
    log_provider_resolution(provider_resolution)

    needs_checkpoints = any(cfg != "BASELINE-TEAL" for cfg in args.configs)
    if args.skip_training and needs_checkpoints and args.checkpoint_dir is None:
        raise ValueError(
            "--skip_training requires --checkpoint_dir pointing to existing checkpoints."
        )
    
    logger.info("="*70)
    logger.info("SPON Allocation Sweep Experiment")
    logger.info("="*70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Configurations: {args.configs}")
    logger.info(f"Sparsity levels: {args.sparsity}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Default sparse modules: {args.default_sparse_modules}")
    logger.info("="*70)
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(args.experiment_name, args.output_dir)
    
    # Create run configuration
    run_config = RunConfig(
        model_name=args.model,
        sparsity=args.sparsity[0],  # Primary sparsity
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        block_size=args.block_size,
        calibration_samples=args.num_samples,
        seed=args.seed,
        extra={
            "all_sparsities": args.sparsity,
            "all_configs": args.configs,
            "default_sparse_modules": args.default_sparse_modules,
            "data_driven_overrides": args.data_driven_overrides,
            "model_provider_requested": provider_resolution.requested_provider,
            "model_provider_effective": provider_resolution.effective_provider,
            "ollama_model": provider_resolution.ollama_model_name,
            "provider_fallback_reason": provider_resolution.reason,
        }
    )
    
    # Start run
    run = exp_manager.start_run(config=run_config)
    run.log_message(f"Starting allocation sweep with {len(args.configs)} configs")
    
    # Load tokenizer (can be reused)
    logger.info(f"Loading tokenizer...")
    tokenizer = load_tokenizer(provider_resolution.hf_model_name)
    
    logger.info("Creating calibration dataloaders...")
    train_dataloader = create_calibration_dataloader(
        tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        dataset_subset=args.dataset_subset,
        split="train",
        # Keep ordering fixed for fair config-to-config comparisons.
        shuffle=False,
        seed=args.seed
    )
    
    # Determine eval samples: 0 -> num_samples//4, -1 -> large (full test set)
    if args.num_eval_samples == -1:
        eval_num_samples = 100000  # effectively the full test set
    elif args.num_eval_samples > 0:
        eval_num_samples = args.num_eval_samples
    else:
        eval_num_samples = args.num_samples // 4
    
    eval_dataloader = create_calibration_dataloader(
        tokenizer,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_samples=eval_num_samples,
        dataset_subset=args.dataset_subset,
        split="test",
        shuffle=False,
        seed=args.seed
    )
    
    # Compute dense baseline (optional but recommended)
    dense_ppl = None
    if args.dense_baseline:
        dense_ppl = compute_dense_baseline(
            provider_resolution.hf_model_name, tokenizer, eval_dataloader, args.device
        )
        run.log_metric("dense_baseline_ppl", dense_ppl)
    
    # Load allocation configs
    config_path = Path(__file__).resolve().parents[2] / "configs" / "allocation_configs.yaml"
    builder = AllocationBuilder(
        str(config_path),
        data_driven_overrides_path=args.data_driven_overrides
    )
    
    # Track TEAL baselines keyed by (sparsity, module set)
    teal_ppls: Dict[Tuple[float, Tuple[str, ...]], float] = {}
    
    # Main experiment loop
    all_results = []
    
    for sparsity in args.sparsity:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Sparsity Level: {sparsity:.0%}")
        logger.info(f"{'#'*70}\n")
        
        run.log_message(f"Starting sparsity={sparsity}")
        
        for config_name in args.configs:
            # Build config for this model
            try:
                config = builder.build_config(config_name, args.model)
            except ValueError as e:
                logger.error(f"Failed to build config '{config_name}': {e}")
                continue
            
            target_modules = resolve_target_modules(config, args.default_sparse_modules)
            teal_key = (sparsity, tuple(sorted(target_modules)))

            # Compute module-matched TEAL baseline once per sparsity/module-set.
            if teal_key not in teal_ppls:
                teal_ppls[teal_key] = compute_teal_baseline(
                    provider_resolution.hf_model_name,
                    eval_dataloader,
                    args.device,
                    sparsity,
                    target_modules,
                )
                run.log_metric(
                    "teal_baseline_ppl",
                    teal_ppls[teal_key],
                    config_name=f"modules={','.join(target_modules)}|s={sparsity}",
                )
            teal_ppl = teal_ppls[teal_key]
            
            # Train and evaluate
            try:
                result = train_and_evaluate_config(
                    provider_resolution.hf_model_name,
                    tokenizer,
                    config,
                    sparsity,
                    train_dataloader,
                    eval_dataloader,
                    args,
                    run,
                    dense_ppl=dense_ppl,
                    teal_ppl=teal_ppl
                )
                
                all_results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed on {config_name} @ {sparsity}: {e}")
                import traceback
                traceback.print_exc()
                run.log_message(f"ERROR: {config_name} failed: {e}", level="error")
                continue
    
    # Finish run
    summary = run.finish()
    
    # Aggregate across all runs in this experiment
    stats = exp_manager.aggregate_results()
    
    # Export LaTeX table
    exp_manager.export_latex_table(
        "allocation_results",
        metric_keys=["perplexity", "relative_params", "ppl_improvement"],
        caption="SPON Allocation Results: Perplexity and Parameter Efficiency"
    )
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"Total configurations tested: {len(all_results)}")
    logger.info(f"Best config: {summary.best_config} (PPL={summary.best_perplexity:.2f})")
    logger.info(f"Pareto-optimal: {summary.pareto_configs}")
    logger.info(f"Results saved to: {run.run_dir}")
    logger.info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
