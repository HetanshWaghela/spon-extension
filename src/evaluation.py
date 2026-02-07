"""
Evaluation utilities for SPON experiments.

Computes perplexity, downstream task accuracy, and analysis metrics.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass 
class EvalResults:
    """Container for evaluation results."""
    perplexity: float
    loss: float
    num_tokens: int
    config_name: str = ""
    sparsity: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "perplexity": self.perplexity,
            "loss": self.loss,
            "num_tokens": self.num_tokens,
            "config_name": self.config_name,
            "sparsity": self.sparsity
        }


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    use_sparse: bool = False,
    sparsity: float = 0.5,
    spon_biases: Optional[Dict[str, torch.Tensor]] = None,
    target_modules: List[str] = ["down_proj"]
) -> EvalResults:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Language model
        dataloader: DataLoader with tokenized data
        device: Device to run on
        use_sparse: Whether to use sparse forward pass
        sparsity: Sparsity level if use_sparse
        spon_biases: SPON biases to apply if use_sparse
        target_modules: Which modules to sparsify (default: ["down_proj"])
    
    Returns:
        EvalResults with perplexity and related metrics
    
    NOTE: This function uses hooks to apply sparsification, so it does NOT
    permanently modify the model. The model is restored after evaluation.
    
    IMPORTANT: When use_sparse=True, this function applies IDENTICAL sparsification
    to what SPONTrainer uses during training:
    - Sparsifies the INPUT to target modules (magnitude-based top-k)
    - Recomputes the linear output with sparse input
    - Adds SPON biases to the output (if provided)
    
    This ensures train/eval consistency, which is critical for valid measurements.
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

    if device is None:
        from . import get_device
        device = get_device()

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Use the SAME sparsification hooks that training uses
    # This guarantees train/eval consistency
    hooks_dict = {}
    if use_sparse:
        from .sparse_forward import register_sparsification_hooks, remove_hooks
        hooks_dict = register_sparsification_hooks(
            model,
            sparsity=sparsity,
            target_modules=target_modules,
            spon_biases=spon_biases
        )
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                input_ids = batch["input_ids"].to(device)
                
                # Shift for next-token prediction
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += labels.numel()
    finally:
        # Always remove hooks to restore model
        if hooks_dict:
            from .sparse_forward import remove_hooks
            remove_hooks(hooks_dict)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return EvalResults(
        perplexity=perplexity,
        loss=avg_loss,
        num_tokens=total_tokens,
        sparsity=sparsity if use_sparse else 0.0
    )


def compute_hidden_state_shift(
    model: torch.nn.Module,
    dataloader: DataLoader,
    sparsity: float,
    spon_biases: Optional[Dict[str, torch.Tensor]] = None,
    layer_indices: Optional[List[int]] = None,
    device: torch.device = None,
    target_modules: List[str] = ["down_proj"]
) -> Dict[str, float]:
    """
    Compute L2 shift between dense and sparse hidden states.
    
    This measures how much SPON reduces the representation drift
    caused by sparsification. Key metric for understanding SPON effectiveness.
    
    Args:
        model: Language model
        dataloader: DataLoader with tokenized data
        sparsity: Sparsity level for sparse forward
        spon_biases: Optional SPON biases to apply
        layer_indices: Which layers to analyze (default: all)
        device: Device to run on
        target_modules: Which modules to sparsify
    
    Returns:
        Dictionary mapping layer_module key to average L2 shift
        
    Example:
        shifts = compute_hidden_state_shift(model, dataloader, sparsity=0.5)
        # shifts = {"layer_0_down_proj": 0.15, "layer_1_down_proj": 0.23, ...}
    """
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError(f"Sparsity must be between 0 and 1, got {sparsity}")

    if device is None:
        from . import get_device
        device = get_device()

    from .sparse_forward import (
        ActivationCollector, 
        register_sparsification_hooks, 
        remove_hooks
    )
    
    model.eval()
    
    if layer_indices is None:
        layer_indices = list(range(len(model.model.layers)))
    
    # Step 1: Collect dense activations (no sparsification)
    dense_collector = ActivationCollector()
    dense_collector.register(model, layer_indices, target_modules)
    
    dense_activations = {}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            _ = model(input_ids)
            
            for key, acts in dense_collector.get_activations().items():
                if key not in dense_activations:
                    dense_activations[key] = []
                dense_activations[key].append(acts["output"])
            
            dense_collector.clear()
    
    dense_collector.remove_hooks()
    
    # Aggregate dense activations
    for key in dense_activations:
        dense_activations[key] = torch.cat(dense_activations[key], dim=0)
    
    # Step 2: Collect sparse activations (with sparsification + optional SPON)
    # Use hooks for non-destructive sparsification
    sparse_hooks = register_sparsification_hooks(
        model, sparsity, target_modules, spon_biases
    )
    
    sparse_collector = ActivationCollector()
    sparse_collector.register(model, layer_indices, target_modules)
    
    sparse_activations = {}
    try:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                _ = model(input_ids)
                
                for key, acts in sparse_collector.get_activations().items():
                    if key not in sparse_activations:
                        sparse_activations[key] = []
                    sparse_activations[key].append(acts["output"])
                
                sparse_collector.clear()
    finally:
        # Always clean up hooks
        sparse_collector.remove_hooks()
        remove_hooks(sparse_hooks)
    
    # Aggregate sparse activations
    for key in sparse_activations:
        sparse_activations[key] = torch.cat(sparse_activations[key], dim=0)
    
    # Step 3: Compute L2 shifts
    l2_shifts = {}
    for key in dense_activations:
        if key in sparse_activations:
            dense = dense_activations[key]
            sparse = sparse_activations[key]
            
            # L2 norm per token, then average
            l2_diff = torch.norm(dense - sparse, p=2, dim=-1)
            l2_shifts[key] = l2_diff.mean().item()
    
    return l2_shifts


def run_lm_eval_harness(
    model_path: str,
    output_dir: str,
    tasks: List[str] = ["mmlu", "arc_easy", "hellaswag"],
    batch_size: str = "auto",
    device: str = "auto"
) -> Dict[str, float]:
    """
    Run lm-eval-harness for downstream task evaluation.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        tasks: List of evaluation tasks
        batch_size: Batch size (or "auto")
        device: Device to run on
    
    Returns:
        Dictionary of task -> accuracy
    """
    if device == "auto":
        from . import get_device
        device = str(get_device())

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", batch_size,
        "--output_path", output_dir,
        "--log_samples"
    ]
    
    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"lm-eval failed: {e.stderr}")
        raise
    
    # Parse results
    results_file = Path(output_dir) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        return {task: results.get(task, {}).get("acc", 0.0) for task in tasks}
    
    return {}


def compute_pareto_frontier(
    results: List[Dict],
    x_key: str = "relative_params",
    y_key: str = "perplexity"
) -> List[Dict]:
    """
    Compute Pareto frontier from results.
    
    A point is Pareto-optimal if no other point dominates it
    (lower x AND lower y).
    
    Args:
        results: List of result dictionaries
        x_key: Key for x-axis (e.g., parameter count)
        y_key: Key for y-axis (e.g., perplexity, lower is better)
    
    Returns:
        List of Pareto-optimal results
    """
    # Sort by x
    sorted_results = sorted(results, key=lambda r: r[x_key])
    
    pareto = []
    min_y = float('inf')
    
    for result in sorted_results:
        if result[y_key] < min_y:
            pareto.append(result)
            min_y = result[y_key]
    
    return pareto


def compare_configurations(
    results: Dict[str, EvalResults],
    baseline_name: str = "UNIF-ALL"
) -> Dict[str, Dict]:
    """
    Compare configurations against a baseline.
    
    Args:
        results: Dictionary mapping config name to EvalResults
        baseline_name: Name of baseline configuration
    
    Returns:
        Comparison statistics
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline {baseline_name} not found in results")
    
    baseline = results[baseline_name]
    comparisons = {}
    
    for name, result in results.items():
        if name == baseline_name:
            continue
        
        comparisons[name] = {
            "perplexity": result.perplexity,
            "baseline_perplexity": baseline.perplexity,
            "ppl_ratio": result.perplexity / baseline.perplexity,
            "ppl_diff": result.perplexity - baseline.perplexity,
            "improvement": (baseline.perplexity - result.perplexity) / baseline.perplexity * 100
        }
    
    return comparisons
