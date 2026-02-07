#!/usr/bin/env python3
"""
Layer Importance Analysis

Implements principled methods for determining layer importance:
1. Shapley value-based importance
2. ILA (Identifying Important Layers for Alignment)
3. Cornerstone layer detection
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def compute_layer_shapley_values(
    model: torch.nn.Module,
    eval_fn: Callable[[torch.nn.Module, List[int]], float],
    n_samples: int = 500,
    verbose: bool = True,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Compute Shapley values for each layer using Monte Carlo sampling.
    
    Following Zhang et al. (arXiv:2409.14381), this provides a principled
    measure of each layer's contribution to model performance.
    
    Shapley value for layer i:
    φ_i = (1/n!) Σ_π [v(S_π^i ∪ {i}) - v(S_π^i)]
    
    Where S_π^i is the set of layers before i in permutation π.
    
    Args:
        model: The language model
        eval_fn: Function that takes (model, active_layers) and returns performance metric
                 (higher is better)
        n_samples: Number of permutation samples
        verbose: Whether to show progress bar
        seed: Random seed for reproducibility
    
    Returns:
        Array of Shapley values per layer
    """
    if seed is not None:
        np.random.seed(seed)
    n_layers = len(model.model.layers)
    shapley_values = np.zeros(n_layers)
    
    iterator = tqdm(range(n_samples), desc="Computing Shapley values") if verbose else range(n_samples)
    
    for _ in iterator:
        # Sample random permutation
        perm = np.random.permutation(n_layers)
        
        # Compute marginal contributions
        coalition = set()
        prev_perf = eval_fn(model, list(coalition))
        
        for layer_i in perm:
            # Add layer to coalition
            coalition.add(layer_i)
            curr_perf = eval_fn(model, list(coalition))
            
            # Marginal contribution
            shapley_values[layer_i] += (curr_perf - prev_perf) / n_samples
            prev_perf = curr_perf
    
    return shapley_values


def compute_layer_sensitivity_fast(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None
) -> np.ndarray:
    """
    Fast approximation of layer sensitivity using gradient-based importance.
    
    This is much faster than Shapley sampling but less principled.
    Uses the gradient norm w.r.t. layer outputs as a proxy for importance.
    
    Returns:
        Array of sensitivity scores per layer
    """
    if device is None:
        from src import get_device
        device = get_device()
    n_layers = len(model.model.layers)
    sensitivities = np.zeros(n_layers)
    
    # Register hooks to capture gradients
    gradients = {}
    
    def make_hook(layer_idx):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[layer_idx] = grad_output[0].detach().norm().item()
        return hook
    
    handles = []
    for i, layer in enumerate(model.model.layers):
        handle = layer.register_full_backward_hook(make_hook(i))
        handles.append(handle)
    
    model.train()  # Need gradients
    
    n_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        
        # Forward
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Simple loss: predict next token
        labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward
        loss.backward()
        
        # Accumulate gradients
        for i, grad in gradients.items():
            sensitivities[i] += grad
        
        gradients.clear()
        model.zero_grad()
        n_batches += 1
        
        if n_batches >= 10:  # Quick estimate
            break
    
    # Cleanup
    for handle in handles:
        handle.remove()
    
    model.eval()
    
    # Normalize
    if n_batches == 0:
        raise ValueError("No batches available for gradient sensitivity computation.")
    sensitivities /= n_batches
    return sensitivities


def identify_cornerstone_layers(
    model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float = 0.5,
    device: torch.device = None
) -> List[int]:
    """
    Identify "cornerstone" layers whose removal causes model collapse.
    
    Following Zhang et al., cornerstone layers exhibit dominant contribution—
    removing one causes PPL to increase by more than the threshold ratio.
    
    Args:
        model: Language model
        dataloader: Evaluation data
        threshold: Minimum PPL increase ratio to be considered cornerstone
        device: Device to run on
    
    Returns:
        List of cornerstone layer indices
    """
    if device is None:
        from src import get_device
        device = get_device()
    from src.evaluation import compute_perplexity
    
    n_layers = len(model.model.layers)
    
    # Baseline PPL
    baseline_ppl = compute_perplexity(model, dataloader, device).perplexity
    logger.info(f"Baseline PPL: {baseline_ppl:.2f}")
    
    cornerstone_layers = []
    ppl_increases = np.zeros(n_layers)
    
    for layer_idx in tqdm(range(n_layers), desc="Testing layer removal"):
        layer = model.model.layers[layer_idx]
        
        # Save original forward
        orig_forward = layer.forward
        
        # Replace with skip connection
        def skip_forward(hidden_states, *args, **kwargs):
            # Return input unchanged, with None for optional outputs
            # Adjust based on actual return signature
            return (hidden_states, None, None)
        
        layer.forward = skip_forward
        
        # Compute PPL
        try:
            removed_ppl = compute_perplexity(model, dataloader, device).perplexity
        except Exception as e:
            logger.warning(f"Error evaluating layer {layer_idx}: {e}")
            removed_ppl = float('inf')
        
        # Restore
        layer.forward = orig_forward
        
        # Check if cornerstone
        ppl_increase = (removed_ppl - baseline_ppl) / baseline_ppl
        ppl_increases[layer_idx] = ppl_increase
        
        if ppl_increase > threshold:
            cornerstone_layers.append(layer_idx)
            logger.info(f"Layer {layer_idx}: PPL increase = {ppl_increase:.2%} (CORNERSTONE)")
        else:
            logger.debug(f"Layer {layer_idx}: PPL increase = {ppl_increase:.2%}")
    
    return cornerstone_layers


def compute_ila_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_steps: int = 100,
    lr: float = 0.01,
    device: torch.device = None
) -> np.ndarray:
    """
    ILA: Identifying Important Layers for Alignment.
    
    Learns binary masks indicating which layers are most important.
    Based on arXiv:2410.17875.
    
    Args:
        model: Language model
        dataloader: Training data
        n_steps: Number of optimization steps
        lr: Learning rate for mask optimization
        device: Device
    
    Returns:
        Array of importance scores (0-1) per layer
    """
    if device is None:
        from src import get_device
        device = get_device()
    n_layers = len(model.model.layers)
    
    # Initialize learnable mask
    mask = torch.nn.Parameter(torch.ones(n_layers, device=device))
    optimizer = torch.optim.Adam([mask], lr=lr)
    
    # Freeze model parameters (restore at end)
    prev_requires_grad = [param.requires_grad for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad = False
    
    # Register scaling hooks
    def make_scale_hook(layer_idx):
        def hook(module, input, output):
            # Scale output by mask value
            scale = torch.sigmoid(mask[layer_idx])  # Constrain to [0, 1]
            if isinstance(output, tuple):
                return (output[0] * scale,) + output[1:]
            return output * scale
        return hook
    
    handles = []
    for i, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(make_scale_hook(i))
        handles.append(handle)
    
    model.train()
    
    step = 0
    for batch in dataloader:
        if step >= n_steps:
            break
        
        input_ids = batch["input_ids"].to(device)
        
        # Forward
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Compute loss
        labels = input_ids[:, 1:].contiguous()
        ce_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Add sparsity regularization (encourage some layers to be masked out)
        sparsity_loss = torch.sigmoid(mask).mean() * 0.1
        
        loss = ce_loss + sparsity_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
    
    # Cleanup
    for handle in handles:
        handle.remove()
    for param, was_trainable in zip(model.parameters(), prev_requires_grad):
        param.requires_grad = was_trainable

    model.eval()
    
    # Return importance scores
    importance = torch.sigmoid(mask).detach().cpu().numpy()
    return importance


def get_optimal_layers(
    importance_scores: np.ndarray,
    method: str = "top_k",
    k: Optional[int] = None,
    fraction: float = 0.5,
    threshold: float = 0.5
) -> List[int]:
    """
    Select optimal layers based on importance scores.
    
    Args:
        importance_scores: Array of importance scores per layer
        method: Selection method ("top_k", "threshold", "elbow")
        k: Number of layers for top_k (if None, uses fraction)
        fraction: Fraction of layers to select
        threshold: Threshold for importance (for threshold method)
    
    Returns:
        List of selected layer indices
    """
    n_layers = len(importance_scores)
    
    if method == "top_k":
        if k is None:
            k = max(1, int(n_layers * fraction))
        indices = np.argsort(importance_scores)[::-1][:k]
        return sorted(indices.tolist())
    
    elif method == "threshold":
        return [i for i, score in enumerate(importance_scores) if score > threshold]
    
    elif method == "elbow":
        # Find elbow point in sorted importance scores
        sorted_scores = np.sort(importance_scores)[::-1]
        
        # Compute second derivative to find elbow
        diffs = np.diff(sorted_scores)
        elbow_idx = np.argmin(diffs) + 1
        
        # Select layers above elbow threshold
        elbow_threshold = sorted_scores[min(elbow_idx, len(sorted_scores) - 1)]
        return [i for i, score in enumerate(importance_scores) if score >= elbow_threshold]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_layer_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
    methods: List[str] = ["gradient", "ila"]
) -> Dict[str, np.ndarray]:
    """
    Run multiple layer importance methods and return results.
    
    Args:
        model: Language model
        dataloader: Evaluation data
        device: Device
        methods: List of methods to run
    
    Returns:
        Dictionary mapping method name to importance scores
    """
    if device is None:
        from src import get_device
        device = get_device()
    results = {}
    
    if "gradient" in methods:
        logger.info("Computing gradient-based sensitivity...")
        results["gradient"] = compute_layer_sensitivity_fast(model, dataloader, device)
    
    if "ila" in methods:
        logger.info("Computing ILA importance...")
        results["ila"] = compute_ila_importance(model, dataloader, device=device)
    
    if "shapley" in methods:
        logger.info("Computing Shapley values (this may take a while)...")

        def eval_with_layers(model, active_layers):
            from src.evaluation import compute_perplexity

            n_layers = len(model.model.layers)
            active_set = set(active_layers)
            inactive_layers = [i for i in range(n_layers) if i not in active_set]
            handles = []

            def make_identity_hook():
                def hook(module, inputs, output):
                    hidden_states = inputs[0]
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states

                return hook

            for idx in inactive_layers:
                handle = model.model.layers[idx].register_forward_hook(make_identity_hook())
                handles.append(handle)

            try:
                ppl = compute_perplexity(model, dataloader, device).perplexity
            finally:
                for handle in handles:
                    handle.remove()

            # Higher is better for Shapley value aggregation.
            return -ppl
        
        results["shapley"] = compute_layer_shapley_values(model, eval_with_layers, n_samples=100)
    
    return results
