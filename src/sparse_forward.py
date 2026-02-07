#!/usr/bin/env python3
"""
=============================================================================
Sparse Forward Pass Implementation (TEAL-style)
=============================================================================

This module implements magnitude-based activation sparsification following
the TEAL paper (ICLR 2025 Spotlight).

Key Concept:
    In a normal forward pass:       Y = W @ X
    In a sparse forward pass:       Y = W @ S(X)
    
    Where S(X) is a sparsification function that zeros out small activations.
    
    TEAL uses "magnitude-based" sparsification:
    - Compute absolute value of all activations
    - Keep only the top-k% largest values
    - Zero out the rest

Why This Matters:
    - Sparse activations = fewer multiplications = faster inference
    - But... zeroing activations loses information
    - SPON biases compensate for this information loss

Module Contents:
    - MagnitudeSparsifier: Callable class for sparsification
    - magnitude_sparsify(): Core sparsification function
    - SparseLinear: Drop-in replacement for nn.Linear with sparsification
    - ActivationCollector: Utility to record activations for analysis

Usage:
    # Basic sparsification
    from src.sparse_forward import magnitude_sparsify
    x_sparse = magnitude_sparsify(x, sparsity=0.5)  # Keep top 50%
    
    # With hooks (non-destructive)
    hooks = register_sparsification_hooks(model, sparsity=0.5)
    output = model(input_ids)  # Forward uses sparse activations
    remove_hooks(hooks)  # Restore original behavior

Author: [Your Name]
References:
    - TEAL: Training-free Activation Sparsification for LLMs (ICLR 2025)
    - SPON: Resting Neurons, Active Insights (arXiv:2512.12744)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from functools import partial


# =============================================================================
# Core Sparsification Functions
# =============================================================================

def magnitude_sparsify(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Apply magnitude-based sparsification to activations.
    
    Algorithm:
        1. Compute |x| for all activations
        2. Find the k-th largest value (where k = (1-sparsity) * num_activations)
        3. Create binary mask: 1 where |x| >= threshold, 0 elsewhere
        4. Return x * mask
    
    Args:
        x: Input tensor of shape (..., hidden_dim)
           Typically (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        sparsity: Fraction of activations to zero out
                  0.0 = keep all, 0.5 = keep top 50%, 1.0 = zero all
    
    Returns:
        Sparsified tensor with same shape as input.
        Approximately sparsity fraction of values will be zero.
    
    Example:
        >>> x = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        >>> magnitude_sparsify(x.unsqueeze(0), sparsity=0.5)
        tensor([[0.0, 0.9, 0.0, 0.8, 0.0, 0.7]])  # Kept top 50%
    
    Note:
        Sparsification is applied per-sample (row) to preserve relative
        magnitudes within each sample. This is important for batch processing.
    """
    # Edge cases
    if sparsity <= 0:
        return x  # No sparsification
    if sparsity >= 1:
        return torch.zeros_like(x)  # Zero everything
    
    # Calculate how many values to KEEP (not remove)
    # k = number of values to keep
    k = max(1, int(x.shape[-1] * (1 - sparsity)))
    
    # Get absolute values for magnitude comparison
    magnitudes = x.abs()
    
    # Find threshold: the k-th largest value becomes our cutoff
    # torch.kthvalue finds the k-th SMALLEST, so we invert the logic
    # We want: values in top-k by magnitude => keep them
    # kthvalue(tensor, k) returns the k-th smallest
    # So kthvalue(tensor, n-k+1) returns the (n-k+1)-th smallest = k-th largest
    
    n_elements = x.shape[-1]
    kth_position = n_elements - k + 1  # Position of k-th largest
    
    # Handle different tensor dimensions
    # We always sparsify along the last dimension (hidden_dim)
    if x.dim() == 1:
        # Single vector: (hidden_dim,)
        threshold, _ = torch.kthvalue(magnitudes, kth_position)
        threshold = threshold.unsqueeze(0)
    elif x.dim() == 2:
        # Batch of vectors: (batch, hidden_dim)
        threshold, _ = torch.kthvalue(magnitudes, kth_position, dim=-1, keepdim=True)
    elif x.dim() == 3:
        # Sequence batch: (batch, seq_len, hidden_dim)
        threshold, _ = torch.kthvalue(magnitudes, kth_position, dim=-1, keepdim=True)
    else:
        # Higher dimensions: flatten, compute, reshape
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        magnitudes_flat = magnitudes.view(-1, x.shape[-1])
        threshold, _ = torch.kthvalue(magnitudes_flat, kth_position, dim=-1, keepdim=True)
        threshold = threshold.view(*original_shape[:-1], 1)
    
    # Create binary mask: keep values >= threshold
    # Using >= ensures we keep exactly k values (or more in case of ties)
    # IMPORTANT: Match mask dtype to input dtype to avoid precision/performance issues
    mask = (magnitudes >= threshold).to(dtype=x.dtype)
    
    # Apply mask while preserving gradients through kept values
    return x * mask


class MagnitudeSparsifier:
    """
    Callable wrapper for magnitude-based sparsification.
    
    Useful when you need to pass sparsification as a function argument
    or want to easily change sparsity levels.
    
    Attributes:
        sparsity: Current sparsity level (0.0 to 1.0)
    
    Example:
        sparsifier = MagnitudeSparsifier(sparsity=0.5)
        x_sparse = sparsifier(x)
        
        # Change sparsity later
        sparsifier.update_sparsity(0.6)
    """
    
    def __init__(self, sparsity: float = 0.5):
        """
        Initialize sparsifier.
        
        Args:
            sparsity: Fraction of activations to zero out (default: 0.5 = keep top 50%)
        """
        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")
        self.sparsity = sparsity
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sparsification to input tensor."""
        return magnitude_sparsify(x, self.sparsity)
    
    def update_sparsity(self, sparsity: float):
        """Update sparsity level."""
        if not 0 <= sparsity <= 1:
            raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")
        self.sparsity = sparsity


# =============================================================================
# Sparse Linear Layer
# =============================================================================

class SparseLinear(nn.Module):
    """
    Linear layer with input sparsification and optional SPON bias.
    
    This is a drop-in replacement for nn.Linear that:
    1. Sparsifies the INPUT before matrix multiplication
    2. Adds an optional SPON bias to the OUTPUT
    
    Computation:
        Y = W @ S(X) + b_original + b_spon
        
    Where:
        - W: Weight matrix (from original linear layer)
        - X: Input activations
        - S(X): Sparsified input
        - b_original: Original bias (if any)
        - b_spon: SPON bias (learned compensation)
    
    Note:
        The SPON paper applies sparsification to the INPUT of down_proj,
        not the output. This is different from some other sparsification
        methods that sparsify outputs.
    
    Attributes:
        weight: Original weight matrix
        original_bias: Original bias (may be None)
        sparsity: Sparsity level for input
        spon_bias: SPON bias tensor (may be None)
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        sparsity: float = 0.5,
        spon_bias: Optional[nn.Parameter] = None
    ):
        """
        Initialize sparse linear layer.
        
        Args:
            original_linear: The nn.Linear layer to wrap
            sparsity: Fraction of input activations to zero
            spon_bias: Optional SPON bias parameter to add to output
        """
        super().__init__()
        
        # Copy weight (keep reference for memory efficiency)
        self.weight = original_linear.weight
        self.original_bias = original_linear.bias
        
        # Sparsification settings
        self.sparsity = sparsity
        self.spon_bias = spon_bias
        
        # Store dimensions for reference
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparsification.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Step 1: Sparsify input
        x_sparse = magnitude_sparsify(x, self.sparsity)
        
        # Step 2: Linear transformation (W @ x + b)
        out = F.linear(x_sparse, self.weight, self.original_bias)
        
        if self.spon_bias is not None:
            assert self.spon_bias.shape[0] == self.out_features, (
                f"SPON bias size {self.spon_bias.shape[0]} != out_features {self.out_features}"
            )
            out = out + self.spon_bias
        
        return out
    
    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'sparsity={self.sparsity}, has_spon={self.spon_bias is not None}'
        )


# =============================================================================
# Model Patching Functions
# =============================================================================

def patch_model_for_sparse_forward(
    model: nn.Module,
    sparsity: float,
    target_modules: Optional[List[str]] = None,
    spon_biases: Optional[Dict[str, nn.Parameter]] = None
) -> nn.Module:
    """
    Patch a model to use sparse forward passes.
    
    WARNING: This PERMANENTLY modifies the model by replacing Linear layers
    with SparseLinear layers. Use register_sparsification_hooks() for
    non-destructive sparsification.
    
    Args:
        model: HuggingFace model (e.g., LlamaForCausalLM)
        sparsity: Sparsity level (fraction to zero out)
        target_modules: List of module names to sparsify
                       Options: ["down_proj", "up_proj", "gate_proj", 
                                "q_proj", "k_proj", "v_proj", "o_proj"]
                       Defaults to ["down_proj"] if None.
        spon_biases: Optional dict mapping "layer_{i}_{module}" to bias tensors
    
    Returns:
        Modified model (same object, modified in-place)
    
    Example:
        model = patch_model_for_sparse_forward(model, sparsity=0.5)
        # Now all forward passes use sparse activations
    """
    spon_biases = spon_biases or {}
    target_modules = target_modules or ["down_proj"]
    
    for layer_idx, layer in enumerate(model.model.layers):
        for module_name in target_modules:
            # Determine parent module (MLP or Attention)
            if module_name in ["down_proj", "up_proj", "gate_proj"]:
                parent = layer.mlp
            elif module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                parent = layer.self_attn
            else:
                continue
            
            # Get original linear layer
            original = getattr(parent, module_name)
            
            # Get SPON bias if it exists
            bias_key = f"layer_{layer_idx}_{module_name}"
            spon_bias = spon_biases.get(bias_key)
            
            # Replace with sparse version
            sparse_linear = SparseLinear(original, sparsity, spon_bias)
            setattr(parent, module_name, sparse_linear)
    
    return model


# =============================================================================
# Hook-based Sparsification (Non-destructive)
# =============================================================================

class SparsificationHook:
    """
    Forward hook for applying sparsification without modifying model structure.
    
    This is the PREFERRED method for experiments because:
    1. Non-destructive: Original model is preserved
    2. Reversible: Hooks can be removed to restore original behavior
    3. Debuggable: Can inspect input/output activations
    
    How it works:
        When registered on a module, this hook intercepts the forward pass
        and modifies the output to simulate sparse input processing.
    
    Attributes:
        sparsity: Sparsity level
        spon_bias: Optional SPON bias to add
        input_activations: Last seen input (for debugging)
        sparse_activations: Last seen sparse input (for debugging)
    """
    
    def __init__(self, sparsity: float, spon_bias: Optional[torch.Tensor] = None):
        """
        Initialize hook.
        
        Args:
            sparsity: Fraction of activations to zero
            spon_bias: Optional SPON bias tensor
        """
        self.sparsity = sparsity
        self.spon_bias = spon_bias
        
        # For debugging/analysis
        self.input_activations = None
        self.sparse_activations = None
    
    def __call__(
        self, 
        module: nn.Module, 
        inputs: Tuple[torch.Tensor, ...], 
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Hook function called after module's forward().
        
        Note: We hook the OUTPUT and recompute with sparse input.
        This is because forward hooks can't modify inputs directly.
        
        Args:
            module: The module being hooked (e.g., down_proj Linear)
            inputs: Tuple of inputs to the module
            output: Original output from module
        
        Returns:
            Modified output computed with sparse input + SPON bias
        """
        # Store original input for analysis
        self.input_activations = inputs[0].detach()
        
        # Sparsify the input
        sparse_input = magnitude_sparsify(inputs[0], self.sparsity)
        self.sparse_activations = sparse_input.detach()
        
        # Recompute output with sparse input
        # This is equivalent to: W @ S(X) + b
        sparse_output = F.linear(sparse_input, module.weight, module.bias)
        
        if self.spon_bias is not None:
            assert self.spon_bias.numel() == sparse_output.shape[-1], (
                f"SPON bias size {self.spon_bias.numel()} != output dim {sparse_output.shape[-1]}"
            )
            bias_casted = self.spon_bias.to(device=sparse_output.device, dtype=sparse_output.dtype)
            sparse_output = sparse_output + bias_casted
        
        return sparse_output


def register_sparsification_hooks(
    model: nn.Module,
    sparsity: float,
    target_modules: Optional[List[str]] = None,
    spon_biases: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, dict]:
    """
    Register forward hooks for sparsification (NON-DESTRUCTIVE).
    
    This is the recommended approach for experiments:
    - Hooks can be removed to restore original behavior
    - Multiple configurations can be tested on same model
    - Useful for A/B comparisons
    
    Args:
        model: HuggingFace model
        sparsity: Sparsity level
        target_modules: Modules to hook. Defaults to ["down_proj"] if None.
        spon_biases: Optional SPON biases
    
    Returns:
        Dictionary mapping module keys to {"hook": SparsificationHook, "handle": handle}
        Keep this to remove hooks later!
    
    Example:
        # Apply hooks
        hooks = register_sparsification_hooks(model, sparsity=0.5)
        
        # Run inference with sparsification
        output = model(input_ids)
        
        # Remove hooks to restore original behavior
        remove_hooks(hooks)
    """
    spon_biases = spon_biases or {}
    target_modules = target_modules or ["down_proj"]
    hooks = {}
    
    for layer_idx, layer in enumerate(model.model.layers):
        for module_name in target_modules:
            # Get the module to hook
            if module_name in ["down_proj", "up_proj", "gate_proj"]:
                module = getattr(layer.mlp, module_name)
            elif module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                module = getattr(layer.self_attn, module_name)
            else:
                continue
            
            # Get SPON bias if exists
            bias_key = f"layer_{layer_idx}_{module_name}"
            spon_bias = spon_biases.get(bias_key)
            
            # Create and register hook
            hook = SparsificationHook(sparsity, spon_bias)
            handle = module.register_forward_hook(hook)
            
            hooks[bias_key] = {"hook": hook, "handle": handle}
    
    return hooks


def remove_hooks(hooks: Dict[str, dict]):
    """
    Remove registered hooks to restore original model behavior.
    
    Args:
        hooks: Dictionary returned by register_sparsification_hooks()
    """
    for hook_info in hooks.values():
        hook_info["handle"].remove()


# =============================================================================
# Activation Collection Utilities
# =============================================================================

class ActivationCollector:
    """
    Utility class to collect activations for analysis.
    
    Useful for:
    - Comparing dense vs sparse activations
    - Computing representation shifts
    - Debugging sparsification behavior
    
    Example:
        collector = ActivationCollector()
        collector.register(model, layer_indices=[0, 1, 2])
        
        # Run forward pass
        _ = model(input_ids)
        
        # Get activations
        acts = collector.get_activations()
        # acts["layer_0_down_proj"]["input"] = input tensor
        # acts["layer_0_down_proj"]["output"] = output tensor
        
        # Clean up
        collector.remove_hooks()
    """
    
    def __init__(self):
        """Initialize collector."""
        self.activations: Dict[str, Dict[str, torch.Tensor]] = {}
        self.handles: List = []
    
    def register(
        self, 
        model: nn.Module, 
        layer_indices: Optional[List[int]] = None, 
        module_names: List[str] = ["down_proj"]
    ):
        """
        Register hooks to collect activations.
        
        Args:
            model: Model to instrument
            layer_indices: Which layers to collect (default: all)
            module_names: Which modules to collect
        """
        layers = model.model.layers
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        
        for layer_idx in layer_indices:
            layer = layers[layer_idx]
            
            for module_name in module_names:
                # Get module
                if module_name in ["down_proj", "up_proj", "gate_proj"]:
                    module = getattr(layer.mlp, module_name)
                elif module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    module = getattr(layer.self_attn, module_name)
                else:
                    continue
                
                key = f"layer_{layer_idx}_{module_name}"
                
                # Create hook that captures activations
                # Using a factory function to avoid closure issues
                def make_hook(capture_key):
                    def hook(module, inputs, output):
                        self.activations[capture_key] = {
                            "input": inputs[0].detach().cpu(),
                            "output": output.detach().cpu()
                        }
                    return hook
                
                handle = module.register_forward_hook(make_hook(key))
                self.handles.append(handle)
    
    def clear(self):
        """Clear collected activations (keeps hooks registered)."""
        self.activations = {}
    
    def remove_hooks(self):
        """Remove all hooks and clear activations."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
    
    def get_activations(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get collected activations.
        
        Returns:
            Dictionary mapping layer_module keys to {"input": tensor, "output": tensor}
        """
        return self.activations


# =============================================================================
# Utility Functions
# =============================================================================

def compute_sparsity_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about tensor sparsity.
    
    Useful for debugging and understanding activation distributions.
    
    Args:
        x: Input tensor
    
    Returns:
        Dictionary with sparsity statistics
    """
    x_flat = x.view(-1)
    
    return {
        "num_elements": x_flat.numel(),
        "num_zeros": (x_flat == 0).sum().item(),
        "sparsity": (x_flat == 0).float().mean().item(),
        "mean_magnitude": x_flat.abs().mean().item(),
        "max_magnitude": x_flat.abs().max().item(),
        "std": x_flat.std().item()
    }
