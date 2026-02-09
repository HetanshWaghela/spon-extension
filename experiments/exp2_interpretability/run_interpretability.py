#!/usr/bin/env python3
"""
Experiment 2: Mechanistic Interpretability Analysis

Analyzes what SPON biases encode and how they affect representations.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model_provider import (
    add_model_provider_args,
    load_causal_lm,
    load_tokenizer,
    log_provider_resolution,
    resolve_model_provider,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Prompt categories for interpretability analysis
PROMPT_CATEGORIES = {
    "math": [
        "Calculate the derivative of x^3 + 2x^2 - 5x + 1",
        "Solve the equation 3x + 7 = 22",
        "What is the integral of sin(x)?",
        "Find the roots of x^2 - 5x + 6 = 0",
        "Compute 247 * 38",
    ],
    "commonsense": [
        "Why do birds fly south for the winter?",
        "What happens when you mix oil and water?",
        "Why is the sky blue?",
        "How do plants get their food?",
        "What causes thunder?",
    ],
    "coding": [
        "Write a Python function to reverse a string",
        "Implement binary search in JavaScript",
        "How do you handle exceptions in Python?",
        "Explain the difference between a list and a tuple",
        "What is recursion?",
    ],
    "safety": [
        "How can I protect my computer from malware?",
        "What are best practices for password security?",
        "How do I spot a phishing email?",
        "What is two-factor authentication?",
        "How can I safely share files online?",
    ],
    "factual": [
        "Who was the first president of the United States?",
        "What is the capital of France?",
        "When did World War II end?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
    ]
}


def infer_modules_and_layers_from_biases(spon_biases: dict) -> tuple[list[str], list[int]]:
    """Infer target modules and layers from SPON bias keys."""
    modules = set()
    layers = set()

    for key in spon_biases:
        parts = key.split("_")
        if len(parts) < 3 or parts[0] != "layer":
            continue
        try:
            layers.add(int(parts[1]))
        except ValueError:
            continue
        modules.add("_".join(parts[2:]))

    return sorted(modules), sorted(layers)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SPON interpretability analysis")
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--spon_checkpoint", 
        type=str, 
        required=True,
        help="Path to trained SPON checkpoint (primary)"
    )
    parser.add_argument(
        "--spon_checkpoint_b",
        type=str,
        default=None,
        help="Path to a second SPON checkpoint for cross-config comparison"
    )
    parser.add_argument(
        "--sparsity", 
        type=float, 
        default=0.5,
        help="Sparsity level"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/interpretability",
        help="Output directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device"
    )
    add_model_provider_args(parser)
    return parser.parse_args()


def collect_hidden_states(
    model, 
    tokenizer, 
    prompts: list,
    device: torch.device,
    module_names: list[str],
    layer_indices: list[int] | None = None,
) -> dict:
    """Collect hidden states for a set of prompts."""
    from src.sparse_forward import ActivationCollector
    
    collector = ActivationCollector()
    collector.register(model, layer_indices=layer_indices, module_names=module_names)
    
    all_hidden_states = {}
    
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            _ = model(**inputs)
            
            for key, acts in collector.get_activations().items():
                if key not in all_hidden_states:
                    all_hidden_states[key] = []
                all_hidden_states[key].append(acts["output"].mean(dim=1))  # Average over sequence
            
            collector.clear()
    
    collector.remove_hooks()
    
    # Stack tensors and validate results
    stacked_states = {}
    for key in all_hidden_states:
        if all_hidden_states[key]:  # Non-empty list
            stacked = torch.cat(all_hidden_states[key], dim=0)
            if not isinstance(stacked, torch.Tensor):
                logger.warning(f"Unexpected type for hidden states key '{key}': {type(stacked)}")
                continue
            stacked_states[key] = stacked
        else:
            logger.warning(f"No hidden states collected for key '{key}'")
    
    return stacked_states


def compute_pca_alignment(
    spon_biases: dict,
    hidden_states: dict,
    n_components: int = 10
) -> dict:
    """
    Compute alignment between SPON biases and principal components of hidden states.
    
    Returns cosine similarity of each layer's SPON bias to top-k PCs.
    """
    from sklearn.decomposition import PCA
    
    alignments = {}
    
    for layer_key, bias in spon_biases.items():
        if layer_key not in hidden_states:
            logger.debug(f"Skipping PCA for '{layer_key}': not in hidden_states (available: {list(hidden_states.keys())})")
            continue
        
        states_val = hidden_states[layer_key]
        
        # Defensive check: ensure we have a tensor, not a dict or other type
        if not isinstance(states_val, torch.Tensor):
            logger.warning(
                f"Skipping PCA for '{layer_key}': expected Tensor but got {type(states_val).__name__}. "
                f"This may indicate a mismatch between collector output and expected format."
            )
            continue
        
        states = states_val.cpu().float().numpy()
        bias_np = bias.cpu().float().numpy()
        
        # Ensure sufficient samples for PCA
        if states.shape[0] < 2:
            logger.warning(f"Skipping PCA for '{layer_key}': only {states.shape[0]} samples (need >= 2)")
            continue
        
        # Fit PCA
        n_comps = min(n_components, states.shape[0] - 1, states.shape[1])
        if n_comps < 1:
            logger.warning(f"Skipping PCA for '{layer_key}': computed n_components={n_comps} < 1")
            continue
            
        pca = PCA(n_components=n_comps)
        pca.fit(states)
        
        # Compute cosine similarity to each PC
        pc_sims = []
        for i, pc in enumerate(pca.components_):
            cos_sim = np.dot(bias_np, pc) / (np.linalg.norm(bias_np) * np.linalg.norm(pc) + 1e-8)
            pc_sims.append(float(cos_sim))
        
        alignments[layer_key] = {
            "pc_similarities": pc_sims,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "max_alignment": max(abs(s) for s in pc_sims),
            "best_pc": int(np.argmax(np.abs(pc_sims)))
        }
    
    return alignments


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Centered Kernel Alignment between two sets of representations.
    
    CKA is more robust than cosine similarity for comparing representations.
    """
    def center_gram(K):
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H
    
    # Linear kernel
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    # Center
    K_X = center_gram(K_X)
    K_Y = center_gram(K_Y)
    
    # CKA
    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)
    
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-8)
    return float(cka)


def analyze_category_shifts(
    model,
    tokenizer,
    spon_biases: dict,
    sparsity: float,
    device: torch.device,
    target_modules: list[str],
    layer_indices: list[int] | None = None,
) -> dict:
    """
    Analyze how SPON affects representations differently across prompt categories.
    
    Uses hook-based sparsification (non-destructive) to compare dense vs sparse+SPON.
    """
    from src.sparse_forward import register_sparsification_hooks, remove_hooks
    
    results = {}
    
    for category, prompts in PROMPT_CATEGORIES.items():
        logger.info(f"Analyzing category: {category}")
        
        # Collect dense hidden states (no hooks)
        dense_states = collect_hidden_states(
            model,
            tokenizer,
            prompts,
            device,
            module_names=target_modules,
            layer_indices=layer_indices,
        )
        
        # Collect sparse hidden states (with SPON) using hooks (NON-DESTRUCTIVE)
        hooks = register_sparsification_hooks(
            model, sparsity, target_modules, spon_biases
        )
        try:
            sparse_states = collect_hidden_states(
                model,
                tokenizer,
                prompts,
                device,
                module_names=target_modules,
                layer_indices=layer_indices,
            )
        finally:
            remove_hooks(hooks)
        
        # Compute L2 shifts per layer
        l2_shifts = {}
        cka_scores = {}
        
        for key in dense_states:
            if key in sparse_states:
                dense_val = dense_states[key]
                sparse_val = sparse_states[key]
                
                # Defensive check: ensure tensors
                if not isinstance(dense_val, torch.Tensor) or not isinstance(sparse_val, torch.Tensor):
                    logger.warning(f"Skipping L2/CKA for '{key}': expected Tensors but got {type(dense_val).__name__}, {type(sparse_val).__name__}")
                    continue
                
                dense = dense_val.cpu().float().numpy()
                sparse = sparse_val.cpu().float().numpy()
                
                # L2 shift
                l2 = np.linalg.norm(dense - sparse, axis=-1).mean()
                l2_shifts[key] = float(l2)
                
                # CKA
                cka = compute_cka(dense, sparse)
                cka_scores[key] = cka
        
        avg_l2 = float(np.mean(list(l2_shifts.values()))) if l2_shifts else None
        avg_cka = float(np.mean(list(cka_scores.values()))) if cka_scores else None

        results[category] = {
            "l2_shifts": l2_shifts,
            "cka_scores": cka_scores,
            "avg_l2": avg_l2,
            "avg_cka": avg_cka,
        }
    
    return results


def main():
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
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(provider_resolution.hf_model_name)
    
    device = torch.device(args.device)
    
    model = load_causal_lm(provider_resolution.hf_model_name, device)
    
    # Load SPON checkpoint
    logger.info(f"Loading SPON checkpoint: {args.spon_checkpoint}")
    checkpoint = torch.load(args.spon_checkpoint, map_location=device, weights_only=True)
    spon_biases = {k: v.to(device) for k, v in checkpoint["biases"].items()}
    modules_from_bias, layers_from_bias = infer_modules_and_layers_from_biases(spon_biases)

    extra = checkpoint.get("extra", {})
    target_modules = extra.get("modules") if isinstance(extra.get("modules"), list) else modules_from_bias
    layer_indices = extra.get("layer_mask") if isinstance(extra.get("layer_mask"), list) else layers_from_bias
    if not target_modules:
        target_modules = ["down_proj"]
        logger.warning(
            "Could not infer target modules from checkpoint; defaulting to ['down_proj']"
        )
    if not layer_indices:
        layer_indices = None

    logger.info(f"Loaded {len(spon_biases)} SPON biases")
    logger.info(f"Target modules inferred: {target_modules}")
    logger.info(f"Layer indices inferred: {layer_indices}")
    
    # Analysis 1: PCA alignment
    logger.info("\n=== Analysis 1: PCA Alignment ===")
    all_prompts = sum(PROMPT_CATEGORIES.values(), [])
    hidden_states = collect_hidden_states(
        model,
        tokenizer,
        all_prompts,
        device,
        module_names=target_modules,
        layer_indices=layer_indices,
    )
    
    pca_alignments = compute_pca_alignment(spon_biases, hidden_states)
    
    logger.info("SPON-PC alignment summary:")
    for layer, alignment in pca_alignments.items():
        logger.info(f"  {layer}: max_align={alignment['max_alignment']:.3f}, best_pc={alignment['best_pc']}")
    
    # Analysis 2: Category-specific shifts
    logger.info("\n=== Analysis 2: Category-specific Shifts ===")
    category_results = analyze_category_shifts(
        model,
        tokenizer,
        spon_biases,
        args.sparsity,
        device,
        target_modules=target_modules,
        layer_indices=layer_indices,
    )
    
    logger.info("L2 shift by category:")
    for category, result in category_results.items():
        avg_l2 = result["avg_l2"]
        avg_cka = result["avg_cka"]
        l2_text = f"{avg_l2:.4f}" if avg_l2 is not None else "n/a"
        cka_text = f"{avg_cka:.4f}" if avg_cka is not None else "n/a"
        logger.info(f"  {category}: avg_l2={l2_text}, avg_cka={cka_text}")
    
    # Analysis 3: SPON bias statistics
    logger.info("\n=== Analysis 3: SPON Bias Statistics ===")
    bias_stats = {}
    for key, bias in spon_biases.items():
        bias_np = bias.cpu().float().numpy()
        bias_stats[key] = {
            "mean": float(np.mean(bias_np)),
            "std": float(np.std(bias_np)),
            "l2_norm": float(np.linalg.norm(bias_np)),
            "max_abs": float(np.max(np.abs(bias_np))),
            "sparsity": float(np.mean(np.abs(bias_np) < 1e-6)),
            "pos_frac": float(np.mean(bias_np > 0)),
            "kurtosis": float(
                np.mean((bias_np - np.mean(bias_np))**4) / (np.std(bias_np)**4 + 1e-12) - 3
            ),
        }
    
    logger.info("Bias statistics:")
    for key, stats in bias_stats.items():
        logger.info(f"  {key}: norm={stats['l2_norm']:.4f}, std={stats['std']:.4f}")
    
    # Analysis 4: Hidden-state shift quantification (replicates paper Fig 3)
    # Measures how much TEAL-only shifts representations vs Dense,
    # and how SPON recovers from that shift.
    logger.info("\n=== Analysis 4: Hidden-State Shift Quantification ===")
    from src.sparse_forward import register_sparsification_hooks, remove_hooks as _remove_hooks

    shift_prompts = sum(list(PROMPT_CATEGORIES.values())[:3], [])  # use first 3 categories
    shift_prompts = shift_prompts[:15]  # cap at 15 prompts for speed

    # 4a. Dense hidden states
    dense_hs = collect_hidden_states(
        model, tokenizer, shift_prompts, device,
        module_names=target_modules, layer_indices=layer_indices,
    )

    # 4b. TEAL-only hidden states (sparse, NO SPON)
    teal_hooks = register_sparsification_hooks(
        model, args.sparsity, target_modules, spon_biases=None
    )
    teal_hs = collect_hidden_states(
        model, tokenizer, shift_prompts, device,
        module_names=target_modules, layer_indices=layer_indices,
    )
    _remove_hooks(teal_hooks)

    # 4c. SPON hidden states (sparse + SPON biases)
    spon_hooks = register_sparsification_hooks(
        model, args.sparsity, target_modules, spon_biases=spon_biases
    )
    spon_hs = collect_hidden_states(
        model, tokenizer, shift_prompts, device,
        module_names=target_modules, layer_indices=layer_indices,
    )
    _remove_hooks(spon_hooks)

    shift_results = {}
    for key in sorted(dense_hs.keys()):
        if key not in teal_hs or key not in spon_hs:
            continue
        d, t, s = dense_hs[key], teal_hs[key], spon_hs[key]
        if not all(isinstance(x, torch.Tensor) for x in (d, t, s)):
            continue
        teal_shift = torch.norm(d - t, p=2, dim=-1).mean().item()
        spon_shift = torch.norm(d - s, p=2, dim=-1).mean().item()
        recovery = ((teal_shift - spon_shift) / (teal_shift + 1e-8)) * 100
        shift_results[key] = {
            "teal_shift_l2": round(teal_shift, 4),
            "spon_shift_l2": round(spon_shift, 4),
            "recovery_pct": round(recovery, 2),
        }
        logger.info(
            f"  {key}: TEAL shift={teal_shift:.4f}, SPON shift={spon_shift:.4f}, "
            f"recovery={recovery:.1f}%"
        )

    # Aggregate shift recovery
    if shift_results:
        avg_teal = np.mean([v["teal_shift_l2"] for v in shift_results.values()])
        avg_spon = np.mean([v["spon_shift_l2"] for v in shift_results.values()])
        avg_recovery = np.mean([v["recovery_pct"] for v in shift_results.values()])
        logger.info(
            f"  ** Avg across layers: TEAL={avg_teal:.4f}, SPON={avg_spon:.4f}, "
            f"recovery={avg_recovery:.1f}%"
        )
        shift_results["_aggregate"] = {
            "avg_teal_shift": round(float(avg_teal), 4),
            "avg_spon_shift": round(float(avg_spon), 4),
            "avg_recovery_pct": round(float(avg_recovery), 2),
        }

    # Analysis 5: Layer-wise bias norm ranking (which layers need SPON most?)
    logger.info("\n=== Analysis 5: Layer-wise Bias Norm Ranking ===")
    layer_norms = []
    for key, bias in spon_biases.items():
        layer_norms.append((key, float(torch.norm(bias, p=2).item())))
    layer_norms.sort(key=lambda x: x[1], reverse=True)
    logger.info("Layers ranked by SPON bias L2 norm (most → least correction):")
    for rank, (key, norm) in enumerate(layer_norms, 1):
        logger.info(f"  #{rank}: {key}  norm={norm:.4f}")
    bias_norm_ranking = [{"layer": k, "l2_norm": n} for k, n in layer_norms]

    # Analysis 6: Cross-config comparison (if second checkpoint provided)
    cross_config_results = {}
    if args.spon_checkpoint_b:
        logger.info("\n=== Analysis 6: Cross-Config SPON Bias Comparison ===")
        ckpt_b = torch.load(args.spon_checkpoint_b, map_location=device, weights_only=True)
        biases_b = {k: v.to(device) for k, v in ckpt_b["biases"].items()}
        extra_b = ckpt_b.get("extra", {})
        config_b_name = extra_b.get("config_name", Path(args.spon_checkpoint_b).stem)

        shared_keys = sorted(set(spon_biases.keys()) & set(biases_b.keys()))
        for key in shared_keys:
            a_flat = spon_biases[key].float().flatten()
            b_flat = biases_b[key].float().flatten()
            cos = float(torch.nn.functional.cosine_similarity(
                a_flat.unsqueeze(0), b_flat.unsqueeze(0)
            ).item())
            l2_diff = float(torch.norm(a_flat - b_flat, p=2).item())
            cross_config_results[key] = {
                "cosine_similarity": round(cos, 4),
                "l2_difference": round(l2_diff, 4),
            }
            logger.info(f"  {key}: cos_sim={cos:.4f}, l2_diff={l2_diff:.4f}")

        if cross_config_results:
            avg_cos = np.mean([v["cosine_similarity"] for v in cross_config_results.values()])
            logger.info(f"  ** Avg cosine similarity across shared layers: {avg_cos:.4f}")
            cross_config_results["_aggregate"] = {
                "avg_cosine_similarity": round(float(avg_cos), 4),
                "config_a": Path(args.spon_checkpoint).stem,
                "config_b": config_b_name,
            }

    # Save results
    results = {
        "args": vars(args),
        "pca_alignments": pca_alignments,
        "category_analysis": category_results,
        "bias_statistics": bias_stats,
        "hidden_state_shifts": shift_results,
        "bias_norm_ranking": bias_norm_ranking,
        "cross_config_comparison": cross_config_results,
        "timestamp": timestamp
    }
    
    results_file = output_dir / f"interpretability_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
