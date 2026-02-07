# SPON Extensions

> **Extending "Resting Neurons, Active Insights: Improving Input Sparsification for Large Language Models"**

[![Paper](https://img.shields.io/badge/arXiv-2512.12744-b31b1b.svg)](https://arxiv.org/abs/2512.12744)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains experiments extending the SPON (Spontaneous Neuron Activation) paper for efficient sparse LLM inference.

---

## 📚 Documentation

The primary documentation for this repository is contained within this `README.md` and the sub-directory READMEs.

---

## 🎯 Research Questions

### Experiment 1: Optimal SPON Allocation
> *Can we achieve equivalent PPL improvement with fewer SPON parameters?*

The original SPON paper adds biases to all layers. We investigate:
- **Layer-wise allocation**: Are early layers more important than later layers?
- **Module-wise allocation**: Is down_proj sufficient, or do we need attention projections?
- **Pareto efficiency**: What's the optimal parameter-performance trade-off?

### Experiment 2: Mechanistic Interpretability
> *What do SPON biases actually encode?*

We analyze:
- Alignment with principal components of hidden states
- Category-specific effects (math vs. commonsense vs. coding)
- Sparse autoencoder feature decomposition

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd spon-extensions

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model provider options (Ollama + Hugging Face fallback)

- All experiment entrypoints support: `--model_provider {auto,huggingface,ollama}`.
- `auto` prefers Ollama when available, then falls back to Hugging Face.
- Current SPON training/analysis requires direct `transformers` model internals, so when `ollama` is requested it still falls back to Hugging Face runtime and logs the fallback reason.
- Optional Ollama pull: `--pull_ollama --ollama_model llama3.2:1b`.
- For gated models (for example Meta Llama on Hugging Face), you still need:

```bash
huggingface-cli login
```

### 2. Verify Setup

```bash
# Run setup verification (catches issues early)
python test_setup.py

# Skip model loading if no GPU
python test_setup.py --skip-model
```

Optional pre-download:

```bash
# Download both Hugging Face + Ollama model variants
bash scripts/download_models.sh both
```

### 3. Run Experiments

```bash
# Quick test run (1 epoch, 2 configs)
python experiments/exp1_allocation/run_allocation_sweep.py \
    --model meta-llama/Llama-3.2-1B \
    --model_provider auto \
    --configs BASELINE-TEAL UNIF-ALL TOP-50 \
    --sparsity 0.5 \
    --epochs 1 \
    --experiment_name test \
    --output_dir results

# Full experiment
python experiments/exp1_allocation/run_allocation_sweep.py \
    --model meta-llama/Llama-3.2-1B \
    --configs BASELINE-TEAL UNIF-ALL TOP-25 TOP-50 TOP-75 BOTTOM-50 \
    --sparsity 0.5 0.6 \
    --epochs 10 \
    --dense_baseline \
    --experiment_name full_sweep \
    --output_dir results

# Experiment 2 (after you have a trained SPON checkpoint)
python experiments/exp2_interpretability/run_interpretability.py \
    --model meta-llama/Llama-3.2-1B \
    --model_provider auto \
    --spon_checkpoint results/full_sweep/runs/<run_id>/checkpoints/spon_TOP-50_s0.50.pt \
    --sparsity 0.5 \
    --output_dir results/interpretability
```

For data-driven configs (`CRITICAL-ONLY`, `SHAPLEY-GUIDED`, `CORNERSTONE`), first generate overrides and pass them to the sweep:

```bash
python -m experiments.exp1_allocation.generate_data_driven_configs \
    --model meta-llama/Llama-3.2-1B \
    --method shapley \
    --output configs/data_driven_allocation_overrides.yaml

python experiments/exp1_allocation/run_allocation_sweep.py \
    --configs CRITICAL-ONLY SHAPLEY-GUIDED CORNERSTONE \
    --data_driven_overrides configs/data_driven_allocation_overrides.yaml \
    --output_dir results
```

### 4. View Results

Results are saved in a structured format:

```
results/allocation_sweep/
├── metadata.json                    # Experiment metadata
├── runs/
│   └── run_20240101_120000/
│       ├── summary.json             # Complete run summary
│       ├── results.json             # Per-configuration results
│       ├── metrics.csv              # Training metrics over time
│       ├── config.json              # Run configuration
│       ├── checkpoints/             # Saved SPON biases (.pt files)
│       ├── figures/                 # Generated plots
│       └── logs.json                # Execution logs
├── aggregated/
│   ├── summary.csv                  # All results in tabular form
│   ├── pareto.json                  # Pareto-optimal configurations
│   └── latex_tables/                # Publication-ready LaTeX tables
└── analysis/
    ├── statistics.json              # Aggregated statistics
    └── comparisons.json             # Statistical comparisons
```

---

## 📁 Project Structure

```
spon-extensions/
├── configs/
│   ├── allocation_configs.yaml      # SPON allocation strategies
│   └── training_configs.yaml        # Training hyperparameters
│
├── src/
│   ├── __init__.py                  # Package exports
│   ├── allocation.py                # Configuration management
│   ├── sparse_forward.py            # TEAL-style sparsification
│   ├── spon_trainer.py              # KL-divergence training
│   ├── evaluation.py                # Metrics computation
│   └── result_manager.py            # Structured result saving
│
├── notebooks/
│   ├── 01_quickstart_demo.ipynb     # Interactive SPON tutorial
│   ├── 02_visualize_results.ipynb   # Pareto frontiers & analysis
│   └── 03_layer_analysis.ipynb      # Layer importance deep-dive
│
├── analysis/
│   ├── layer_importance.py          # Backward-compatible import shim
│   └── visualizations.py            # Backward-compatible import shim
│
├── experiments/
│   ├── exp1_allocation/
│   │   ├── run_allocation_sweep.py
│   │   ├── generate_data_driven_configs.py
│   │   ├── run_loss_ablation.py
│   │   ├── layer_importance.py
│   │   ├── layer_sensitivity.py
│   │   └── visualizations.py
│   ├── exp2_interpretability/
│   │   ├── run_interpretability.py
│   │   └── visualizations.py
│   ├── run_lm_eval_harness.py       # Shared downstream eval wrapper
│   ├── run_allocation_sweep.py      # Compatibility wrapper
│   └── run_interpretability.py      # Compatibility wrapper
│
├── scripts/
│   ├── download_models.sh           # Model download helper
│   └── eval_with_harness.sh         # lm-eval-harness wrapper
│
├── test_setup.py                    # Setup verification script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Configuration Options

### SPON Allocation Configurations

| Config Name | Description | Relative Params |
|-------------|-------------|-----------------|
| `BASELINE-TEAL` | No SPON, sparse only (control) | 0.0× |
| `UNIF-ALL` | SPON on all layers (paper baseline) | 1.0× |
| `TOP-25` | SPON on first 25% of layers | 0.25× |
| `TOP-50` | SPON on first 50% of layers | 0.50× |
| `TOP-75` | SPON on first 75% of layers | 0.75× |
| `BOTTOM-50` | SPON on last 50% of layers (control) | 0.50× |
| `ATTN-ONLY` | SPON on attention O-projection only | 0.50× |
| `HYBRID-OD` | SPON on O-proj + down-proj | 1.50× |
| `SHAPLEY-GUIDED` | Layers selected by Shapley importance | Variable |
| `CORNERSTONE` | Only cornerstone layers | ~0.10× |

### Model Support

| Model | Layers | Status |
|-------|--------|--------|
| LLaMA-3.2-1B | 16 | ✅ Primary |
| LLaMA-3.2-3B | 28 | ✅ Validated |
| LLaMA-3.1-8B | 32 | ✅ Validated |

---

## 📊 Expected Results

Based on the original paper's findings:

| Configuration | 50% Sparsity PPL | vs TEAL |
|---------------|------------------|---------|
| Dense (reference) | ~10-12 | - |
| TEAL only | ~18-22 | baseline |
| UNIF-ALL | ~15-16 | -15-20% |
| TOP-50 | ~15-17 | -10-18% |
| TOP-25 | ~16-18 | -5-12% |
| BOTTOM-50 | ~17-19 | -3-8% |

**Key hypothesis**: TOP-50 should achieve ~95% of UNIF-ALL's improvement with 50% of parameters.

---

## 🔬 API Reference

### Core Classes

```python
from src import (
    SPONTrainer,      # Train SPON biases
    SPONConfig,       # Configuration for allocation
    TrainingArgs,     # Training hyperparameters
    ExperimentManager # Manage experimental runs
)

# Example: Train SPON
config = SPONConfig(
    name="my_config",
    layer_mask=[0, 1, 2, 3],  # First 4 layers
    modules=["down_proj"]
)

trainer = SPONTrainer(model, config, sparsity=0.5, args)
trainer.train(dataloader)
biases = trainer.get_spon_biases()
```

### Key Functions

```python
from src import (
    magnitude_sparsify,    # Core sparsification
    compute_perplexity,    # Evaluate model
    compute_pareto_frontier # Find optimal configs
)

# Sparsify activations
x_sparse = magnitude_sparsify(x, sparsity=0.5)

# Evaluate
result = compute_perplexity(model, dataloader, use_sparse=True)
print(f"PPL: {result.perplexity:.2f}")
```

---

## 📝 Citation

If you use this code, please cite the original SPON paper:

```bibtex
@article{xu2024resting,
  title={Resting Neurons, Active Insights: Improving Input Sparsification for Large Language Models},
  author={Xu, Haotian and Gao, Tian and Weng, Tsui-Wei and Ma, Tengfei},
  journal={arXiv preprint arXiv:2512.12744},
  year={2024}
}
```

---

## 🔗 References

- **SPON Paper**: [arXiv:2512.12744](https://arxiv.org/abs/2512.12744)
- **TEAL**: Training-free Activation Sparsity (ICLR 2025 Spotlight)
- **Shapley Layer Importance**: [arXiv:2409.14381](https://arxiv.org/abs/2409.14381)
- **ILA**: Identifying Important Layers for Alignment [arXiv:2410.17875](https://arxiv.org/abs/2410.17875)
- **Prof. Weng's Lab**: [UCSD HDSI](https://datascience.ucsd.edu/people/tsui-wei-weng/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
