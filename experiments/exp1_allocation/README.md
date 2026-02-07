# Experiment 1: Allocation Sweep

This experiment studies where SPON biases should be allocated for best PPL-vs-parameter tradeoff.

## Entry points

- `run_allocation_sweep.py`: full config/sparsity sweep with structured result logging.
- `generate_data_driven_configs.py`: generates layer-selection overrides from importance scores.
- `run_loss_ablation.py`: small ablation for SPON training loss behavior.
- `layer_importance.py`: gradient/ILA/Shapley/cornerstone layer-importance methods.
- `layer_sensitivity.py`: per-layer sensitivity and cornerstone analysis helpers.
- `visualizations.py`: Pareto/sensitivity/loss plotting utilities.

## Typical command

```bash
python experiments/exp1_allocation/run_allocation_sweep.py \
  --model meta-llama/Llama-3.2-1B \
  --model_provider auto \
  --configs BASELINE-TEAL UNIF-ALL TOP-50 \
  --sparsity 0.5 \
  --epochs 1 \
  --experiment_name quick_test \
  --output_dir results
```

For data-driven configs, generate overrides first and pass `--data_driven_overrides`.

Provider flags are shared across entrypoints:
- `--model_provider {auto,huggingface,ollama}`
- `--ollama_model llama3.2:1b`
- `--pull_ollama`
