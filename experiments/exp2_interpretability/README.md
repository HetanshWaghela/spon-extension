# Experiment 2: Mechanistic Interpretability

This experiment analyzes what trained SPON biases encode in model representations.

## Entry points

- `run_interpretability.py`: computes PCA alignment, category-wise shifts, CKA, and bias stats.
- `visualizations.py`: interpretability-focused plotting utilities.

## Typical command

```bash
python experiments/exp2_interpretability/run_interpretability.py \
  --model meta-llama/Llama-3.2-1B \
  --model_provider auto \
  --spon_checkpoint path/to/checkpoint.pt \
  --sparsity 0.5
```

Provider flags:
- `--model_provider {auto,huggingface,ollama}`
- `--ollama_model llama3.2:1b`
- `--pull_ollama`
