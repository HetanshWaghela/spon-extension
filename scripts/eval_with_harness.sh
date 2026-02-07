#!/bin/bash
# Evaluate a model using lm-eval-harness

set -e

MODEL_PATH=${1:-"meta-llama/Llama-3.2-1B"}
OUTPUT_DIR=${2:-"results/harness_eval"}
TASKS=${3:-"mmlu,arc_easy,hellaswag"}

echo "Evaluating $MODEL_PATH on $TASKS..."

mkdir -p "$OUTPUT_DIR"

lm_eval --model hf \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
    --tasks "$TASKS" \
    --device cuda:0 \
    --batch_size auto \
    --output_path "$OUTPUT_DIR" \
    --log_samples

echo "Results saved to $OUTPUT_DIR"
