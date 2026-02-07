#!/bin/bash
# Download models for SPON experiments

set -e

PROVIDER="${1:-both}"  # huggingface | ollama | both

echo "Downloading models for SPON experiments..."

HF_MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

OLLAMA_MODELS=(
    "llama3.2:1b"
    "llama3.2:3b"
)

if [[ "$PROVIDER" == "huggingface" || "$PROVIDER" == "both" ]]; then
    # Check if huggingface-cli is available
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface_hub..."
        pip install huggingface_hub
    fi

    for MODEL in "${HF_MODELS[@]}"; do
        echo "Downloading (Hugging Face) $MODEL..."
        huggingface-cli download "$MODEL" --exclude "*.gguf" "original/*" || {
            echo "Warning: Could not download $MODEL (may require authentication)"
            echo "Run: huggingface-cli login"
        }
    done
fi

if [[ "$PROVIDER" == "ollama" || "$PROVIDER" == "both" ]]; then
    if ! command -v ollama &> /dev/null; then
        echo "Warning: 'ollama' is not installed; skipping Ollama pulls."
    else
        for MODEL in "${OLLAMA_MODELS[@]}"; do
            echo "Pulling (Ollama) $MODEL..."
            ollama pull "$MODEL" || {
                echo "Warning: Could not pull $MODEL from Ollama."
            }
        done
    fi
fi

echo "Done!"
