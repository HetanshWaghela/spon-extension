"""Model provider utilities for Hugging Face + Ollama with safe fallback semantics."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ProviderResolution:
    requested_provider: str
    effective_provider: str
    hf_model_name: str
    ollama_model_name: str
    ollama_available: bool
    used_fallback: bool = False
    reason: str = ""


def add_model_provider_args(parser: argparse.ArgumentParser, default_provider: str = "auto") -> None:
    """Add shared model-provider CLI args."""
    parser.add_argument(
        "--model_provider",
        type=str,
        choices=["auto", "huggingface", "ollama"],
        default=default_provider,
        help="Model source provider. 'auto' prefers Ollama if available, then falls back to Hugging Face.",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=None,
        help="Ollama model tag (e.g. llama3.2:1b). If omitted, inferred from --model when possible.",
    )
    parser.add_argument(
        "--pull_ollama",
        action="store_true",
        help="If Ollama is selected/available, run `ollama pull` for the chosen model.",
    )


def is_ollama_available() -> bool:
    """Check whether the ollama CLI is available and responsive."""
    if shutil.which("ollama") is None:
        return False
    try:
        subprocess.run(
            ["ollama", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def infer_ollama_model_name(model_name: str) -> str:
    """Infer an Ollama model tag from a Hugging Face style model id."""
    lowered = model_name.strip().lower()
    mapping = {
        "meta-llama/llama-3.2-1b": "llama3.2:1b",
        "meta-llama/llama-3.2-3b": "llama3.2:3b",
        "meta-llama/llama-3.1-8b": "llama3.1:8b",
    }
    if lowered in mapping:
        return mapping[lowered]
    if ":" in model_name and "/" not in model_name:
        return model_name
    return "llama3.2:1b"


def maybe_pull_ollama_model(model_name: str) -> bool:
    """Try to pull an Ollama model. Returns True on success."""
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except Exception as exc:
        logger.warning("Failed to pull Ollama model '%s': %s", model_name, exc)
        return False


def resolve_model_provider(
    model_name: str,
    provider: str = "auto",
    ollama_model: Optional[str] = None,
    pull_ollama: bool = False,
    require_transformers_model: bool = True,
) -> ProviderResolution:
    """
    Resolve model provider with safe fallback.

    For SPON workflows, `require_transformers_model=True` is required because hooks,
    layer access, and gradient-based training need a local torch/transformers model.
    """
    requested = provider
    ollama_name = ollama_model or infer_ollama_model_name(model_name)
    ollama_available = is_ollama_available()

    selected = provider
    if provider == "auto":
        selected = "ollama" if ollama_available else "huggingface"

    used_fallback = False
    reason = ""

    if selected == "ollama":
        if not ollama_available:
            used_fallback = True
            reason = "Ollama CLI not available"
            selected = "huggingface"
        else:
            if pull_ollama:
                maybe_pull_ollama_model(ollama_name)
            if require_transformers_model:
                used_fallback = True
                reason = (
                    "SPON training/analysis requires direct transformers model internals; "
                    "falling back to Hugging Face runtime."
                )
                selected = "huggingface"

    return ProviderResolution(
        requested_provider=requested,
        effective_provider=selected,
        hf_model_name=model_name,
        ollama_model_name=ollama_name,
        ollama_available=ollama_available,
        used_fallback=used_fallback,
        reason=reason,
    )


def log_provider_resolution(resolution: ProviderResolution) -> None:
    """Emit standardized provider resolution logs."""
    logger.info(
        "Model provider requested=%s effective=%s ollama_available=%s ollama_model=%s",
        resolution.requested_provider,
        resolution.effective_provider,
        resolution.ollama_available,
        resolution.ollama_model_name,
    )
    if resolution.used_fallback and resolution.reason:
        logger.warning("Model provider fallback: %s", resolution.reason)


def load_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, device: torch.device):
    """Load a Hugging Face causal LM with dtype chosen by device."""
    from transformers import AutoModelForCausalLM
    from . import get_default_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_default_dtype(device),
        trust_remote_code=True,
    )
    model.to(device)
    return model
