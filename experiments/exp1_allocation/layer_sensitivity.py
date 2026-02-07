"""
Layer sensitivity analysis utilities for Experiment 1 (allocation sweep).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import compute_perplexity

logger = logging.getLogger(__name__)


class LayerSensitivityAnalyzer:
    """Analyze per-layer sensitivity to SPON for allocation decisions."""

    def __init__(self, model: torch.nn.Module, device: torch.device | None = None):
        if device is None:
            from src import get_device

            device = get_device()

        self.model = model
        self.device = device
        self.num_layers = len(model.model.layers)

    def compute_layer_sensitivity(
        self,
        dataloader: DataLoader,
        sparsity: float = 0.5,
    ) -> np.ndarray:
        """
        Compute sensitivity of each layer to SPON.

        Sensitivity = PPL improvement when adding SPON to that layer only.
        """
        from src.allocation import SPONConfig
        from src.spon_trainer import SPONTrainer, TrainingArgs

        baseline_ppl = compute_perplexity(
            self.model,
            dataloader,
            self.device,
            use_sparse=True,
            sparsity=sparsity,
            spon_biases=None,
        ).perplexity

        sensitivities = np.zeros(self.num_layers)

        for layer_idx in tqdm(range(self.num_layers), desc="Computing layer sensitivity"):
            config = SPONConfig(
                name=f"layer_{layer_idx}_only",
                layer_mask=[layer_idx],
                modules=["down_proj"],
            )

            args = TrainingArgs(epochs=3, device=str(self.device))
            trainer = SPONTrainer(self.model, config, sparsity, args)
            trainer.train(dataloader)

            spon_biases = trainer.get_spon_biases()
            layer_ppl = compute_perplexity(
                self.model,
                dataloader,
                self.device,
                use_sparse=True,
                sparsity=sparsity,
                spon_biases=spon_biases,
            ).perplexity

            sensitivities[layer_idx] = baseline_ppl - layer_ppl

        return sensitivities

    def compute_cornerstone_layers(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
    ) -> List[int]:
        """
        Identify cornerstone layers (removal causes >threshold PPL increase).
        """
        baseline_ppl = compute_perplexity(self.model, dataloader, self.device).perplexity

        cornerstone_layers = []

        for layer_idx in tqdm(range(self.num_layers), desc="Finding cornerstone layers"):
            layer = self.model.model.layers[layer_idx]
            orig_forward = layer.forward

            def make_skip_forward(orig_fn):
                def skip_forward(hidden_states, *args, **kwargs):
                    return (hidden_states,) + (None,) * 2

                return skip_forward

            layer.forward = make_skip_forward(orig_forward)

            try:
                removed_ppl = compute_perplexity(self.model, dataloader, self.device).perplexity
            except Exception:
                removed_ppl = float("inf")

            layer.forward = orig_forward

            ppl_increase = (removed_ppl - baseline_ppl) / baseline_ppl
            if ppl_increase > threshold:
                cornerstone_layers.append(layer_idx)
                logger.info(
                    "Layer %s is cornerstone (PPL increase: %.2f%%)",
                    layer_idx,
                    ppl_increase * 100,
                )

        return cornerstone_layers
