"""
Visualization utilities for Experiment 2 (mechanistic interpretability).
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_style() -> None:
    """Set consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_pca_alignment(
    alignments: Dict[str, Dict],
    n_pcs: int = 5,
    save_path: Optional[str] = None,
):
    """Plot SPON bias alignment with principal components."""
    set_style()

    layers = sorted(alignments.keys(), key=lambda x: int(x.split("_")[1]))
    n_layers = len(layers)

    alignment_matrix = np.zeros((n_layers, n_pcs))
    for i, layer in enumerate(layers):
        sims = alignments[layer]["pc_similarities"][:n_pcs]
        alignment_matrix[i, : len(sims)] = sims

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        alignment_matrix,
        xticklabels=[f"PC{i + 1}" for i in range(n_pcs)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Cosine Similarity"},
    )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Layer")
    ax.set_title("SPON Bias Alignment with Hidden State PCs")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_category_shifts(
    category_results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """Plot L2 shifts by prompt category and layer."""
    set_style()

    categories = list(category_results.keys())
    first_cat = category_results[categories[0]]
    layers = sorted(first_cat["l2_shifts"].keys(), key=lambda x: int(x.split("_")[1]))
    n_layers = len(layers)

    data = np.zeros((len(categories), n_layers))
    for i, cat in enumerate(categories):
        for j, layer in enumerate(layers):
            data[i, j] = category_results[cat]["l2_shifts"].get(layer, 0)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_layers)
    width = 0.15

    for i, cat in enumerate(categories):
        offset = (i - len(categories) / 2) * width
        ax.bar(x + offset, data[i], width, label=cat.capitalize())

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Shift")
    ax.set_title("Representation Shift by Prompt Category")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], rotation=45)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
