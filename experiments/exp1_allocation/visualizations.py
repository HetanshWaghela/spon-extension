"""
Visualization utilities for Experiment 1 (allocation sweep).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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


def plot_pareto_frontier(
    results: List[Dict],
    x_key: str = "relative_params",
    y_key: str = "perplexity",
    label_key: str = "config_name",
    save_path: Optional[str] = None,
):
    """Plot Pareto frontier of configuration results."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = [r[x_key] for r in results]
    y = [r[y_key] for r in results]
    labels = [r[label_key] for r in results]

    ax.scatter(x, y, s=100, alpha=0.6, edgecolors="black", linewidths=1)

    from src.evaluation import compute_pareto_frontier

    pareto = compute_pareto_frontier(results, x_key, y_key)
    pareto_x = [p[x_key] for p in pareto]
    pareto_y = [p[y_key] for p in pareto]

    sorted_pareto = sorted(zip(pareto_x, pareto_y))
    pareto_x_sorted = [p[0] for p in sorted_pareto]
    pareto_y_sorted = [p[1] for p in sorted_pareto]

    ax.plot(pareto_x_sorted, pareto_y_sorted, "r-", linewidth=2, label="Pareto Frontier")
    ax.scatter(pareto_x, pareto_y, s=150, c="red", marker="*", zorder=5, label="Pareto Optimal")

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=9, alpha=0.8)

    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_title("SPON Allocation: Pareto Frontier")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layer_sensitivity_heatmap(
    sensitivity_by_config: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot heatmap of layer sensitivity across configurations."""
    set_style()

    config_names = list(sensitivity_by_config.keys())
    n_layers = len(list(sensitivity_by_config.values())[0])
    data = np.array([sensitivity_by_config[c] for c in config_names])

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        data,
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=config_names,
        cmap="RdYlBu_r",
        center=0,
        ax=ax,
        cbar_kws={"label": "Sensitivity"},
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Configuration")
    ax.set_title("Layer Sensitivity to SPON")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_layer_importance_comparison(
    importance_scores: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """Compare multiple layer-importance methods."""
    set_style()

    methods = list(importance_scores.keys())
    n_layers = len(list(importance_scores.values())[0])

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        scores = importance_scores[method]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        colors = plt.cm.RdYlBu_r(scores_norm)
        ax.barh(range(n_layers), scores, color=colors)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])
        ax.set_xlabel("Importance Score")
        ax.set_title(f"{method.upper()} Importance")
        ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_loss(training_losses: List[float], save_path: Optional[str] = None):
    """Plot training loss curve."""
    set_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_losses, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence Loss")
    ax.set_title("SPON Training Loss")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_results_report(results_dir: str, output_path: str = "report.html") -> None:
    """Generate a minimal HTML report from result JSON files."""
    results_dir = Path(results_dir)
    result_files = list(results_dir.glob("*.json"))

    html = [
        "<html><head><style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        ".section { margin: 20px 0; }",
        "h2 { color: #333; }",
        "</style></head><body>",
        "<h1>SPON Allocation Experiment Results</h1>",
    ]

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        html.append(f"<div class='section'><h2>{result_file.stem}</h2>")

        if "results" in data:
            html.append("<table><tr><th>Config</th><th>Sparsity</th><th>PPL</th><th>Rel. Params</th></tr>")
            for row in data["results"]:
                html.append(f"<tr><td>{row.get('config_name', 'N/A')}</td>")
                html.append(f"<td>{row.get('sparsity', 'N/A')}</td>")
                html.append(f"<td>{row.get('perplexity', 'N/A'):.2f}</td>")
                html.append(f"<td>{row.get('relative_params', 'N/A')}</td></tr>")
            html.append("</table>")

        html.append("</div>")

    html.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html))

    print(f"Report saved to {output_path}")
