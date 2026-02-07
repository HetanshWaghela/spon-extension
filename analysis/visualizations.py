"""
Backward-compatible visualization imports.

Primary visualization modules now live under:
- experiments.exp1_allocation.visualizations
- experiments.exp2_interpretability.visualizations
"""

from experiments.exp1_allocation.visualizations import (
    create_results_report,
    plot_layer_importance_comparison,
    plot_layer_sensitivity_heatmap,
    plot_pareto_frontier,
    plot_training_loss,
)
from experiments.exp2_interpretability.visualizations import (
    plot_category_shifts,
    plot_pca_alignment,
)

__all__ = [
    "create_results_report",
    "plot_layer_importance_comparison",
    "plot_layer_sensitivity_heatmap",
    "plot_pareto_frontier",
    "plot_training_loss",
    "plot_category_shifts",
    "plot_pca_alignment",
]
