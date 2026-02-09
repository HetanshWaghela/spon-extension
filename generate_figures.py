"""Generate all figures for FINDINGS.md from experimental results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})
PALETTE = sns.color_palette("Set2", 8)
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)

# Find the run directory
runs_dir = os.path.join(ROOT, "results", "allocation_sweep", "runs")
run_id = sorted(os.listdir(runs_dir))[-1]
run_dir = os.path.join(runs_dir, run_id)

with open(os.path.join(run_dir, "results.json")) as f:
    raw = json.load(f)

# Flatten results into a convenient dict + compute dense PPL
results_data = {"configurations": {}}
for key, val in raw["results"].items():
    results_data["configurations"][key] = val

# Derive dense PPL from the first entry: ppl / ppl_vs_dense
first_entry = list(raw["results"].values())[0]
results_data["dense_perplexity"] = first_entry["perplexity"] / first_entry["ppl_vs_dense"]

with open(os.path.join(ROOT, "results", "allocation_sweep", "aggregated", "pareto.json")) as f:
    pareto_data = json.load(f)

# Find interpretability file
interp_dir = os.path.join(ROOT, "results", "interpretability")
interp_file = sorted([f for f in os.listdir(interp_dir) if f.endswith(".json")])[-1]
with open(os.path.join(interp_dir, interp_file)) as f:
    interp_data = json.load(f)


# ── Helper ─────────────────────────────────────────────────────────────
def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


# =====================================================================
# FIGURE 1 – Perplexity comparison bar chart (grouped by sparsity)
# =====================================================================
def fig1_perplexity_bars():
    configs_order = ["BASELINE-TEAL", "BOTTOM-50", "TOP-50", "TOP-75", "UNIF-ALL", "ATTN-ONLY"]
    dense_ppl = results_data["dense_perplexity"]

    data_50, data_60 = {}, {}
    for key, val in results_data["configurations"].items():
        name = val["config_name"]
        s = val["sparsity"]
        if s == 0.5:
            data_50[name] = val["perplexity"]
        else:
            data_60[name] = val["perplexity"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(configs_order))
    w = 0.32

    bars1 = ax.bar(x - w/2, [data_50.get(c, 0) for c in configs_order], w,
                   label="50 % sparsity", color=PALETTE[0], edgecolor="white", linewidth=0.6)
    bars2 = ax.bar(x + w/2, [data_60.get(c, 0) for c in configs_order], w,
                   label="60 % sparsity", color=PALETTE[1], edgecolor="white", linewidth=0.6)

    ax.axhline(dense_ppl, color="#d62728", ls="--", lw=1.3, label=f"Dense baseline ({dense_ppl:.2f})")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs_order, rotation=20, ha="right")
    ax.set_ylabel("Perplexity (↓ lower is better)")
    ax.set_title("Experiment 1: Perplexity by SPON Configuration")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(21.2, 24.5)
    savefig(fig, "fig1_perplexity_bars.png")

# =====================================================================
# FIGURE 2 – Damage recovery comparison
# =====================================================================
def fig2_damage_recovery():
    dense = results_data["dense_perplexity"]

    modules = ["down_proj\n(MLP)", "o_proj\n(Attention)"]
    # down_proj best = UNIF-ALL, o_proj = ATTN-ONLY
    # 50%
    teal_dp50, spon_dp50 = 22.601, 22.382
    teal_op50, spon_op50 = 22.19, 21.747  # o_proj baselines from FINDINGS
    # 60%
    teal_dp60, spon_dp60 = 23.929, 23.396
    teal_op60, spon_op60 = 22.73, 21.887

    rec_dp50 = (teal_dp50 - spon_dp50)/(teal_dp50 - dense)*100
    rec_op50 = (teal_op50 - spon_op50)/(teal_op50 - dense)*100
    rec_dp60 = (teal_dp60 - spon_dp60)/(teal_dp60 - dense)*100
    rec_op60 = (teal_op60 - spon_op60)/(teal_op60 - dense)*100

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.32
    bars1 = ax.bar(x - w/2, [rec_dp50, rec_op50], w, label="50 % sparsity",
                   color=PALETTE[2], edgecolor="white")
    bars2 = ax.bar(x + w/2, [rec_dp60, rec_op60], w, label="60 % sparsity",
                   color=PALETTE[3], edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(modules, fontsize=12)
    ax.set_ylabel("Sparsification Damage Recovered (%)")
    ax.set_title("Where Should SPON Go? Attention Recovers 3-4× More Damage")
    ax.legend()
    ax.set_ylim(0, 100)
    savefig(fig, "fig2_damage_recovery.png")

# =====================================================================
# FIGURE 3 – Pareto frontier
# =====================================================================
def fig3_pareto():
    configs_order = ["BASELINE-TEAL", "BOTTOM-50", "TOP-50", "TOP-75", "UNIF-ALL", "ATTN-ONLY"]
    rel_params = {"BASELINE-TEAL": 0.0, "BOTTOM-50": 0.5, "TOP-50": 0.5,
                  "TOP-75": 0.75, "UNIF-ALL": 1.0, "ATTN-ONLY": 1.0}
    ppls = {}
    for val in results_data["configurations"].values():
        if val["sparsity"] == 0.5:
            ppls[val["config_name"]] = val["perplexity"]

    pareto_names = {p["config_name"] for p in pareto_data}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in configs_order:
        is_pareto = c in pareto_names
        color = PALETTE[0] if is_pareto else "#999999"
        marker = "★" if is_pareto else "o"
        size = 180 if is_pareto else 80
        ax.scatter(rel_params[c], ppls[c], s=size, c=[color], zorder=5,
                   edgecolors="black" if is_pareto else "#666666", linewidths=0.8)
        offset_y = -0.12 if c != "ATTN-ONLY" else 0.08
        offset_x = 0.02
        ax.annotate(c, (rel_params[c], ppls[c]),
                    textcoords="offset points",
                    xytext=(5, -15 if c != "ATTN-ONLY" else 10),
                    fontsize=8.5, fontweight="bold" if is_pareto else "normal",
                    color="black" if is_pareto else "#666666")

    # Draw Pareto frontier line
    pareto_points = sorted([(p["relative_params"], p["perplexity"]) for p in pareto_data])
    px, py = zip(*pareto_points)
    ax.plot(px, py, "--", color=PALETTE[0], alpha=0.6, lw=1.5, label="Pareto frontier")

    dense_ppl = results_data["dense_perplexity"]
    ax.axhline(dense_ppl, color="#d62728", ls=":", lw=1, alpha=0.7,
               label=f"Dense ({dense_ppl:.2f})")

    ax.set_xlabel("Relative SPON Parameters (1.0 = all 16 layers)")
    ax.set_ylabel("Perplexity (↓ lower is better)")
    ax.set_title("Pareto Frontier: Parameter Efficiency at 50% Sparsity")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(21.4, 22.8)
    savefig(fig, "fig3_pareto_frontier.png")

# =====================================================================
# FIGURE 4 – PCA alignment heatmap
# =====================================================================
def fig4_pca_heatmap():
    layers = sorted(interp_data["pca_alignments"].keys())
    matrix = []
    for layer in layers:
        sims = interp_data["pca_alignments"][layer]["pc_similarities"]
        matrix.append([abs(s) for s in sims])  # absolute values
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=0.1,
                annot=True, fmt=".3f", linewidths=0.5,
                xticklabels=[f"PC {i}" for i in range(10)],
                yticklabels=[l.replace("_down_proj", "") for l in layers],
                cbar_kws={"label": "|Cosine Similarity|"})
    ax.set_title("PCA Alignment: SPON Biases vs Top-10 Principal Components\n(Values near 0 = orthogonal → biases operate in different dimensions)")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Layer")
    savefig(fig, "fig4_pca_alignment.png")

# =====================================================================
# FIGURE 5 – Hidden-state shift (TEAL vs SPON)
# =====================================================================
def fig5_hidden_state_shift():
    shifts = interp_data["hidden_state_shifts"]
    layers = [f"layer_{i}" for i in range(8)]
    teal_shifts = [shifts[f"{l}_down_proj"]["teal_shift_l2"] for l in layers]
    spon_shifts = [shifts[f"{l}_down_proj"]["spon_shift_l2"] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(8)
    w = 0.35
    ax.bar(x - w/2, teal_shifts, w, label="TEAL only (no SPON)", color=PALETTE[4], edgecolor="white")
    ax.bar(x + w/2, spon_shifts, w, label="TEAL + SPON", color=PALETTE[5], edgecolor="white")

    # Annotate recovery %
    for i in range(8):
        rec = shifts[f"layer_{i}_down_proj"]["recovery_pct"]
        max_h = max(teal_shifts[i], spon_shifts[i])
        ax.text(x[i] + w/2, max_h + 0.003, f"{rec:+.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="#d62728", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {i}" for i in range(8)])
    ax.set_ylabel("L2 Distance from Dense Hidden States")
    ax.set_title("Hidden-State Shift: SPON Makes States MORE Different (Not Less!)\nNegative recovery % = SPON increases divergence from dense")
    ax.legend()
    savefig(fig, "fig5_hidden_state_shift.png")

# =====================================================================
# FIGURE 6 – Cross-config cosine similarity
# =====================================================================
def fig6_cross_config():
    cc = interp_data["cross_config_comparison"]
    layers = [f"layer_{i}_down_proj" for i in range(8)]
    sims = [cc[l]["cosine_similarity"] for l in layers]
    diffs = [cc[l]["l2_difference"] for l in layers]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = PALETTE[0]
    color2 = PALETTE[3]

    ax1.bar(range(8), sims, color=color1, alpha=0.85, edgecolor="white", label="Cosine similarity")
    ax1.set_ylabel("Cosine Similarity (TOP-50 vs UNIF-ALL)", color=color1)
    ax1.set_ylim(0.85, 1.0)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(range(8), diffs, "o-", color=color2, lw=2, markersize=7, label="L2 difference")
    ax2.set_ylabel("L2 Difference", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f"Layer {i}" for i in range(8)])
    ax1.set_title("Cross-Config Convergence: Different Training → Similar Biases\n(High cosine similarity = biases capture genuine damage structure)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    savefig(fig, "fig6_cross_config.png")

# =====================================================================
# FIGURE 7 – Bias statistics: norms + kurtosis across layers
# =====================================================================
def fig7_bias_stats():
    stats = interp_data["bias_statistics"]
    layers = [f"layer_{i}_down_proj" for i in range(8)]
    norms = [stats[l]["l2_norm"] for l in layers]
    kurtosis = [stats[l]["kurtosis"] for l in layers]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = PALETTE[1]
    color2 = PALETTE[6]

    bars = ax1.bar(range(8), norms, color=color1, alpha=0.85, edgecolor="white", label="L2 Norm")
    ax1.set_ylabel("Bias L2 Norm", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(range(8), kurtosis, "s-", color=color2, lw=2, markersize=7, label="Kurtosis")
    ax2.axhline(0, color="gray", ls=":", lw=0.8)
    ax2.set_ylabel("Kurtosis (negative = uniform spread)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(range(8))
    ax1.set_xticklabels([f"Layer {i}" for i in range(8)])
    ax1.set_title("SPON Bias Properties Across Layers\nDeeper layers → larger norms (more correction) + less uniform spread")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    savefig(fig, "fig7_bias_statistics.png")


# ── Run all ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1_perplexity_bars()
    fig2_damage_recovery()
    fig3_pareto()
    fig4_pca_heatmap()
    fig5_hidden_state_shift()
    fig6_cross_config()
    fig7_bias_stats()
    print(f"\nDone! {len(os.listdir(OUT))} figures saved to {OUT}/")
