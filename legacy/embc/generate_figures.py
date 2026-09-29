#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for the EMBC paper.

Generates:
  1. Box plot comparing Dice distributions across all methods
  2. Bar chart of Dice ± std for SOTA comparison
  3. Ablation bar chart
  4. Per-image Dice histogram for proposed model

Usage:
    python generate_figures.py --results_dir ./results
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# IEEE-friendly figure settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load_experiment(results_dir, name):
    path = results_dir / name / "test_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig_boxplot(results_dir, out_dir):
    """Box plot of per-image Dice across all methods."""
    # Collect all experiments with per-image values
    data = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        exp = load_experiment(results_dir, d.name)
        if exp and "dice" in exp and "values" in exp["dice"]:
            # Clean up name for display
            label = d.name.replace("sota_", "").replace("ablation_", "abl: ").replace("_", " ")
            if d.name == "proposed_full":
                label = "Proposed"
            data[label] = exp["dice"]["values"]

    if not data:
        print("No per-image data found for boxplot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(data.keys())
    values = [data[k] for k in labels]

    bp = ax.boxplot(values, vert=True, patch_artist=True, labels=labels,
                     widths=0.6, showfliers=True, flierprops=dict(markersize=3))

    # Color proposed differently
    for i, (patch, label) in enumerate(zip(bp["boxes"], labels)):
        if label == "Proposed":
            patch.set_facecolor("#2196F3")
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor("#E0E0E0")
            patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Image Dice Score Distribution Across Methods")
    ax.axhline(0.95, color="green", ls="--", lw=0.8, alpha=0.7, label="95% target")
    ax.axhline(0.90, color="orange", ls="--", lw=0.8, alpha=0.7, label="90% clinical threshold")
    ax.legend(loc="lower left")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_boxplot_comparison.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_boxplot_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_boxplot_comparison.pdf'}")


def fig_sota_bars(results_dir, out_dir):
    """Bar chart for SOTA comparison."""
    sota_names = [
        ("proposed_full", "Proposed"),
        ("sota_unet_resnet34", "U-Net (R34)"),
        ("sota_unet_resnet50", "U-Net (R50)"),
        ("sota_unetpp_resnet50", "UNet++ (R50)"),
        ("sota_attention_unet_resnet50", "Att. U-Net"),
        ("sota_deeplabv3p_resnet50", "DeepLabV3+"),
        ("sota_manet_resnet50", "MA-Net"),
        ("sota_fpn_resnet50", "FPN"),
    ]

    names, means, stds = [], [], []
    for key, label in sota_names:
        exp = load_experiment(results_dir, key)
        if exp:
            names.append(label)
            means.append(exp["dice"]["mean"] * 100)
            stds.append(exp["dice"]["std"] * 100)

    if not names:
        print("No SOTA results found for bar chart.")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names))
    colors = ["#2196F3" if n == "Proposed" else "#90CAF9" for n in names]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                   edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Dice Score (%)")
    ax.set_title("Comparison with State-of-the-Art Methods")
    ax.axhline(95, color="green", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylim(70, 100)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{m:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_dir / "fig_sota_bars.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_sota_bars.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_sota_bars.pdf'}")


def fig_ablation_bars(results_dir, out_dir):
    """Bar chart for ablation study."""
    abl_names = [
        ("proposed_full", "Full Model"),
        ("ablation_no_gabor", "w/o Gabor"),
        ("ablation_no_weighted_sampling", "w/o Weighted\nSampling"),
        ("ablation_standard_loss", "w/o Stabilized\nLoss"),
        ("ablation_no_scse", "w/o SCSE"),
        ("ablation_unet_not_unetpp", "w/o UNet++"),
    ]

    names, means, stds = [], [], []
    for key, label in abl_names:
        exp = load_experiment(results_dir, key)
        if exp:
            names.append(label)
            means.append(exp["dice"]["mean"] * 100)
            stds.append(exp["dice"]["std"] * 100)

    if not names:
        print("No ablation results found.")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(names))
    colors = ["#2196F3"] + ["#FFAB91"] * (len(names) - 1)

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                   edgecolor="black", linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Dice Score (%)")
    ax.set_title("Ablation Study: Contribution of Each Component")
    ax.set_ylim(max(70, min(means) - 5), 100)
    ax.grid(axis="y", alpha=0.3)

    # Delta labels
    if len(means) > 1:
        base = means[0]
        for bar, m in zip(bars[1:], means[1:]):
            delta = m - base
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{delta:+.1f}", ha="center", va="bottom", fontsize=7,
                    color="red" if delta < 0 else "green")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_ablation_bars.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ablation_bars.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_ablation_bars.pdf'}")


def fig_histogram(results_dir, out_dir):
    """Per-image Dice histogram for proposed model."""
    exp = load_experiment(results_dir, "proposed_full")
    if not exp or "values" not in exp.get("dice", {}):
        print("No proposed results for histogram.")
        return

    values = np.array(exp["dice"]["values"])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(values * 100, bins=20, color="#2196F3", edgecolor="black",
            linewidth=0.5, alpha=0.8)
    ax.axvline(values.mean() * 100, color="red", ls="--", lw=1.2,
               label=f"Mean: {values.mean()*100:.1f}%")
    ax.axvline(95, color="green", ls="--", lw=0.8,
               label="95% target")
    ax.set_xlabel("Dice Score (%)")
    ax.set_ylabel("Number of Test Images")
    ax.set_title("Distribution of Per-Image Dice Scores (Proposed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_dice_histogram.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_dice_histogram.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_dice_histogram.pdf'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Generating publication figures...")
    fig_boxplot(results_dir, fig_dir)
    fig_sota_bars(results_dir, fig_dir)
    fig_ablation_bars(results_dir, fig_dir)
    fig_histogram(results_dir, fig_dir)
    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
