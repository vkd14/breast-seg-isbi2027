#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for the EMBC paper.

Generates:
  1. SOTA bar chart (only converged baselines, y-axis starts at 0)
  2. Ablation bar chart (y-axis starts at 0 to show collapse)
  3. Box plot (only proposed + converged SOTA baselines + no-gabor ablation)
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


def fig_sota_bars(results_dir, out_dir):
    """Bar chart for SOTA comparison — only converged methods."""
    sota_names = [
        ("proposed_full", "Proposed"),
        ("sota_attention_unet_resnet50", "Att. U-Net\n(R50)"),
        ("sota_deeplabv3p_resnet50", "DeepLabV3+\n(R50)"),
        ("sota_fpn_resnet50", "FPN\n(R50)"),
        ("sota_unet_resnet34", "U-Net\n(R34)"),
        ("sota_unet_resnet50", "U-Net\n(R50)"),
    ]

    names, means, stds = [], [], []
    for key, label in sota_names:
        exp = load_experiment(results_dir, key)
        if exp:
            dice_mean = exp["dice"]["mean"] * 100
            if dice_mean > 10:
                names.append(label)
                means.append(dice_mean)
                stds.append(exp["dice"]["std"] * 100)

    if not names:
        print("No SOTA results found for bar chart.")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(names))
    colors = ["#2196F3" if "Proposed" in n else "#90CAF9" for n in names]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                   edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Dice Score (%)")
    ax.set_title("Comparison with State-of-the-Art Methods")
    ax.axhline(95, color="green", ls="--", lw=0.8, alpha=0.7, label="95% target")
    ax.axhline(90, color="orange", ls="--", lw=0.8, alpha=0.5, label="90% clinical")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, m) in enumerate(zip(bars, means)):
        y_pos = min(m + 1.5, 103)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=7,
                fontweight="bold" if i == 0 else "normal")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_sota_bars.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_sota_bars.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_sota_bars.pdf'}")


def fig_ablation_bars(results_dir, out_dir):
    """Ablation bar chart — y-axis from 0 to show collapse clearly."""
    abl_names = [
        ("proposed_full", "Full\nModel"),
        ("ablation_no_gabor", "w/o\nGabor"),
        ("ablation_unet_not_unetpp", "w/o\nUNet++"),
        ("ablation_no_weighted_sampling", "w/o Wt.\nSampling"),
        ("ablation_no_scse", "w/o\nSCSE"),
        ("ablation_standard_loss", "w/o Stab.\nLoss"),
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
    colors = []
    for m in means:
        if m > 90:
            colors.append("#2196F3")
        elif m > 20:
            colors.append("#FFB74D")
        else:
            colors.append("#EF5350")

    errs = [s if m > 5 else 0 for m, s in zip(means, stds)]

    bars = ax.bar(x, means, yerr=errs, capsize=3, color=colors,
                   edgecolor="black", linewidth=0.5, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Dice Score (%)")
    ax.set_title("Ablation Study: Contribution of Each Component")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    base = means[0]
    for i, (bar, m) in enumerate(zip(bars, means)):
        y_pos = max(m + 1.5, 4)
        label = f"{m:.1f}%"
        if i > 0:
            delta = m - base
            label = f"{m:.1f}%\n({delta:+.1f})"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                label, ha="center", va="bottom", fontsize=6.5,
                color="black" if m > 20 else "#B71C1C",
                fontweight="bold" if i == 0 else "normal")

    plt.tight_layout()
    plt.savefig(out_dir / "fig_ablation_bars.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_ablation_bars.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_ablation_bars.pdf'}")


def fig_boxplot(results_dir, out_dir):
    """Box plot — only methods with meaningful distributions."""
    include = [
        ("proposed_full", "Proposed"),
        ("ablation_no_gabor", "w/o Gabor"),
        ("sota_attention_unet_resnet50", "Att. U-Net (R50)"),
        ("sota_deeplabv3p_resnet50", "DeepLabV3+ (R50)"),
        ("sota_fpn_resnet50", "FPN (R50)"),
        ("sota_unet_resnet34", "U-Net (R34)"),
        ("sota_unet_resnet50", "U-Net (R50)"),
    ]

    names, values = [], []
    for key, label in include:
        exp = load_experiment(results_dir, key)
        if exp and "dice" in exp and "values" in exp["dice"]:
            if exp["dice"]["mean"] > 0.10:
                names.append(label)
                values.append(exp["dice"]["values"])

    if not names:
        print("No per-image data found for boxplot.")
        return

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bp = ax.boxplot(values, vert=True, patch_artist=True, labels=names,
                     widths=0.6, showfliers=True,
                     flierprops=dict(markersize=3, markerfacecolor="gray"))

    for i, (patch, name) in enumerate(zip(bp["boxes"], names)):
        if name == "Proposed":
            patch.set_facecolor("#2196F3")
            patch.set_alpha(0.85)
        elif "w/o" in name:
            patch.set_facecolor("#FFB74D")
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor("#E0E0E0")
            patch.set_alpha(0.7)

    ax.set_ylabel("Dice Score")
    ax.set_title("Per-Image Dice Score Distribution")
    ax.axhline(0.95, color="green", ls="--", lw=0.8, alpha=0.7, label="95% target")
    ax.axhline(0.90, color="orange", ls="--", lw=0.8, alpha=0.5, label="90% clinical")
    ax.legend(loc="lower left", fontsize=7)
    plt.xticks(rotation=25, ha="right")
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_boxplot_comparison.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "fig_boxplot_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir / 'fig_boxplot_comparison.pdf'}")


def fig_histogram(results_dir, out_dir):
    """Per-image Dice histogram for proposed model."""
    exp = load_experiment(results_dir, "proposed_full")
    if not exp or "values" not in exp.get("dice", {}):
        print("No proposed results for histogram.")
        return

    values = np.array(exp["dice"]["values"])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(values * 100, bins=20, color="#2196F3", edgecolor="black",
            linewidth=0.5, alpha=0.8, range=(0, 100))
    ax.axvline(values.mean() * 100, color="red", ls="--", lw=1.2,
               label=f"Mean: {values.mean()*100:.1f}%")
    ax.axvline(95, color="green", ls="--", lw=0.8,
               label="95% target")
    ax.set_xlabel("Dice Score (%)")
    ax.set_ylabel("Number of Test Images")
    ax.set_title("Distribution of Per-Image Dice Scores (Proposed)")
    ax.legend(fontsize=7)
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
    fig_sota_bars(results_dir, fig_dir)
    fig_ablation_bars(results_dir, fig_dir)
    fig_boxplot(results_dir, fig_dir)
    fig_histogram(results_dir, fig_dir)
    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()