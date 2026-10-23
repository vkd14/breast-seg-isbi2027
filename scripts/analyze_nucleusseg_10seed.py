#!/usr/bin/env python3
"""Aggregate NucleusSegData seeds and create measured submission figures."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from breastseg.data import load_nucleusseg
from train_public_study import save_json


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = {
    "Object-oriented": {"Huh7": (.9075, .8398), "HepG2": (.8421, .7645)},
    "ARGraphs": {"Huh7": (.8829, .8174), "HepG2": (.8028, .7024)},
    "Iterative h-minima": {"Huh7": (.8929, .7846), "HepG2": (.8322, .7177)},
}


def mean_sd(values):
    return {"mean": statistics.mean(values), "seed_sd": statistics.stdev(values),
            "minimum": min(values), "maximum": max(values)}


def normalized_gray(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99.5))
    return np.clip((image - low) / (high - low + 1e-7), 0, 1)


def overlay(image, labels, color):
    gray = normalized_gray(image)
    output = np.repeat(gray[..., None], 3, axis=2)
    output[find_boundaries(labels, mode="inner")] = color
    return output


def comparison_figure(aggregate, output):
    methods = ["Ours"] + list(PUBLISHED)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 4.7), constrained_layout=True)
    for axis, (subset, metric, title) in zip(axes.ravel(), (
            ("Huh7", "nucleus_f1", "Huh7 nucleus F1"),
            ("Huh7", "pixel_f1", "Huh7 pixel F1"),
            ("HepG2", "nucleus_f1", "HepG2 nucleus F1"),
            ("HepG2", "pixel_f1", "HepG2 pixel F1"))):
        metric_index = 0 if metric == "nucleus_f1" else 1
        values = [aggregate[subset][metric]["mean"]] + [PUBLISHED[name][subset][metric_index] for name in PUBLISHED]
        positions = np.arange(len(methods))
        axis.bar(positions, values, color=colors, width=.72)
        axis.errorbar(
            positions[0], values[0], yerr=aggregate[subset][metric]["seed_sd"],
            fmt="none", color="black", capsize=3, linewidth=1.2)
        axis.set_ylim(max(0, min(values) - .08), 1.0)
        axis.set_title(title, fontsize=9)
        axis.set_xticks(positions, ["Ours", "Object-\noriented", "ARGraphs", "Iterative\nh-minima"], fontsize=7)
        axis.set_ylabel("F1", fontsize=8)
        axis.grid(axis="y", alpha=.25)
        for position, value in zip(positions, values):
            axis.text(position, value + .008, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def output_figure(study, summaries, cases, output):
    seed_scores = {
        summary["seed"]: np.mean([
            summary["subsets"][subset]["bilkent_micro"]["nucleus_f1"]
            for subset in ("Huh7", "HepG2")])
        for summary in summaries
    }
    ordered = sorted(seed_scores, key=seed_scores.get)
    median_seed = ordered[len(ordered) // 2]
    run = study / f"vanilla_contact_b7_s{median_seed}"
    with (run / "test_per_image.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    case_by_id = {case["id"]: case for case in cases}
    selected = {}
    for subset in ("Huh7", "HepG2"):
        subset_rows = [row for row in rows if row["subset"] == subset]
        values = [float(row["bilkent_nucleus_f1"]) for row in subset_rows]
        median = float(np.median(values))
        selected[subset] = min(subset_rows, key=lambda row: abs(float(row["bilkent_nucleus_f1"]) - median))

    figure, axes = plt.subplots(2, 4, figsize=(7.1, 4.2), constrained_layout=True)
    for row_index, subset in enumerate(("Huh7", "HepG2")):
        result = selected[subset]
        case = case_by_id[result["id"]]
        prediction = cv2.imread(
            str(run / "predictions" / f"{case['id']}_instances.tif"), cv2.IMREAD_UNCHANGED)
        boundaries = []
        for seed in range(10):
            path = study / f"vanilla_contact_b7_s{seed}" / "predictions" / f"{case['id']}_instances.tif"
            labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            boundary = find_boundaries(labels, mode="inner")
            boundaries.append(ndi.binary_dilation(boundary, iterations=1))
        frequency = np.mean(boundaries, axis=0)
        panels = (
            (normalized_gray(case["image"]), f"({chr(97 + row_index * 4)}) {subset} input", "gray"),
            (overlay(case["image"], case["instances"], (1, .15, .05)),
             f"({chr(98 + row_index * 4)}) reference", None),
            (overlay(case["image"], prediction, (.05, .9, .25)),
             f"({chr(99 + row_index * 4)}) median-seed output", None),
            (frequency, f"({chr(100 + row_index * 4)}) 10-seed boundary frequency", "magma"),
        )
        for axis, (panel, title, cmap) in zip(axes[row_index], panels):
            image = axis.imshow(panel, cmap=cmap, vmin=0, vmax=1)
            axis.set_title(title, fontsize=7.5)
            axis.axis("off")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"median_seed": median_seed,
            "representative_images": {subset: row["id"] for subset, row in selected.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=ROOT / "results/nucleusseg_contact_10seed")
    parser.add_argument("--data", type=Path, default=ROOT / "data/nucleusseg")
    parser.add_argument("--manuscript", type=Path, default=ROOT / "submission/isbi2027_overleaf")
    args = parser.parse_args()
    summaries = []
    for seed in range(10):
        path = args.study / f"vanilla_contact_b7_s{seed}" / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed seed: {path}")
        summaries.append(json.loads(path.read_text()))
    aggregate = {"n_seeds": 10, "published_values_are_aggregate_only": True}
    for subset in ("Huh7", "HepG2"):
        aggregate[subset] = {}
        for metric in ("nucleus_precision", "nucleus_recall", "nucleus_f1",
                       "pixel_precision", "pixel_recall", "pixel_f1"):
            aggregate[subset][metric] = mean_sd([
                summary["subsets"][subset]["bilkent_micro"][metric] for summary in summaries])
    aggregate["selected_contact_thresholds"] = [summary["contact_threshold"] for summary in summaries]
    aggregate["selected_min_sizes"] = [summary["min_size"] for summary in summaries]
    args.manuscript.mkdir(parents=True, exist_ok=True)
    comparison_figure(aggregate, args.manuscript / "fig_nucleusseg_comparison.png")
    figure_selection = output_figure(args.study, summaries, load_nucleusseg(args.data),
                                     args.manuscript / "fig_nucleusseg_outputs.png")
    aggregate["qualitative_figure_selection"] = figure_selection
    save_json(args.study / "aggregate.json", aggregate)
    with (args.study / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subset", "metric", "mean", "seed_sd", "minimum", "maximum"))
        for subset in ("Huh7", "HepG2"):
            for metric, values in aggregate[subset].items():
                writer.writerow((subset, metric, values["mean"], values["seed_sd"],
                                 values["minimum"], values["maximum"]))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
