#!/usr/bin/env python3
"""Create unrestricted-draft figures from real public images and measured predictions."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from breastseg.core import Gabor, grayscale_uint8
from breastseg.data import load_bbbc039
from breastseg.external import load_external


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "full_draft_figures"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def median_case(pattern, metric="dice"):
    values = defaultdict(list)
    groups = {}
    for path in ROOT.glob(pattern):
        for row in rows(path):
            values[row["id"]].append(float(row[metric]))
            groups[row["id"]] = row["group"]
    if not values or any(len(value) != 3 for value in values.values()):
        raise RuntimeError(f"Expected three complete seeds: {pattern}")
    averaged = {key: float(np.mean(value)) for key, value in values.items()}
    median = float(np.median(list(averaged.values())))
    selected = min(averaged, key=lambda key: (abs(averaged[key] - median), key))
    return selected, groups[selected], averaged[selected], median


def prediction(path):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return value > 0


def overlay(image, mask):
    if image.ndim == 2:
        rgb = np.repeat((image.astype(np.float32) / max(float(image.max()), 1))[..., None], 3, 2)
    else:
        rgb = image.astype(np.float32)
        if rgb.max() > 1:
            rgb /= 255
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    result = rgb.copy()
    result[boundary] = [1, .15, .05]
    return np.clip(result, 0, 1)


def dataset_prediction_figure(data_root):
    bbbc_cases = {case["id"]: case for case in load_bbbc039(data_root / "bbbc039", ("test",))}
    tnbc_cases = {case["id"]: case for case in load_external(data_root, "tnbc")}
    bbbc_id, _, bbbc_score, _ = median_case(
        "results/public_study/vanilla_b7_s*/test_per_image.csv"
    )
    tnbc_id, tnbc_patient, tnbc_score, _ = median_case(
        "results/tnbc_grouped_cv/vanilla_b7_f*_s*/test_per_image.csv"
    )
    config = json.loads((ROOT / "configs/tnbc_grouped_cv.json").read_text())
    tnbc_fold = next(
        index for index, groups in enumerate(config["outer_folds"])
        if tnbc_patient in groups
    )

    bbbc = bbbc_cases[bbbc_id]
    tnbc = tnbc_cases[tnbc_id]
    bbbc_gray = grayscale_uint8(bbbc["image"])
    tnbc_gray = grayscale_uint8(tnbc["image"])
    panels = [
        [
            bbbc_gray,
            Gabor()(bbbc_gray),
            bbbc["mask"],
            prediction(ROOT / "results/public_study/vanilla_b7_s1/predictions" / f"{bbbc_id}.png"),
            prediction(ROOT / "results/public_study/gabor_b7_s1/predictions" / f"{bbbc_id}.png"),
            prediction(ROOT / "results/public_architecture_baselines/unetpp_xception_scse_s1/predictions" / f"{bbbc_id}.png"),
            overlay(bbbc_gray, bbbc["mask"]),
        ],
        [
            tnbc["image"],
            Gabor()(tnbc_gray),
            tnbc["mask"],
            prediction(ROOT / f"results/tnbc_grouped_cv/vanilla_b7_f{tnbc_fold}_s1/predictions" / f"{tnbc_id}.png"),
            prediction(ROOT / f"results/tnbc_grouped_cv/gabor_b7_f{tnbc_fold}_s1/predictions" / f"{tnbc_id}.png"),
            prediction(ROOT / f"results/tnbc_grouped_cv/unetpp_xception_scse_f{tnbc_fold}_s1/predictions" / f"{tnbc_id}.png"),
            overlay(tnbc["image"], tnbc["mask"]),
        ],
    ]
    titles = [
        "Native input",
        "24-filter Gabor map",
        "Reference mask",
        "Vanilla B7",
        "Gabor bundle",
        "Xception/SCSE",
        "Reference boundary",
    ]
    fig, axes = plt.subplots(2, 7, figsize=(7.2, 2.45))
    for row_index, row in enumerate(panels):
        for column_index, panel in enumerate(row):
            axis = axes[row_index, column_index]
            axis.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            if row_index == 0:
                axis.set_title(titles[column_index], fontsize=6.4)
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            "BBBC039\nfluorescence" if row_index == 0 else "TNBC\nH&E histology",
            fontsize=7,
        )
    fig.tight_layout(pad=.15, w_pad=.12, h_pad=.18)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"public_dataset_examples.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    (OUT / "public_dataset_examples.json").write_text(json.dumps({
        "selection_rule": "image nearest the median three-seed vanilla Dice within each dataset",
        "bbbc039": {"id": bbbc_id, "three_seed_vanilla_dice": bbbc_score, "display_seed": 1},
        "tnbc": {
            "id": tnbc_id,
            "patient": tnbc_patient,
            "fold": tnbc_fold,
            "three_seed_vanilla_dice": tnbc_score,
            "display_seed": 1,
        },
        "images": "real provider images; no generated microscopy content",
    }, indent=2))


def cross_dataset_figure():
    study = json.loads((ROOT / "results/study_summary.json").read_text())
    bbbc_baselines = json.loads(
        (ROOT / "results/public_architecture_baselines/summary.json").read_text()
    )
    tnbc = json.loads((ROOT / "results/tnbc_grouped_cv/summary.json").read_text())
    bbbc_lookup = {record["variant"]: record for record in study}
    xception = next(
        record for record in bbbc_baselines
        if record["model"] == "unetpp_xception_scse"
    )
    names = ["Vanilla B7", "Gabor bundle", "Xception/SCSE"]
    bbbc_values = [
        bbbc_lookup["vanilla_b7"]["dice"],
        bbbc_lookup["gabor_b7"]["dice"],
        xception["dice"],
    ]
    bbbc_errors = [
        bbbc_lookup["vanilla_b7"]["dice_seed_sd"],
        bbbc_lookup["gabor_b7"]["dice_seed_sd"],
        xception["dice_seed_sd"],
    ]
    tnbc_lookup = {record["model"]: record for record in tnbc}
    tnbc_values = [
        tnbc_lookup["vanilla_b7"]["dice"],
        tnbc_lookup["gabor_b7"]["dice"],
        tnbc_lookup["unetpp_xception_scse"]["dice"],
    ]
    tnbc_errors = np.array([
        [
            tnbc_lookup[key]["dice"] - tnbc_lookup[key]["dice_ci95"][0],
            tnbc_lookup[key]["dice_ci95"][1] - tnbc_lookup[key]["dice"],
        ]
        for key in ("vanilla_b7", "gabor_b7", "unetpp_xception_scse")
    ]).T
    colors = ["#4c78a8", "#e45756", "#72b7b2"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), gridspec_kw={"width_ratios": [1, 1, .8]})
    x = np.arange(3)
    axes[0].bar(x, np.array(bbbc_values) * 100, yerr=np.array(bbbc_errors) * 100,
                color=colors, edgecolor="black", linewidth=.5, capsize=2)
    axes[0].set_ylim(96.3, 97.05)
    axes[0].set_ylabel("Semantic Dice (%)")
    axes[0].set_title("BBBC039 official test\nmean ± seed SD")
    axes[1].bar(x, np.array(tnbc_values) * 100, yerr=tnbc_errors * 100,
                color=colors, edgecolor="black", linewidth=.5, capsize=2)
    axes[1].set_ylim(58, 80)
    axes[1].set_title("TNBC patient-grouped CV\n95% patient-bootstrap CI")
    delta = [
        100 * (bbbc_values[1] - bbbc_values[0]),
        100 * (tnbc_values[1] - tnbc_values[0]),
    ]
    axes[2].bar([0, 1], delta, color=["#59a14f" if value >= 0 else "#e45756" for value in delta],
                edgecolor="black", linewidth=.5)
    axes[2].axhline(0, color="black", linewidth=.7)
    axes[2].set_title("Gabor − vanilla")
    axes[2].set_ylabel("Dice-point difference")
    axes[2].set_xticks([0, 1], ["BBBC039", "TNBC"], rotation=18)
    for axis in axes[:2]:
        axis.set_xticks(x, names, rotation=28, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
        axis.title.set_fontsize(8)
    fig.tight_layout(pad=.35, w_pad=.8)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"cross_dataset_results.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_prediction_figure(ROOT / "data")
    cross_dataset_figure()


if __name__ == "__main__":
    main()
