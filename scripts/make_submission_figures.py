#!/usr/bin/env python3
"""Create submission figures only from measured images, masks, and CSV/JSON results."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from breastseg.core import Net
from breastseg.external import load_external
from evaluate_grayscale_transfer import explicit_gray, input_tensor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "submission_figures"


def result_figure():
    study = json.loads((ROOT / "results/study_summary.json").read_text())
    baselines = json.loads((ROOT / "results/public_architecture_baselines/summary.json").read_text())
    transfer = json.loads((ROOT / "results/grayscale_transfer_audit/aggregate.json").read_text())
    labels = ["B7", "Gabor", "G+Bd", "SCSE", "RG", "RS", "RG+B",
              "U34", "U50", "B0++", "Xcep++", "DLab", "FPN"]
    dice = [100 * x["dice"] for x in study] + [100 * next(x for x in baselines if x["model"] == key)["dice"]
                                                   for key in ("unet_resnet34", "unet_resnet50", "unetpp_effb0",
                                                               "unetpp_xception_scse", "deeplabv3p_resnet50", "fpn_resnet50")]
    sd = [100 * x["dice_seed_sd"] for x in study] + [100 * next(x for x in baselines if x["model"] == key)["dice_seed_sd"]
                                                        for key in ("unet_resnet34", "unet_resnet50", "unetpp_effb0",
                                                                    "unetpp_xception_scse", "deeplabv3p_resnet50", "fpn_resnet50")]
    colors = ["#4c78a8", "#e45756", "#f58518", "#72b7b2", "#54a24b", "#b279a2", "#ff9da6"] + ["#9d9da1"] * 6
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(labels))
    axes[0].bar(x, dice, yerr=sd, color=colors, linewidth=.4, edgecolor="black", capsize=1.8)
    axes[0].set_ylim(93.8, 97.3)
    axes[0].set_ylabel("BBBC039 test Dice (%)")
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].grid(axis="y", alpha=.25)
    axes[0].text(.01, .97, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")

    tlabels = ["B7", "Gabor", "G+Bd", "SCSE", "RG", "RS", "RG+B"]
    gray = [100 * x["explicit_grayscale"]["mean"] for x in transfer]
    inverted = [100 * x["inverted_grayscale"]["mean"] for x in transfer]
    width = .38
    x = np.arange(len(tlabels))
    axes[1].bar(x - width / 2, gray, width, color="#bab0ac", label="grayscale")
    axes[1].bar(x + width / 2, inverted, width, color="#59a14f", label="inverted")
    axes[1].set_ylim(0, 58)
    axes[1].set_ylabel("TNBC patient-macro Dice (%)")
    axes[1].set_xticks(x, tlabels, rotation=55, ha="right")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].text(.98, .97, "(b)", transform=axes[1].transAxes, va="top", ha="right", fontweight="bold")
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=.4, w_pad=.8)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"submission_results.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def polarity_figure(data):
    cases = load_external(data, "tnbc")
    run = "vanilla_b7_s1"
    gray_rows = {r["id"]: r for r in csv.DictReader(open(ROOT / "results/grayscale_transfer_audit" / run / "explicit_grayscale/per_image.csv"))}
    inv_rows = {r["id"]: r for r in csv.DictReader(open(ROOT / "results/grayscale_transfer_audit" / run / "inverted_grayscale/per_image.csv"))}
    ranked = sorted(cases, key=lambda c: float(inv_rows[c["id"]]["dice"]) - float(gray_rows[c["id"]]["dice"]))
    case = ranked[len(ranked) // 2]
    checkpoint = ROOT / "checkpoints/public_study" / run / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Net(state["variant"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.cuda().eval()
    predictions = []
    for inverted in (False, True):
        prob = model(input_tensor(case["image"], model, inverted=inverted)[None].cuda()).sigmoid().float().cpu().numpy()[0, 0]
        predictions.append(cv2.resize(prob, (case["mask"].shape[1], case["mask"].shape[0]), interpolation=cv2.INTER_LINEAR) > .5)
    gray = explicit_gray(case["image"])
    panels = [case["image"], gray, np.iinfo(gray.dtype).max - gray, case["mask"], predictions[0], predictions[1]]
    titles = ["H&E RGB", "Explicit gray", "Polarity inverted", "Reference", "Gray prediction", "Inverted prediction"]
    fig, axes = plt.subplots(1, 6, figsize=(7.1, 1.35))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=.15, w_pad=.15)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"tnbc_polarity_example.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (OUT / "tnbc_polarity_example.json").write_text(json.dumps({
        "selection": "middle-ranked per-image Dice improvement for validation-selected vanilla seed 1",
        "id": case["id"], "group": case["group"],
        "gray_dice": float(gray_rows[case["id"]]["dice"]),
        "inverted_dice": float(inv_rows[case["id"]]["dice"]),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
    }, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_figure()
    polarity_figure(ROOT / "data")


if __name__ == "__main__":
    main()
