#!/usr/bin/env python3
"""Aggregate the matched-budget architecture baseline study."""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from breastseg.statistics import holm, paired


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "public_architecture_baselines"
LABELS = {
    "unet_resnet34": "U-Net / ResNet-34",
    "unet_resnet50": "U-Net / ResNet-50",
    "unetpp_effb0": "UNet++ / EfficientNet-B0",
    "unetpp_xception_scse": "UNet++ / Xception / SCSE",
    "deeplabv3p_resnet50": "DeepLabV3+ / ResNet-50",
    "fpn_resnet50": "FPN / ResNet-50",
}


def averaged_images(pattern):
    values = defaultdict(list)
    for path in glob.glob(str(pattern)):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                values[row["id"]].append(float(row["dice"]))
    if not values or any(len(v) != 3 for v in values.values()):
        raise RuntimeError(f"Expected three complete seeds: {pattern}")
    return {key: float(np.mean(value)) for key, value in values.items()}


def main():
    records = [json.loads(path.read_text()) for path in OUT.glob("*_s*/summary.json")]
    if len(records) != 18:
        raise RuntimeError(f"Expected 18 summaries, found {len(records)}")
    gabor = averaged_images(ROOT / "results/public_study/gabor_b7_s*/test_per_image.csv")
    rows, comparisons = [], []
    for name in LABELS:
        selected = sorted((r for r in records if r["baseline"] == name), key=lambda r: r["seed"])
        if [r["seed"] for r in selected] != [0, 1, 2]:
            raise RuntimeError(f"Incomplete seeds for {name}")
        target = averaged_images(OUT / f"{name}_s*/test_per_image.csv")
        ids = sorted(gabor)
        comparison = paired([gabor[i] for i in ids], [target[i] for i in ids])
        comparisons.append(comparison)
        rows.append({
            "model": name,
            "label": LABELS[name],
            "parameters": selected[0]["parameters"],
            "dice": float(np.mean([r["mean"]["dice"] for r in selected])),
            "dice_seed_sd": float(np.std([r["mean"]["dice"] for r in selected], ddof=1)),
            "iou": float(np.mean([r["mean"]["iou"] for r in selected])),
            "boundary_f1_2px": float(np.mean([r["mean"]["boundary_f1_2px"] for r in selected])),
            "hd95_px": float(np.mean([r["mean"]["hd95_px"] for r in selected])),
            "gabor_minus_baseline_dice": comparison["mean_difference"],
            "gabor_minus_baseline_ci95": comparison["ci95"],
            "gabor_minus_baseline_p": comparison["p"],
        })
    for row, adjusted in zip(rows, holm([x["p"] for x in comparisons])):
        row["gabor_minus_baseline_p_holm_6"] = adjusted
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    flat = []
    for row in rows:
        item = dict(row)
        item["gabor_minus_baseline_ci95_low"] = item.pop("gabor_minus_baseline_ci95")[0]
        item["gabor_minus_baseline_ci95_high"] = row["gabor_minus_baseline_ci95"][1]
        flat.append(item)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
