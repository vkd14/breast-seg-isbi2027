#!/usr/bin/env python3
"""Combine the original seeds 0--2 with locked extension seeds 3--9."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from breastseg.statistics import bootstrap_ci, holm, paired


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/primary_10seed"
METRICS = ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px")
BBBC_PATHS = {
    "vanilla_b7": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
    "gabor_b7": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
    "unetpp_xception_scse": [
        ("public_architecture_baselines", range(3)),
        ("bbbc_xception_10seed_additional", range(3, 10)),
    ],
}
TNBC_PATHS = [("tnbc_grouped_cv", range(3)), ("tnbc_grouped_cv_10seed_additional", range(3, 10))]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bbbc_run(model, study, seed):
    return ROOT / "results" / study / f"{model}_s{seed}"


def summarize_bbbc():
    model_seed_rows = {}
    records = []
    for model, sources in BBBC_PATHS.items():
        model_seed_rows[model] = {}
        seed_means = defaultdict(list)
        for study, seeds in sources:
            for seed in seeds:
                run = bbbc_run(model, study, seed)
                summary = json.loads((run / "summary.json").read_text())
                rows = read_csv(run / "test_per_image.csv")
                if len(rows) != 50 or len({row["id"] for row in rows}) != 50:
                    raise RuntimeError(f"Incomplete BBBC039 run: {run}")
                model_seed_rows[model][seed] = rows
                for metric in METRICS:
                    seed_means[metric].append(float(summary["mean"][metric]))
        if sorted(model_seed_rows[model]) != list(range(10)):
            raise RuntimeError(f"Expected seeds 0--9 for {model}")
        by_image = {
            metric: {
                image_id: float(np.mean([
                    float(next(row[metric] for row in model_seed_rows[model][seed] if row["id"] == image_id))
                    for seed in range(10)
                ]))
                for image_id in sorted(row["id"] for row in model_seed_rows[model][0])
            }
            for metric in METRICS
        }
        record = {"model": model, "n_images": 50, "n_seeds": 10}
        for metric in METRICS:
            values = list(by_image[metric].values())
            record[metric] = float(np.mean(seed_means[metric]))
            record[f"{metric}_seed_sd"] = float(np.std(seed_means[metric], ddof=1))
            record[f"{metric}_ci95"] = bootstrap_ci(values)
        records.append(record)
        model_seed_rows[model]["by_image"] = by_image

    comparisons = []
    for baseline in ("vanilla_b7", "unetpp_xception_scse"):
        for metric in ("dice", "boundary_f1_2px"):
            ids = sorted(model_seed_rows["gabor_b7"]["by_image"][metric])
            result = paired(
                [model_seed_rows["gabor_b7"]["by_image"][metric][key] for key in ids],
                [model_seed_rows[baseline]["by_image"][metric][key] for key in ids],
            )
            comparisons.append({"a": "gabor_b7", "b": baseline, "metric": metric, **result})
    for comparison, adjusted in zip(comparisons, holm([item["p"] for item in comparisons])):
        comparison["p_holm_4"] = adjusted
    return records, comparisons


def summarize_tnbc():
    config = json.loads((ROOT / "configs/tnbc_grouped_cv_10seed_additional.json").read_text())
    models, patients = config["models"], sorted(set().union(*map(set, config["outer_folds"])))
    values, records = {}, []
    for model in models:
        values[model] = {metric: np.zeros((10, len(patients))) for metric in METRICS}
        for study, seeds in TNBC_PATHS:
            for seed in seeds:
                rows = []
                for fold in range(len(config["outer_folds"])):
                    run = ROOT / "results" / study / f"{model}_f{fold}_s{seed}"
                    rows.extend(read_csv(run / "test_per_image.csv"))
                if len(rows) != 50 or len({row["id"] for row in rows}) != 50:
                    raise RuntimeError(f"Incomplete TNBC OOF result: {model}, seed {seed}")
                for metric in METRICS:
                    for index, patient in enumerate(patients):
                        values[model][metric][seed, index] = np.mean([
                            float(row[metric]) for row in rows if row["group"] == patient
                        ])
        record = {"model": model, "n_patients": len(patients), "n_images": 50, "n_seeds": 10}
        for metric in METRICS:
            patient_mean = values[model][metric].mean(axis=0)
            record[metric] = float(values[model][metric].mean())
            record[f"{metric}_seed_sd"] = float(values[model][metric].mean(axis=1).std(ddof=1))
            record[f"{metric}_ci95"] = bootstrap_ci(patient_mean)
        records.append(record)
    comparisons = []
    for baseline in ("vanilla_b7", "unetpp_xception_scse"):
        for metric in ("dice", "boundary_f1_2px"):
            result = paired(values["gabor_b7"][metric].mean(axis=0), values[baseline][metric].mean(axis=0))
            comparisons.append({"a": "gabor_b7", "b": baseline, "metric": metric, **result})
    for comparison, adjusted in zip(comparisons, holm([item["p"] for item in comparisons])):
        comparison["p_holm_4"] = adjusted
    return records, comparisons


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bbbc, bbbc_comparisons = summarize_bbbc()
    tnbc, tnbc_comparisons = summarize_tnbc()
    output = {
        "bbbc039": bbbc,
        "bbbc039_comparisons": bbbc_comparisons,
        "tnbc": tnbc,
        "tnbc_comparisons": tnbc_comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
