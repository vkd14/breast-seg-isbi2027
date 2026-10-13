#!/usr/bin/env python3
"""Summarize patient-macro TNBC out-of-fold results and paired model comparisons."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from breastseg.statistics import bootstrap_ci, holm, paired


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tnbc_grouped_cv"
CONFIG = ROOT / "configs" / "tnbc_grouped_cv.json"
METRICS = ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px")
LABELS = {
    "vanilla_b7": "Vanilla EfficientNet-B7/UNet++",
    "gabor_b7": "Gabor/projection/SCSE EfficientNet-B7/UNet++",
    "unetpp_xception_scse": "UNet++/Xception/SCSE",
}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    config = json.loads(CONFIG.read_text())
    models = list(config["models"])
    seeds = config["seeds"]
    folds = range(len(config["outer_folds"]))
    expected = len(models) * len(seeds) * len(config["outer_folds"])
    summaries = list(OUT.glob("*_f*_s*/summary.json"))
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} completed runs, found {len(summaries)}")

    oof = {}
    combined = []
    for model in models:
        oof[model] = {}
        for seed in seeds:
            rows = []
            for fold in folds:
                run = OUT / f"{model}_f{fold}_s{seed}"
                summary = json.loads((run / "summary.json").read_text())
                if summary["model"] != model or summary["seed"] != seed or summary["fold"] != fold:
                    raise RuntimeError(f"Run identity mismatch: {run}")
                for row in read_rows(run / "test_per_image.csv"):
                    rows.append({
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "id": row["id"],
                        "patient": row["group"],
                        **{metric: float(row[metric]) for metric in METRICS},
                    })
            if len(rows) != 50 or len({row["id"] for row in rows}) != 50:
                raise RuntimeError(f"Expected exactly 50 out-of-fold images: {model}, seed {seed}")
            if {row["patient"] for row in rows} != set().union(*map(set, config["outer_folds"])):
                raise RuntimeError("Missing TNBC patient")
            oof[model][seed] = rows
            combined.extend(rows)

    with (OUT / "oof_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(combined)

    model_records = []
    patient_records = []
    patient_values = {}
    for model in models:
        patients = sorted({row["patient"] for row in oof[model][seeds[0]]})
        per_seed_patient = {metric: [] for metric in METRICS}
        for seed in seeds:
            for metric in METRICS:
                per_seed_patient[metric].append([
                    float(np.mean([
                        row[metric] for row in oof[model][seed] if row["patient"] == patient
                    ]))
                    for patient in patients
                ])
        arrays = {metric: np.asarray(value) for metric, value in per_seed_patient.items()}
        patient_values[model] = {metric: value.mean(axis=0) for metric, value in arrays.items()}
        record = {
            "model": model,
            "label": LABELS[model],
            "n_patients": len(patients),
            "n_images": 50,
            "n_seeds": len(seeds),
        }
        for metric, value in arrays.items():
            averaged_patients = value.mean(axis=0)
            interval = bootstrap_ci(averaged_patients)
            record[metric] = float(value.mean())
            record[f"{metric}_seed_sd"] = float(value.mean(axis=1).std(ddof=1))
            record[f"{metric}_ci95"] = [float(interval[0]), float(interval[1])]
        model_records.append(record)
        for patient_index, patient in enumerate(patients):
            patient_records.append({
                "model": model,
                "patient": patient,
                **{
                    metric: float(patient_values[model][metric][patient_index])
                    for metric in METRICS
                },
            })

    comparisons = []
    for baseline in ("vanilla_b7", "unetpp_xception_scse"):
        for metric in ("dice", "boundary_f1_2px"):
            result = paired(
                patient_values["gabor_b7"][metric],
                patient_values[baseline][metric],
            )
            comparisons.append({
                "a": "gabor_b7",
                "b": baseline,
                "metric": metric,
                "n_patients": 11,
                "mean_difference": result["mean_difference"],
                "ci95": result["ci95"],
                "p": result["p"],
            })
    adjusted = holm([record["p"] for record in comparisons])
    for record, value in zip(comparisons, adjusted):
        record["p_holm_4"] = value

    (OUT / "summary.json").write_text(json.dumps(model_records, indent=2))
    (OUT / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))
    with (OUT / "patient_macro.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(patient_records)
    flat = []
    for record in model_records:
        row = dict(record)
        for metric in METRICS:
            low, high = row.pop(f"{metric}_ci95")
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        flat.append(row)
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"models": model_records, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
