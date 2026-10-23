#!/usr/bin/env python3
"""Aggregate paper-format instance metrics across the locked ten-seed studies."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from breastseg.statistics import bootstrap_ci, holm, paired


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/published_metrics_10seed"
METRICS = (
    "dice1", "dice2", "aji", "dq", "sq", "pq", "caicedo_af1", "symmetric_best_dice"
)
BBBC_SOURCES = {
    "vanilla_b7": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
    "gabor_b7": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
    "unetpp_xception_scse": [
        ("public_architecture_baselines", range(3)),
        ("bbbc_xception_10seed_additional", range(3, 10)),
    ],
}
TNBC_STUDIES = {
    "grayscale": {
        "models": ("vanilla_b7", "gabor_b7", "unetpp_xception_scse"),
        "sources": [("tnbc_grouped_cv", range(3)), ("tnbc_grouped_cv_10seed_additional", range(3, 10))],
    },
    "rgb": {
        "models": ("vanilla_b7_rgb", "gabor_b7_rgb"),
        "sources": [("tnbc_rgb_grouped_cv_10seed", range(10))],
    },
}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_seed_rows(model, rows_by_seed, unit_key):
    if sorted(rows_by_seed) != list(range(10)):
        raise RuntimeError(f"Expected seeds 0--9 for {model}; found {sorted(rows_by_seed)}")
    reference_ids = sorted(row["id"] for row in rows_by_seed[0])
    for seed, rows in rows_by_seed.items():
        if sorted(row["id"] for row in rows) != reference_ids:
            raise RuntimeError(f"Prediction IDs differ for {model}, seed {seed}")

    units = sorted({row[unit_key] for row in rows_by_seed[0]}) if unit_key else reference_ids
    values = {metric: np.zeros((10, len(units)), dtype=float) for metric in METRICS}
    for seed, rows in rows_by_seed.items():
        for metric in METRICS:
            for index, unit in enumerate(units):
                selected = [float(row[metric]) for row in rows if (row[unit_key] if unit_key else row["id"]) == unit]
                if not selected:
                    raise RuntimeError(f"Missing {unit_key or 'image'} {unit} for {model}, seed {seed}")
                values[metric][seed, index] = np.mean(selected)

    record = {"model": model, "n_seeds": 10, "n_units": len(units)}
    for metric, matrix in values.items():
        seed_means = matrix.mean(axis=1)
        unit_means = matrix.mean(axis=0)
        record[metric] = float(seed_means.mean())
        record[f"{metric}_seed_sd"] = float(seed_means.std(ddof=1))
        record[f"{metric}_ci95"] = bootstrap_ci(unit_means)
    return record, values


def bbbc_summary():
    records, matrices = [], {}
    for model, sources in BBBC_SOURCES.items():
        rows_by_seed = {}
        for study, seeds in sources:
            for seed in seeds:
                path = ROOT / "results" / study / f"{model}_s{seed}" / "published_metrics_components.csv"
                rows = read_rows(path)
                if len(rows) != 50:
                    raise RuntimeError(f"Expected 50 rows in {path}")
                rows_by_seed[seed] = rows
        record, values = aggregate_seed_rows(model, rows_by_seed, unit_key=None)
        record.update({"dataset": "BBBC039", "averaging_unit": "image", "postprocessing": "components"})
        records.append(record)
        matrices[model] = values

    comparisons = []
    for baseline in ("vanilla_b7", "unetpp_xception_scse"):
        for metric in ("aji", "pq", "caicedo_af1"):
            result = paired(matrices["gabor_b7"][metric].mean(0), matrices[baseline][metric].mean(0))
            comparisons.append({"a": "gabor_b7", "b": baseline, "metric": metric, **result})
    for item, adjusted in zip(comparisons, holm([item["p"] for item in comparisons])):
        item["p_holm_6"] = adjusted
    return records, comparisons


def tnbc_summary():
    output = {}
    for mode, specification in TNBC_STUDIES.items():
        records, matrices = [], {}
        for model in specification["models"]:
            rows_by_seed = {}
            for study, seeds in specification["sources"]:
                for seed in seeds:
                    path = ROOT / "results" / study / "published_metrics" / f"{model}_s{seed}_components.csv"
                    rows = read_rows(path)
                    if len(rows) != 50:
                        raise RuntimeError(f"Expected 50 rows in {path}")
                    rows_by_seed[seed] = rows
            record, values = aggregate_seed_rows(model, rows_by_seed, unit_key="patient")
            record.update({"dataset": "TNBC", "input_mode": mode, "averaging_unit": "patient"})
            records.append(record)
            matrices[model] = values
        comparisons = []
        if mode == "grayscale":
            anchor, baselines = "gabor_b7", ("vanilla_b7", "unetpp_xception_scse")
        else:
            anchor, baselines = "gabor_b7_rgb", ("vanilla_b7_rgb",)
        for baseline in baselines:
            for metric in ("dice1", "aji", "pq"):
                result = paired(matrices[anchor][metric].mean(0), matrices[baseline][metric].mean(0))
                comparisons.append({"a": anchor, "b": baseline, "metric": metric, **result})
        for item, adjusted in zip(comparisons, holm([item["p"] for item in comparisons])):
            item[f"p_holm_{len(comparisons)}"] = adjusted
        output[mode] = {"models": records, "comparisons": comparisons}
    return output


def instance_model_summary():
    records = []
    study = ROOT / "results/bbbc_instance_10seed"
    for model in ("vanilla_instance_b7", "gabor_instance_b7"):
        summaries = []
        for seed in range(10):
            path = study / f"{model}_s{seed}" / "summary.json"
            summaries.append(json.loads(path.read_text()))
        record = {"model": model, "dataset": "BBBC039", "n_seeds": 10}
        for output_name in ("interior_mean", "expanded_mean"):
            for metric in METRICS:
                values = [float(row[output_name][metric]) for row in summaries]
                record[f"{output_name}_{metric}"] = float(np.mean(values))
                record[f"{output_name}_{metric}_seed_sd"] = float(np.std(values, ddof=1))
        records.append(record)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bbbc, bbbc_comparisons = bbbc_summary()
    result = {
        "metric_contract": {
            "bbbc039": "official 100/50/50 split; image-macro after averaging ten seeds",
            "tnbc": "fixed patient-disjoint five-fold OOF predictions; patient-macro after averaging ten seeds",
            "cross_paper_note": "Published point estimates can be compared only where split and metric definitions align; no cross-paper significance test is claimed.",
        },
        "bbbc039_semantic_outputs": bbbc,
        "bbbc039_semantic_comparisons": bbbc_comparisons,
        "bbbc039_three_class_outputs": instance_model_summary(),
        "tnbc": tnbc_summary(),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
