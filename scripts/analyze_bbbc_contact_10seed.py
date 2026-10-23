#!/usr/bin/env python3
"""Aggregate the locked BBBC039 contact-aware study and paired image tests."""
import csv
import json
from pathlib import Path

import numpy as np

from breastseg.statistics import bootstrap_ci, holm, paired


ROOT = Path(__file__).resolve().parents[1]
CONTACT = ROOT / "results/bbbc_contact_10seed"
SEMANTIC_SOURCES = {
    "vanilla": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
    "gabor": [("public_study", range(3)), ("bbbc_primary_10seed_additional", range(3, 10))],
}
CONTACT_MODELS = {"vanilla": "vanilla_contact_b7", "gabor": "gabor_contact_b7"}
SEMANTIC_MODELS = {"vanilla": "vanilla_b7", "gabor": "gabor_b7"}
METRICS = ("dice1", "aji", "pq", "caicedo_af1", "dq", "sq", "symmetric_best_dice")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def seed_matrix(rows_by_seed, metric):
    ids = sorted(row["id"] for row in rows_by_seed[0])
    matrix = np.empty((10, len(ids)), dtype=float)
    for seed in range(10):
        indexed = {row["id"]: row for row in rows_by_seed[seed]}
        if sorted(indexed) != ids:
            raise RuntimeError(f"Prediction IDs differ for seed {seed}")
        matrix[seed] = [float(indexed[key][metric]) for key in ids]
    return ids, matrix


def load_contact(kind):
    model = CONTACT_MODELS[kind]
    return {
        seed: read_rows(CONTACT / f"{model}_s{seed}" / "test_per_image.csv")
        for seed in range(10)
    }


def load_semantic(kind):
    model = SEMANTIC_MODELS[kind]
    result = {}
    for study, seeds in SEMANTIC_SOURCES[kind]:
        for seed in seeds:
            result[seed] = read_rows(
                ROOT / "results" / study / f"{model}_s{seed}" / "published_metrics_components.csv")
    return result


def summarize(name, rows_by_seed):
    output = {"model": name, "n_seeds": 10, "n_images": 50}
    matrices = {}
    for metric in METRICS:
        _, matrix = seed_matrix(rows_by_seed, metric)
        matrices[metric] = matrix
        seed_means = matrix.mean(axis=1)
        image_means = matrix.mean(axis=0)
        output[metric] = float(seed_means.mean())
        output[f"{metric}_seed_sd"] = float(seed_means.std(ddof=1))
        output[f"{metric}_ci95"] = bootstrap_ci(image_means)
    return output, matrices


def comparison(a_name, a, b_name, b, metrics):
    rows = []
    for metric in metrics:
        result = paired(a[metric].mean(axis=0), b[metric].mean(axis=0))
        rows.append({"a": a_name, "b": b_name, "metric": metric, **result})
    adjusted = holm([row["p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row[f"p_holm_{len(rows)}"] = value
    return rows


def main():
    summaries, matrices = [], {}
    for kind in ("vanilla", "gabor"):
        summary, values = summarize(f"{kind}_contact_b7", load_contact(kind))
        summaries.append(summary)
        matrices[f"{kind}_contact"] = values
        _, values = summarize(f"{kind}_semantic_cc", load_semantic(kind))
        matrices[f"{kind}_semantic"] = values
    primary_metrics = ("dice1", "aji", "pq", "caicedo_af1")
    output = {
        "metric_contract": "Image-macro official BBBC039 test; mean and sample SD over ten seeds; 95% CIs bootstrap the 50 seed-averaged images.",
        "models": summaries,
        "contact_vs_matched_semantic": {
            kind: comparison(
                f"{kind}_contact", matrices[f"{kind}_contact"],
                f"{kind}_semantic", matrices[f"{kind}_semantic"], primary_metrics)
            for kind in ("vanilla", "gabor")
        },
        "gabor_contact_vs_vanilla_contact": comparison(
            "gabor_contact", matrices["gabor_contact"],
            "vanilla_contact", matrices["vanilla_contact"], primary_metrics),
    }
    (CONTACT / "aggregate.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
