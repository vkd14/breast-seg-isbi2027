#!/usr/bin/env python3
"""Aggregate repeated runs using source groups as the independent units.

Slices from one z-stack/source group are correlated.  Consequently, intervals
and tests over individual slices are pseudoreplicated.  This analysis first
averages each metric within source group, then averages the three training
seeds for each configuration.  Confidence intervals resample source groups;
paired comparisons use source-group means and are Holm-corrected across all
reported hold-out comparisons.

The repository does not retain acquisition identifiers for every image.  The
``volume`` field therefore denotes the conservative shape-defined source
groups in ``volume_split.json``; the manuscript must not call all of them
documented acquisition fields.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "revision")
DEST = os.path.join(OUT, "stats_clustered")
os.makedirs(DEST, exist_ok=True)

METRICS = ("dice", "iou", "precision", "recall", "hd95_um", "assd_um", "nsd2")
SEED_SUFFIX = re.compile(r"_s([012])$")


def base_name(name: str) -> str:
    if name.startswith("proposed_s"):
        return "proposed"
    return SEED_SUFFIX.sub("", name)


def read_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUT, "runs", "*", "per_image.csv"))):
        result_path = os.path.join(os.path.dirname(path), "result.json")
        if not os.path.exists(result_path):
            continue
        result = json.load(open(result_path))
        rows = list(csv.DictReader(open(path)))
        # Hold-out experiments have the same 206-image, seven-group test set.
        if len(rows) == 206:
            runs[result["name"]] = {"rows": rows, "result": result}
    specialist = os.path.join(OUT, "specialist_baselines.json")
    if os.path.exists(specialist):
        for name, value in json.load(open(specialist)).items():
            runs[name] = {"rows": value["per_image"], "result": {"test": value["test"]}}
    return runs


def group_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["volume"]].append(row)
    return {
        volume: {metric: float(np.mean([float(row[metric]) for row in values])) for metric in METRICS}
        for volume, values in sorted(grouped.items())
    }


def percentile_ci(values, n_resamples=100_000, seed=20260916):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    boot = values[indices].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_rank_biserial(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if not len(diff):
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return float((pos - neg) / (pos + neg))


def holm_adjust(tests):
    order = np.argsort([test["p_raw"] for test in tests])
    adjusted = np.empty(len(tests), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, tests[index]["p_raw"] * (len(tests) - rank))
        adjusted[index] = min(1.0, running)
    for test, value in zip(tests, adjusted):
        test["p_holm"] = float(value)


def main():
    raw = read_runs()
    grouped_runs = {name: group_means(value["rows"]) for name, value in raw.items()}
    families = defaultdict(list)
    for name in grouped_runs:
        families[base_name(name)].append(name)

    aggregate = {}
    for family, names in sorted(families.items()):
        # Specialist baselines are deterministic single runs.  All learned
        # comparison configurations should have seeds 0, 1 and 2 after queue completion.
        volumes = sorted(set.intersection(*(set(grouped_runs[name]) for name in names)))
        per_volume = {}
        for volume in volumes:
            per_volume[volume] = {
                metric: float(np.mean([grouped_runs[name][volume][metric] for name in names]))
                for metric in METRICS
            }
        summary = {"n_groups": len(volumes), "n_seeds": len(names), "seeds": sorted(names), "per_group": per_volume}
        for metric in METRICS:
            values = [per_volume[volume][metric] for volume in volumes]
            lo, hi = percentile_ci(values)
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_lo"] = lo
            summary[f"{metric}_hi"] = hi
            seed_values = [np.mean([grouped_runs[name][volume][metric] for volume in volumes]) for name in names]
            summary[f"{metric}_seed_sd"] = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else None
        aggregate[family] = summary

    reference = aggregate["proposed"]
    tests = []
    for family, summary in aggregate.items():
        if family == "proposed":
            continue
        volumes = sorted(set(reference["per_group"]) & set(summary["per_group"]))
        if len(volumes) != reference["n_groups"]:
            continue
        ref = np.array([reference["per_group"][volume]["dice"] for volume in volumes])
        other = np.array([summary["per_group"][volume]["dice"] for volume in volumes])
        diff = ref - other
        if np.allclose(diff, 0):
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = stats.wilcoxon(ref, other, zero_method="wilcox", method="auto")
        lo, hi = percentile_ci(diff)
        tests.append({
            "comparison": f"proposed vs {family}",
            "n_groups": len(volumes),
            "mean_difference": float(np.mean(diff)),
            "difference_lo": lo,
            "difference_hi": hi,
            "median_difference": float(np.median(diff)),
            "wilcoxon_W": float(statistic),
            "p_raw": float(pvalue),
            "paired_rank_biserial": paired_rank_biserial(diff),
        })
    holm_adjust(tests)

    with open(os.path.join(DEST, "numbers.json"), "w") as handle:
        json.dump({"summary": aggregate, "tests": tests}, handle, indent=2)

    summary_columns = ["name", "n_groups", "n_seeds"]
    for metric in METRICS:
        summary_columns.extend((metric, f"{metric}_lo", f"{metric}_hi", f"{metric}_seed_sd"))
    with open(os.path.join(DEST, "summary.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, extrasaction="ignore")
        writer.writeheader()
        for name, values in aggregate.items():
            writer.writerow({"name": name, **values})

    with open(os.path.join(DEST, "tests.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tests[0]) if tests else ["comparison"])
        writer.writeheader()
        writer.writerows(tests)

    print("configuration                 seeds  Dice [group-bootstrap 95% CI]   seed SD")
    for name, values in aggregate.items():
        seed_sd = values["dice_seed_sd"]
        seed_text = "--" if seed_sd is None else f"{seed_sd:.4f}"
        print(f"{name:30s} {values['n_seeds']:>2d}    {values['dice']:.4f} "
              f"[{values['dice_lo']:.4f}, {values['dice_hi']:.4f}]   {seed_text}")
    if tests:
        print("minimum Holm-adjusted p:", min(test["p_holm"] for test in tests))


if __name__ == "__main__":
    main()
