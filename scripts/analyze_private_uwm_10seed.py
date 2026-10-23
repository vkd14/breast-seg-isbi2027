#!/usr/bin/env python3
"""Aggregate UWM primary/control runs with source group as the sampling unit."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/private_uwm_10seed"
METRICS = ("dice", "iou", "precision", "recall", "hd95_um", "assd_um", "nsd2")


def run_name(variant: str, seed: int) -> str:
    if variant == "vanilla_effb7_unetpp" and seed == 0:
        return variant
    return f"{variant}_s{seed}"


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["volume"]].append(row)
    return {
        group: {metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in METRICS}
        for group, values in sorted(buckets.items())
    }


def bootstrap_ci(values, seed=20261023, samples=100_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def summarize_variant(runs: Path, variant: str):
    seeds = {}
    identifiers = None
    for seed in range(10):
        directory = runs / run_name(variant, seed)
        result = json.loads((directory / "result.json").read_text())
        rows = list(csv.DictReader((directory / "per_image.csv").open()))
        if result["config"]["seed"] != seed or len(rows) != 206:
            raise ValueError(f"Incomplete or mismatched run: {directory}")
        current = {(row["volume"], row["file"]) for row in rows}
        if identifiers is None:
            identifiers = current
        elif current != identifiers:
            raise ValueError(f"Test membership changed in {directory}")
        seeds[seed] = grouped(rows)
    groups = sorted(set.intersection(*(set(value) for value in seeds.values())))
    if len(groups) != 7:
        raise ValueError(f"Expected seven held-out groups, found {len(groups)}")
    per_group = {
        group: {metric: float(np.mean([seeds[seed][group][metric] for seed in seeds]))
                for metric in METRICS}
        for group in groups
    }
    summary = {"n_seeds": 10, "n_groups": 7, "n_images": 206,
               "run_names": [run_name(variant, seed) for seed in range(10)],
               "per_group": per_group}
    for metric in METRICS:
        group_values = [per_group[group][metric] for group in groups]
        seed_values = [np.mean([seeds[seed][group][metric] for group in groups])
                       for seed in seeds]
        summary[metric] = float(np.mean(group_values))
        summary[f"{metric}_group_bootstrap_ci"] = bootstrap_ci(group_values)
        summary[f"{metric}_seed_sd"] = float(np.std(seed_values, ddof=1))
        summary[f"{metric}_per_seed"] = [float(value) for value in seed_values]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    runs = args.private_root.expanduser().resolve() / "outputs/revision/runs"
    summaries = {
        variant: summarize_variant(runs, variant)
        for variant in ("proposed", "vanilla_effb7_unetpp")
    }
    proposed = summaries["proposed"]
    control = summaries["vanilla_effb7_unetpp"]
    groups = sorted(proposed["per_group"])
    differences = np.asarray([
        proposed["per_group"][group]["dice"] - control["per_group"][group]["dice"]
        for group in groups
    ])
    statistic, p_value = stats.wilcoxon(differences, zero_method="wilcox", method="auto")
    comparison = {
        "endpoint": "group-macro Dice",
        "difference": float(differences.mean()),
        "group_bootstrap_ci": bootstrap_ci(differences, seed=20261024),
        "wilcoxon_W": float(statistic),
        "p_value": float(p_value),
        "n_independent_groups": len(groups),
        "interpretation": "proposed minus matched vanilla control",
    }
    payload = {
        "protocol": {
            "seeds": list(range(10)), "split": "fixed 516/133/206 image group split",
            "independent_test_units": "seven conservative source groups",
            "tta": False, "checkpoint_selection": "highest validation Dice",
        },
        "models": summaries,
        "paired_comparison": comparison,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "dice", "seed_sd", "ci_low", "ci_high", "n_seeds", "n_groups"))
        for name, summary in summaries.items():
            ci = summary["dice_group_bootstrap_ci"]
            writer.writerow((name, summary["dice"], summary["dice_seed_sd"], ci[0], ci[1], 10, 7))
    for name, summary in summaries.items():
        ci = summary["dice_group_bootstrap_ci"]
        print(f"{name}: Dice {summary['dice']:.4f} (seed SD {summary['dice_seed_sd']:.4f}), "
              f"group-bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    ci = comparison["group_bootstrap_ci"]
    print(f"difference: {comparison['difference']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], "
          f"paired group Wilcoxon p={comparison['p_value']:.4g}")


if __name__ == "__main__":
    main()
