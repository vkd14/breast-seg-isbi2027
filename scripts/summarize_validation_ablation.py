#!/usr/bin/env python3
"""Summarize the frozen validation-only study without reading the test partition."""
import csv
import json
from pathlib import Path

import numpy as np

from breastseg.core import sha256
from breastseg.statistics import holm, paired

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/public_validation_ablation"
CONFIG = ROOT / "configs/public_validation_ablation.json"
METRICS = ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px")
ROW_FIELDS = METRICS + ("empty_prediction", "empty_reference", "fp")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run(condition, seed, protocol_hash):
    run = STUDY / f"{condition}_s{seed}"
    summary = json.loads((run / "summary.json").read_text())
    if summary["protocol_sha256"] != protocol_hash or summary["test_images_accessed"] != 0:
        raise RuntimeError(f"Protocol or test-access violation in {run.name}")
    with (run / "validation_per_image.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or len({r["id"] for r in rows}) != 50:
        raise RuntimeError(f"Incomplete validation predictions in {run.name}")
    return summary, {r["id"]: {m: float(r[m]) for m in ROW_FIELDS} for r in rows}


def main():
    cfg = json.loads(CONFIG.read_text())
    protocol = json.loads((STUDY / "protocol.json").read_text())
    protocol_hash = protocol["protocol_sha256"]
    for relative, expected in protocol["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Active code changed since execution: {relative}")
    for filename, expected in protocol["archive_sha256"].items():
        if sha256(ROOT / "data/bbbc039" / filename) != expected:
            raise RuntimeError(f"Public archive changed since execution: {filename}")
    names = [c["name"] for c in cfg["conditions"]]
    summaries, values = {}, {}
    for name in names:
        summaries[name], values[name] = [], []
        for seed in cfg["seeds"]:
            summary, per_image = load_run(name, seed, protocol_hash)
            summaries[name].append(summary)
            values[name].append(per_image)

    table = []
    for name in names:
        row = {"condition": name, "seeds": len(cfg["seeds"]), "validation_images": 50}
        selected = [i for i in values[name][0]
                    if values[name][0][i]["empty_reference"] == 0]
        foreground_seed_means = [np.mean([seed_values[i]["dice"] for i in selected])
                                 for seed_values in values[name]]
        row["foreground_dice_mean"] = float(np.mean(foreground_seed_means))
        row["foreground_dice_seed_sd"] = float(np.std(foreground_seed_means, ddof=1))
        for metric in METRICS:
            seed_means = [s["mean"][metric] for s in summaries[name]]
            row[f"allcase_{metric}_mean"] = float(np.mean(seed_means))
            row[f"allcase_{metric}_seed_sd"] = float(np.std(seed_means, ddof=1))
        row["mean_best_epoch"] = float(np.mean([s["best_epoch"] for s in summaries[name]]))
        table.append(row)
    write_csv(STUDY / "summary.csv", table)

    strata = []
    for name in names:
        identifiers = sorted(values[name][0])
        foreground = [i for i in identifiers if values[name][0][i]["empty_reference"] == 0]
        empty = [i for i in identifiers if values[name][0][i]["empty_reference"] == 1]
        for label, selected in (("foreground_reference", foreground), ("empty_reference", empty)):
            row = {"condition": name, "stratum": label, "n_images": len(selected),
                   "seeds": len(cfg["seeds"])}
            for metric in METRICS:
                row[f"{metric}_mean"] = float(np.mean([
                    seed_values[i][metric] for seed_values in values[name] for i in selected]))
            row["empty_prediction_rate"] = float(np.mean([
                seed_values[i]["empty_prediction"] for seed_values in values[name] for i in selected]))
            row["false_positive_pixels_mean"] = float(np.mean([
                seed_values[i]["fp"] for seed_values in values[name] for i in selected]))
            strata.append(row)
    write_csv(STUDY / "validation_strata.csv", strata)

    identifiers = sorted(values["gabor_reference"][0])
    foreground_identifiers = [i for i in identifiers
                              if values["gabor_reference"][0][i]["empty_reference"] == 0]
    averaged = {}
    for name in names:
        if any(sorted(seed_values) != identifiers for seed_values in values[name]):
            raise RuntimeError(f"Validation identifiers differ for {name}")
        averaged[name] = {
            metric: np.array([np.mean([seed_values[i][metric] for seed_values in values[name]])
                              for i in foreground_identifiers])
            for metric in METRICS
        }

    comparisons = []
    for name in names:
        if name == "gabor_reference":
            continue
        row = {"condition": name, "reference": "gabor_reference", "paired_units": 49,
               "unit": "foreground-containing validation image after averaging seeds"}
        for metric in METRICS:
            result = paired(averaged[name][metric], averaged["gabor_reference"][metric])
            row[f"{metric}_difference"] = result["mean_difference"]
            row[f"{metric}_ci95_low"] = result["ci95"][0]
            row[f"{metric}_ci95_high"] = result["ci95"][1]
            row[f"{metric}_p_raw"] = result["p"]
        comparisons.append(row)
    for metric in METRICS:
        adjusted = holm([row[f"{metric}_p_raw"] for row in comparisons])
        for row, value in zip(comparisons, adjusted):
            row[f"{metric}_p_holm"] = value
    write_csv(STUDY / "paired_vs_gabor_reference.csv", comparisons)
    initial_hashes = {
        str(seed): sorted({s["initial_state_sha256"] for name in names
                           for s in summaries[name] if s["seed"] == seed})
        for seed in cfg["seeds"]
    }
    if any(len(hashes) != 1 for hashes in initial_hashes.values()):
        raise RuntimeError("Initial model states differ across conditions within a seed")
    audit = {"status": "PASS", "protocol_sha256": protocol_hash,
             "completed_runs": sum(len(v) for v in summaries.values()),
             "expected_runs": len(names) * len(cfg["seeds"]),
             "validation_rows": sum(len(seed_values) for condition in values.values()
                                    for seed_values in condition),
             "foreground_units": len(foreground_identifiers), "empty_units": 1,
             "test_images_accessed": 0, "active_code_and_archive_hashes_match": True,
             "identical_initial_state_within_seed": True, "initial_state_sha256": initial_hashes}
    (STUDY / "validation.json").write_text(json.dumps(audit, indent=2))
    print(f"Wrote summary, foreground/empty strata, and paired comparisons; test partition was not read")


if __name__ == "__main__":
    main()
