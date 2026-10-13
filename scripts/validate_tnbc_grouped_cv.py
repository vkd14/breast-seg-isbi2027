#!/usr/bin/env python3
"""Recompute every saved TNBC prediction metric and audit the locked grouped protocol."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from breastseg.core import metrics
from breastseg.external import load_external


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "tnbc_grouped_cv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "tnbc_grouped_cv.json")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    cases = {case["id"]: case for case in load_external(args.data, "tnbc")}
    expected_runs = 0
    checked_rows = 0
    tested = {(name, seed): [] for name in config["models"] for seed in config["seeds"]}

    for name in config["models"]:
        for seed in config["seeds"]:
            for fold, expected_groups in enumerate(config["outer_folds"]):
                expected_runs += 1
                run = args.results / f"{name}_f{fold}_s{seed}"
                summary = json.loads((run / "summary.json").read_text())
                if summary["model"] != name or summary["seed"] != seed or summary["fold"] != fold:
                    raise AssertionError(f"Run identity mismatch: {run}")
                if summary["test_patients"] != sorted(expected_groups):
                    raise AssertionError(f"Test-patient mismatch: {run}")
                train = set(summary["train_patients"])
                validation = set(summary["validation_patients"])
                test = set(summary["test_patients"])
                if train & validation or train & test or validation & test:
                    raise AssertionError(f"Patient leakage: {run}")

                with (run / "test_per_image.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if len(rows) != summary["n_test"]:
                    raise AssertionError(f"Incomplete test table: {run}")
                if {row["group"] for row in rows} != test:
                    raise AssertionError(f"CSV test groups mismatch: {run}")

                for row in rows:
                    case = cases[row["id"]]
                    if case["group"] != row["group"]:
                        raise AssertionError(f"Patient label mismatch: {run}/{row['id']}")
                    prediction_path = run / "predictions" / f"{row['id']}.png"
                    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_UNCHANGED)
                    if prediction is None:
                        raise FileNotFoundError(prediction_path)
                    actual = metrics(prediction > 0, case["mask"])
                    for key in METRICS:
                        if not np.isclose(actual[key], float(row[key]), rtol=0, atol=1e-12):
                            raise AssertionError(
                                f"Metric mismatch {run.name}/{row['id']} {key}: "
                                f"{actual[key]} != {row[key]}"
                            )
                    checked_rows += 1
                    tested[(name, seed)].append(row["id"])

    expected_ids = sorted(cases)
    for unit, ids in tested.items():
        if sorted(ids) != expected_ids or len(ids) != len(set(ids)):
            raise AssertionError(f"OOF coverage mismatch for {unit}")

    print(
        f"PASS: {expected_runs} patient-disjoint runs, {checked_rows} native-grid "
        f"predictions, and complete 50-image OOF coverage for every model/seed."
    )


if __name__ == "__main__":
    main()
