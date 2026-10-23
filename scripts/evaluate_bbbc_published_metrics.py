#!/usr/bin/env python3
"""Evaluate saved BBBC039 predictions in published paper metric formats."""
import argparse
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from breastseg.instance_metrics import (
    bbbc039_reference_instances,
    published_metrics,
    semantic_to_instances,
)


ROOT = Path(__file__).resolve().parents[1]


def load_raw_annotations(data):
    with zipfile.ZipFile(Path(data) / "masks.zip") as archive:
        output = {}
        for name in archive.namelist():
            if name.startswith("masks/") and name.endswith(".png"):
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
                output[Path(name).stem] = bbbc039_reference_instances(image)
    return output


def evaluate_run(run, references, method, h):
    rows = []
    prediction_files = sorted((run / "predictions").glob("*.png"))
    if len(prediction_files) != 50:
        raise RuntimeError(f"Expected 50 predictions in {run}, found {len(prediction_files)}")
    for path in prediction_files:
        reference = references[path.stem]
        binary = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        prediction = semantic_to_instances(binary, method=method, h=h)
        rows.append({"id": path.stem, **published_metrics(reference, prediction)})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/bbbc039")
    parser.add_argument("--method", choices=("components", "watershed"), default="components")
    parser.add_argument("--h", type=float, default=2.0)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    references = load_raw_annotations(args.data)
    summaries = []
    for supplied in args.runs:
        run = supplied if supplied.is_absolute() else ROOT / supplied
        rows = evaluate_run(run, references, args.method, args.h)
        suffix = args.method if args.method == "components" else f"watershed_h{args.h:g}"
        write_csv(run / f"published_metrics_{suffix}.csv", rows)
        keys = [key for key in rows[0] if key not in ("id", "n_reference", "n_prediction")]
        summary = {
            "run": str(run.relative_to(ROOT)),
            "postprocessing": suffix,
            "n_images": len(rows),
            **{key: float(np.mean([row[key] for row in rows])) for key in keys},
            "mean_reference_instances": float(np.mean([row["n_reference"] for row in rows])),
            "mean_prediction_instances": float(np.mean([row["n_prediction"] for row in rows])),
        }
        (run / f"published_metrics_{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
