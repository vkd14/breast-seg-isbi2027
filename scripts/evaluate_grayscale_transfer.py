#!/usr/bin/env python3
"""One-shot audit of explicit and polarity-inverted grayscale TNBC transfer.

The active feature function already converts RGB to grayscale.  This script first
verifies that an explicit OpenCV RGB-to-gray conversion produces bit-identical
model inputs, then evaluates a biologically motivated, predeclared polarity
inversion.  It never trains, changes thresholds, or selects a checkpoint on TNBC.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from breastseg.core import Net, features, metrics, sha256
from breastseg.external import load_external
from breastseg.statistics import holm, paired


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "vanilla_b7": "Vanilla B7",
    "gabor_b7": "Gabor bundle",
    "gabor_boundary_b7": "Gabor + boundary",
    "scse_b7": "SCSE-only B7",
    "residual_gabor_b7": "Residual Gabor",
    "residual_sobel_b7": "Residual Sobel",
    "residual_gabor_boundary_b7": "Residual Gabor + boundary",
}


def explicit_gray(image):
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()


def input_tensor(image, model, inverted=False):
    gray = explicit_gray(image)
    if inverted:
        if not np.issubdtype(gray.dtype, np.integer):
            raise ValueError("Polarity inversion requires an integer image")
        gray = np.iinfo(gray.dtype).max - gray
    kind = "sobel" if "sobel" in model.variant else "gabor"
    x = features(gray, edge=model.use_edge, edge_kind=kind)
    return x


def tensor_sha256(x):
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


@torch.inference_mode()
def evaluate_checkpoint(checkpoint, cases, mode):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Net(state["variant"])
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.cuda().eval()
    xs = [input_tensor(c["image"], model, inverted=(mode == "inverted_grayscale")) for c in cases]
    rows = []
    start = time.time()
    for first in range(0, len(cases), 8):
        probs = model(torch.stack(xs[first:first + 8]).cuda()).sigmoid().float().cpu().numpy()[:, 0]
        for case, prob in zip(cases[first:first + 8], probs):
            gt = case["mask"]
            pred = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) > .5
            rows.append(dict(id=case["id"], group=case["group"], **metrics(pred, gt)))
    model.cpu()
    torch.cuda.empty_cache()
    patient = []
    for group in sorted({r["group"] for r in rows}):
        patient.append(float(np.mean([r["dice"] for r in rows if r["group"] == group])))
    return state, rows, {
        "patient_macro_dice": float(np.mean(patient)),
        "image_macro_dice": float(np.mean([r["dice"] for r in rows])),
        "patient_scores": patient,
        "seconds": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grayscale_transfer_audit")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    cases = load_external(args.data, "tnbc")
    args.output.mkdir(parents=True, exist_ok=True)

    # Prove that the current RGB entry point is already explicit grayscale.
    equivalence = []
    for dataset, audit_cases in (("TNBC", cases), ("BBBC038", load_external(args.data, "bbbc038"))):
        for case in audit_cases:
            rgb = features(case["image"], edge=True, edge_kind="gabor")
            gray = features(explicit_gray(case["image"]), edge=True, edge_kind="gabor")
            equivalence.append({
                "dataset": dataset,
                "id": case["id"],
                "rgb_entry_sha256": tensor_sha256(rgb),
                "explicit_gray_sha256": tensor_sha256(gray),
                "bit_identical": bool(torch.equal(rgb, gray)),
            })
    if not all(r["bit_identical"] for r in equivalence):
        raise RuntimeError("Explicit grayscale did not reproduce the active RGB entry path")
    (args.output / "grayscale_equivalence.json").write_text(json.dumps(equivalence, indent=2))

    protocol = {
        "dataset": "TNBC v1.1, all 50 images in 11 patient groups",
        "purpose": "post-hoc zero-shot transfer stress test; not confirmatory external validation",
        "training": "none",
        "selection": "none on TNBC; all 21 public-study checkpoints retained",
        "threshold": .5,
        "tta": False,
        "modes": ["explicit_grayscale", "inverted_grayscale"],
        "rationale": "BBBC039 has bright Hoechst nuclei on dark background; H&E nuclei are usually dark on bright tissue",
        "primary_reporting": "patient-macro semantic Dice; image-level rows retained",
        "warning": "TNBC labels are used in this one-shot post-hoc analysis and must not be reused for preprocessing selection",
    }
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    protocol["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    summary_rows = []
    checkpoints = []
    for study in ("public_study", "public_followup"):
        checkpoints.extend(sorted((ROOT / "checkpoints" / study).glob("*/best.pt")))
    for checkpoint in checkpoints:
        for mode in protocol["modes"]:
            run = checkpoint.parent.name
            out = args.output / run / mode
            saved = out / "summary.json"
            if saved.exists():
                summary_rows.append(json.loads(saved.read_text()))
                print("SKIP", run, mode, flush=True)
                continue
            state, rows, summary = evaluate_checkpoint(checkpoint, cases, mode)
            out.mkdir(parents=True, exist_ok=True)
            with (out / "per_image.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            record = {
                "run": run,
                "variant": state["variant"],
                "label": VARIANTS[state["variant"]],
                "seed": int(state["seed"]),
                "mode": mode,
                "checkpoint_sha256": sha256(checkpoint),
                **summary,
            }
            (out / "summary.json").write_text(json.dumps(record, indent=2))
            summary_rows.append(record)
            print("COMPLETE", run, mode, f"patient Dice={summary['patient_macro_dice']:.6f}", flush=True)

    compact = [{k: row[k] for k in ("run", "variant", "label", "seed", "mode", "patient_macro_dice", "image_macro_dice")}
               for row in summary_rows]
    with (args.output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)

    aggregate = []
    for variant in VARIANTS:
        modes = {}
        for mode in protocol["modes"]:
            vals = [r["patient_macro_dice"] for r in summary_rows if r["variant"] == variant and r["mode"] == mode]
            modes[mode] = {"mean": float(np.mean(vals)), "seed_sd": float(np.std(vals, ddof=1)), "values": vals}
        gray_patients = np.mean([r["patient_scores"] for r in summary_rows
                                 if r["variant"] == variant and r["mode"] == "explicit_grayscale"], axis=0)
        inverted_patients = np.mean([r["patient_scores"] for r in summary_rows
                                     if r["variant"] == variant and r["mode"] == "inverted_grayscale"], axis=0)
        patient_comparison = paired(inverted_patients, gray_patients)
        aggregate.append({
            "variant": variant,
            "label": VARIANTS[variant],
            **modes,
            "inversion_delta_percentage_points": 100 * (modes["inverted_grayscale"]["mean"] - modes["explicit_grayscale"]["mean"]),
            "paired_patient_inverted_minus_gray": patient_comparison,
        })
    adjusted = holm([row["paired_patient_inverted_minus_gray"]["p"] for row in aggregate])
    for row, p_adjusted in zip(aggregate, adjusted):
        row["paired_patient_inverted_minus_gray"]["p_holm_7"] = p_adjusted
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
