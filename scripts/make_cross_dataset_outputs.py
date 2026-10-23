#!/usr/bin/env python3
"""Create one real-output figure spanning both public sets and private UWM data."""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from breastseg.data import load_bbbc039, load_nucleusseg
from breastseg.instance_metrics import bbbc039_reference_instances


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission/isbi2027_overleaf/fig_cross_dataset_outputs.png"


def norm01(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99.5))
    return np.clip((image - low) / (high - low + 1e-7), 0, 1)


def label_rgb(labels):
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros((*labels.shape, 3), dtype=float)
    for identifier in np.unique(labels):
        if identifier == 0:
            continue
        hue = (identifier * 0.61803398875) % 1
        output[labels == identifier] = matplotlib.colors.hsv_to_rgb((hue, .72, 1.0))
    return output


def error_rgb(image, truth, prediction):
    truth, prediction = truth > 0, prediction > 0
    output = np.repeat((norm01(image) * .35)[..., None], 3, axis=2)
    output[truth & prediction] = (.10, .82, .24)
    output[~truth & prediction] = (.95, .12, .12)
    output[truth & ~prediction] = (.15, .40, 1.00)
    return output


def bbbc_case():
    study = ROOT / "results/bbbc_contact_10seed"
    summaries = [json.loads((study / f"vanilla_contact_b7_s{s}/summary.json").read_text())
                 for s in range(10)]
    ordered = sorted(summaries, key=lambda row: row["mean"]["caicedo_af1"])
    seed = ordered[len(ordered) // 2]["seed"]
    run = study / f"vanilla_contact_b7_s{seed}"
    rows = list(csv.DictReader((run / "test_per_image.csv").open()))
    median = float(np.median([float(row["caicedo_af1"]) for row in rows]))
    selected = min(rows, key=lambda row: abs(float(row["caicedo_af1"]) - median))
    case = next(item for item in load_bbbc039(ROOT / "data/bbbc039", ("test",))
                if item["id"] == selected["id"])
    with zipfile.ZipFile(ROOT / "data/bbbc039/masks.zip") as archive:
        raw = cv2.imdecode(np.frombuffer(
            archive.read(f"masks/{case['id']}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    truth = bbbc039_reference_instances(raw)
    prediction = cv2.imread(
        str(run / "predictions" / f"{case['id']}_instances.tif"), cv2.IMREAD_UNCHANGED)
    return case["image"], truth, prediction, {
        "dataset": "BBBC039", "case": case["id"], "seed": seed,
        "metric": "object F1", "score": float(selected["caicedo_af1"]),
    }


def nucleus_case():
    aggregate = json.loads((ROOT / "results/nucleusseg_contact_10seed/aggregate.json").read_text())
    seed = aggregate["qualitative_figure_selection"]["median_seed"]
    identifier = aggregate["qualitative_figure_selection"]["representative_images"]["HepG2"]
    case = next(item for item in load_nucleusseg(ROOT / "data/nucleusseg")
                if item["id"] == identifier)
    run = ROOT / f"results/nucleusseg_contact_10seed/vanilla_contact_b7_s{seed}"
    prediction = cv2.imread(
        str(run / "predictions" / f"{identifier}_instances.tif"), cv2.IMREAD_UNCHANGED)
    row = next(row for row in csv.DictReader((run / "test_per_image.csv").open())
               if row["id"] == identifier)
    return case["image"], case["instances"], prediction, {
        "dataset": "NucleusSegData (HepG2)", "case": identifier, "seed": seed,
        "metric": "nucleus F1", "score": float(row["bilkent_nucleus_f1"]),
    }


def private_case(private_root):
    private_root = Path(private_root).expanduser().resolve()
    if not private_root.is_dir():
        raise FileNotFoundError(f"Private study root does not exist: {private_root}")
    sys.path.insert(0, str(private_root / "src/revision"))
    from train_revision import Net, SliceDS, EdgeChannel

    run = private_root / "outputs/revision/runs/proposed_s0"
    config = json.loads((run / "config.json").read_text())
    scores = {(row["volume"], row["file"]): float(row["dice"])
              for row in csv.DictReader((run / "per_image.csv").open())}
    records = [row for row in json.loads((private_root / "outputs/revision/volume_split.json").read_text())
               if row["new_split"] == "test"]
    ordered = sorted(records, key=lambda row: scores[(row["volume"], row["file"])])
    selected = ordered[len(ordered) // 2]
    dataset = SliceDS([selected], config["size"], EdgeChannel("gabor"), False)
    x, _, _ = dataset[0]
    image, truth = dataset.cache[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(config["arch"], config["encoder"], 4,
                bool(config["scse"]), bool(config["proj"])).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(x[None].to(device)).float())
        probability = F.interpolate(
            probability, size=truth.shape, mode="bilinear", align_corners=False)[0, 0]
    prediction = probability.cpu().numpy() > .5
    truth_labels, _ = ndi.label(truth > 0)
    prediction_labels, _ = ndi.label(prediction)
    return image, truth_labels, prediction_labels, {
        "dataset": "Private UWM MCF10A", "case": f"{selected['volume']}/{selected['file']}",
        "seed": 0, "metric": "Dice", "score": scores[(selected["volume"], selected["file"])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-root", type=Path, required=True,
        help="Local root of the non-distributed UWM revision project",
    )
    args = parser.parse_args()
    rows = (bbbc_case(), nucleus_case(), private_case(args.private_root))
    figure, axes = plt.subplots(3, 4, figsize=(7.15, 5.35), constrained_layout=True)
    column_titles = ("Input", "Reference", "Prediction", "Pixel agreement")
    metadata = []
    for row_index, (image, truth, prediction, info) in enumerate(rows):
        metadata.append(info)
        panels = (norm01(image), label_rgb(truth), label_rgb(prediction),
                  error_rgb(image, truth, prediction))
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel, cmap="gray" if column_index == 0 else None, vmin=0, vmax=1)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(column_titles[column_index], fontsize=8)
        axes[row_index, 0].text(
            -.04, .5,
            f"{info['dataset']}\n{info['metric']}={info['score']:.3f}",
            transform=axes[row_index, 0].transAxes, rotation=90,
            va="center", ha="right", fontsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(figure)
    (OUT.with_suffix(".json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
