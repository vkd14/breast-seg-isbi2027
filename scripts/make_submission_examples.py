#!/usr/bin/env python3
"""Create a publication figure from actual public images and annotations."""
import argparse
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries

from breastseg.instance_metrics import bbbc039_reference_instances


def normalize(image):
    values = np.asarray(image, np.float32)
    low, high = np.percentile(values, (1, 99.5))
    return np.clip((values - low) / (high - low + 1e-7), 0, 1)


def overlay(image, labels, color=(1.0, 0.18, 0.05)):
    gray = normalize(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[find_boundaries(labels, mode="inner")] = color
    return rgb


def bilkent_pair(root, subset, image_name, annotation_name):
    image = cv2.cvtColor(cv2.imread(str(root / subset / image_name)), cv2.COLOR_BGR2RGB)
    labels = np.loadtxt(root / subset / annotation_name, dtype=np.int32)
    return image, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbbc", type=Path, default=Path("data/bbbc039"))
    parser.add_argument("--nucleusseg", type=Path, default=Path("data/nucleusseg/NucleusSegData"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/bbbc_contact_pilot/vanilla_contact_b7_s0/predictions"))
    parser.add_argument("--output", type=Path,
                        default=Path("submission/isbi2027_overleaf/fig_dataset_examples.png"))
    args = parser.parse_args()
    identifier = "IXMtest_O04_s2_w19A18B0D8-8E8B-4572-A5C0-5C678F3AD54C"
    with zipfile.ZipFile(args.bbbc / "images.zip") as images, zipfile.ZipFile(args.bbbc / "masks.zip") as masks:
        image = cv2.imdecode(np.frombuffer(images.read(f"images/{identifier}.tif"), np.uint8), -1)
        raw_mask = cv2.imdecode(np.frombuffer(masks.read(f"masks/{identifier}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    reference = bbbc039_reference_instances(raw_mask)
    prediction = cv2.imread(str(args.predictions / f"{identifier}_instances.tif"), cv2.IMREAD_UNCHANGED)
    huh7_image, huh7_labels = bilkent_pair(args.nucleusseg, "Huh7TestSet", "huh7_ts10.jpg", "huh7_ts_ann10")
    hepg2_image, hepg2_labels = bilkent_pair(args.nucleusseg, "HepG2TestSet", "hepg2_ts14.jpg", "hepg2_ts_ann14")

    panels = (
        (normalize(image), "(a) BBBC039 input"),
        (overlay(image, reference), "(b) BBBC039 reference"),
        (overlay(image, prediction, (0.05, 0.85, 0.25)), "(c) Contact prediction"),
        (normalize(huh7_image), "(d) Huh7 input"),
        (overlay(huh7_image, huh7_labels), "(e) Huh7 reference"),
        (overlay(hepg2_image, hepg2_labels), "(f) HepG2 reference"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.3), constrained_layout=True)
    for axis, (panel, title) in zip(axes.ravel(), panels):
        axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
