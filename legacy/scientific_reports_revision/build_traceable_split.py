#!/usr/bin/env python3
"""Create an exact-provenance split file for the eight named annotated stacks.

The 855-slice SuperMasterDataset contains 334 image/mask pairs that match the
eight named ``AnnotatedDATA_REVISED`` stacks byte-for-byte (images) and
pixel-for-pixel after mask binarisation.  The other 523 pairs have no retained
source-stack identifiers in this repository.  This script records only the
traceable subset, using image hashes rather than image dimensions.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import cv2
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "All ZIP Files")
ANNOTATED = os.path.join(DATA, "AnnotatedDATA_REVISED")
MASTER = os.path.join(DATA, "SuperMasterDataset")
OUT = os.path.join(ROOT, "outputs", "revision", "traceable_split.json")


def digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


reference = {}
for directory in sorted(glob.glob(os.path.join(ANNOTATED, "*"))):
    if not os.path.isdir(directory) or directory.endswith("-MASK") or "split" in directory.lower():
        continue
    mask_directory = directory + "-MASK"
    volume = os.path.basename(directory)
    for image_path in sorted(glob.glob(os.path.join(directory, "*.tif"))):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask_path = os.path.join(mask_directory, os.path.basename(image_path))
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"unreadable named pair: {image_path}")
        reference[digest(image)] = (volume, mask > 0, image.shape)

rows = []
for original_split in ("train", "val", "test"):
    pattern = os.path.join(MASTER, original_split, "images", "*.tif")
    for image_path in sorted(glob.glob(pattern)):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        match = reference.get(digest(image))
        if match is None:
            continue
        volume, reference_mask, shape = match
        name = os.path.basename(image_path)
        mask_path = os.path.join(MASTER, original_split, "labels", name)
        master_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 0
        if not np.array_equal(reference_mask, master_mask):
            raise RuntimeError(f"mask mismatch for {original_split}/{name} ({volume})")
        rows.append({
            "volume": volume,
            "named_volume": volume,
            "shape": list(shape),
            "orig_split": original_split,
            "file": name,
            "new_split": "traceable_cv",
            "has_nucleus": bool(master_mask.any()),
            "fg_fraction": round(float(master_mask.mean()), 8),
            "provenance": "exact image hash and binary-mask match to AnnotatedDATA_REVISED",
        })

volumes = sorted({row["volume"] for row in rows})
if len(rows) != 334 or len(volumes) != 8:
    raise RuntimeError(f"expected 334 pairs from 8 stacks; found {len(rows)} from {len(volumes)}")

with open(OUT, "w") as handle:
    json.dump(rows, handle, indent=2)

print(f"wrote {OUT}: {len(rows)} pairs from {len(volumes)} named stacks")
for volume in volumes:
    selected = [row for row in rows if row["volume"] == volume]
    print(f"  {volume}: {len(selected)} slices, shape {selected[0]['shape']}")
