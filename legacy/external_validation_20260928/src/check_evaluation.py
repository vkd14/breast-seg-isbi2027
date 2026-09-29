"""Small checks for mask decoding, metric conventions, and saved inference."""
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import evaluate_bbbc038 as ev


def main():
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    # Nonsquare mask checks Kaggle's column-major, one-indexed RLE.
    expected = np.zeros((3, 5), bool)
    expected[1:, 0] = True
    expected[0, 2] = True
    assert np.array_equal(ev.decode_rle("2 2 7 1", 3, 5), expected)
    perfect = ev.metrics(expected, expected)
    assert perfect["dice"] == perfect["iou"] == perfect["boundary_f1_2px"] == 1
    assert perfect["hd95_px"] == 0
    empty = np.zeros_like(expected)
    assert ev.metrics(empty, empty)["dice"] == 1
    assert ev.metrics(empty, expected)["dice"] == 0
    assert ev.metrics(empty, expected)["hd95_px"] == np.hypot(3, 5)
    for ds, count in (("bbbc038", 65), ("bbbc039", 50), ("tnbc", 50)):
        ev.DATA = ev.ROOT / "data" / ds
        cases = {s: (im, gt) for s, im, gt, n in ev.load_cases(ds)}
        result_dir = ev.ROOT / "outputs" / (ds + "_zero_shot")
        rows = list(csv.DictReader(open(result_dir / "per_image.csv")))
        assert len(rows) == 4 * count
        assert len({(r["method"], r["image_id"]) for r in rows}) == len(rows)
        for row in rows:
            gt = cases[row["image_id"]][1]
            pred = cv2.imread(str(result_dir / "predictions" / row["method"] / (row["image_id"] + ".png")), 0) > 0
            values = ev.metrics(pred, gt)
            for key in ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px"):
                assert np.isclose(values[key], float(row[key]), atol=1e-10)
        print(f"PASS {ds}: all {len(rows)} saved predictions reproduce recorded metrics")
    # Independent preprocessing comparison against the original revised dataset path.
    if not (ev.PROJECT / "data&code/outputs/revision/volume_split.json").exists():
        print("SKIP private preprocessing cross-check: original training project is not present")
        return
    sys.path.insert(0, str(ev.PROJECT / "data&code/src/revision"))
    from train_revision import EdgeChannel, SliceDS
    import json
    rec = next(r for r in json.load(open(ev.PROJECT / "data&code/outputs/revision/volume_split.json")) if r["new_split"] == "test")
    path = ev.PROJECT / "data&code/data/All ZIP Files/SuperMasterDataset" / rec["orig_split"] / "images" / rec["file"]
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgb = np.repeat(raw[..., None], 3, axis=2)
    expected_tensor = SliceDS([rec], 512, EdgeChannel("gabor"), False)[0][0]
    actual_tensor = ev.preprocess(rgb, True)[0]
    assert torch.allclose(expected_tensor, actual_tensor, atol=2e-6), (expected_tensor - actual_tensor).abs().max()
    print("PASS revised preprocessing matches the existing training dataset implementation")


if __name__ == "__main__":
    main()
