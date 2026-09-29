#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained .pt checkpoint on the test set.

Usage:
    python evaluate.py --model_path ./best_model.pt \
                       --data_root /path/to/dataset \
                       --model_type proposed \
                       --save_vis

Produces:
  - Per-image metrics CSV
  - Aggregate metrics JSON
  - (optional) 6-panel visualisation PNGs for every test image
"""

import argparse
import json
import csv
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model
from src.dataset import (BreastCellDataset, build_transforms,
                          find_image_mask_pairs)
from src.metrics import MetricTracker, dice_score, iou_score, precision_recall_f1


def denormalize(img_tensor, num_ch):
    if num_ch == 4:
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(4, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return torch.clamp(img[:3], 0, 1).permute(1, 2, 0).numpy()


def save_visualisation(img_vis, mask_np, pred_prob, pred_bin, save_path,
                        dice, iou, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_vis); axes[0, 0].set_title("Input"); axes[0, 0].axis("off")
    axes[0, 1].imshow(mask_np, cmap="gray"); axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")
    axes[0, 2].imshow(pred_bin, cmap="gray"); axes[0, 2].set_title(f"Prediction (Dice={dice:.3f})"); axes[0, 2].axis("off")

    im = axes[1, 0].imshow(pred_prob, cmap="jet", vmin=0, vmax=1)
    axes[1, 0].set_title("Probability Map"); axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # error map
    err = np.zeros((*mask_np.shape, 3))
    err[(pred_bin > 0) & (mask_np > 0)] = [1, 1, 1]  # TP white
    err[(pred_bin > 0) & (mask_np == 0)] = [1, 0, 0]  # FP red
    err[(pred_bin == 0) & (mask_np > 0)] = [0, 0, 1]  # FN blue
    axes[1, 1].imshow(err); axes[1, 1].set_title("Error (FP=Red, FN=Blue)"); axes[1, 1].axis("off")

    # overlay
    overlay = img_vis.copy()
    ov = np.zeros_like(img_vis)
    ov[:, :, 1] = mask_np
    ov[:, :, 0] = pred_bin
    axes[1, 2].imshow(np.clip(0.6 * overlay + 0.4 * ov, 0, 1))
    axes[1, 2].set_title("Overlay (GT=Green, Pred=Red)"); axes[1, 2].axis("off")

    plt.suptitle(f"{fname}  —  Dice={dice:.4f}  IoU={iou:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_type", default="proposed",
                        help="proposed | unet_resnet34 | etc.")
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent / "eval_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualisations"
    if args.save_vis:
        vis_dir.mkdir(exist_ok=True)

    ch = args.in_channels
    use_gabor = ch == 4

    # model
    model, desc = build_model(args.model_type, in_channels=ch, encoder_weights=None)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected[:5]}")
    model = model.to(device).eval()
    print(f"Loaded: {desc}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}, "
          f"best dice: {ckpt.get('best_dice', '?')}")

    # data
    img_dir = data_root / "test" / "images"
    if not img_dir.exists():
        img_dir = data_root / "test" / "enhanced_images"
    lbl_dir = data_root / "test" / "labels"
    pairs, _ = find_image_mask_pairs(img_dir, lbl_dir)
    print(f"Test pairs: {len(pairs)}")

    ds = BreastCellDataset(
        [p[0] for p in pairs], [p[1] for p in pairs],
        transforms=build_transforms(is_training=False, num_channels=ch),
        is_training=False, use_gabor=use_gabor,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # evaluate
    tracker = MetricTracker()
    per_image = []

    with torch.no_grad():
        for idx, (img, mask) in enumerate(tqdm(loader, desc="Evaluating")):
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            pred_prob = torch.sigmoid(out)
            pred_bin = (pred_prob > 0.5).float()

            d = dice_score(pred_bin, mask)
            io = iou_score(pred_bin, mask)
            p, r, f = precision_recall_f1(pred_bin, mask)
            tracker.update(out, mask)

            fname = pairs[idx][0].name
            per_image.append(dict(
                image=fname, dice=d, iou=io,
                precision=p, recall=r, f1=f,
            ))

            if args.save_vis:
                img_vis = denormalize(img[0], ch)
                mask_np = mask[0, 0].cpu().numpy()
                prob_np = pred_prob[0, 0].cpu().numpy()
                bin_np = pred_bin[0, 0].cpu().numpy()
                save_visualisation(
                    img_vis, mask_np, prob_np, bin_np,
                    vis_dir / f"{idx:04d}_{Path(fname).stem}.png",
                    d, io, fname,
                )

    # save results
    summary = tracker.summary()
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        # remove values list for clean JSON
        clean = {k: {kk: vv for kk, vv in v.items() if kk != "values"}
                  for k, v in summary.items()}
        json.dump(clean, f, indent=2)

    with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_image[0].keys())
        w.writeheader()
        w.writerows(per_image)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    for k in ["dice", "iou", "precision", "recall", "f1"]:
        s = summary[k]
        print(f"  {k:>10}: {s['mean']:.4f} ± {s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
