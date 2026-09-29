#!/usr/bin/env python3
"""
run_all_experiments.py
======================
One command to train every model needed for the EMBC 2026 paper:
  - Proposed model (EfficientNet-B7 / UNet++ / SCSE / 4-ch Gabor)
  - 5 SOTA baselines (3-channel, no Gabor, standard Dice+BCE loss)
  - 4 ablation variants (remove one component at a time)

Usage:
    python run_all_experiments.py --data_root /path/to/dataset \
                                  --epochs 150 --batch_size 4

Results are saved to ./results/<model_name>/ and a final comparison
CSV is written to ./results/comparison_table.csv

Expected dataset layout:
    data_root/
      train/images/   train/labels/
      val/images/     val/labels/
      test/images/    test/labels/
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# ---- project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model, BASELINE_REGISTRY
from src.dataset import (BreastCellDataset, build_transforms,
                          find_image_mask_pairs, make_sampler)
from src.losses import StabilizedCombinedLoss, StandardDiceBCE
from src.trainer import Trainer


def setup_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
        print("Using Apple MPS")
    else:
        d = torch.device("cpu")
        print("Using CPU")
    return d


def build_loaders(data_root, split, batch_size, use_gabor, is_training):
    ch = 4 if use_gabor else 3
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"

    # fallback: try enhanced_images if images/ doesn't exist
    if not img_dir.exists():
        img_dir = data_root / split / "enhanced_images"

    pairs, unmatched = find_image_mask_pairs(img_dir, lbl_dir)
    print(f"  {split}: {len(pairs)} pairs, {len(unmatched)} unmatched")

    ds = BreastCellDataset(
        [p[0] for p in pairs], [p[1] for p in pairs],
        transforms=build_transforms(is_training=is_training, num_channels=ch),
        is_training=is_training, use_gabor=use_gabor,
    )
    sampler = make_sampler(ds) if is_training else None
    loader = DataLoader(
        ds, batch_size=batch_size,
        sampler=sampler, shuffle=(is_training and sampler is None),
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    return loader


# ======================================================================
#  Experiment definitions
# ======================================================================

def get_experiments():
    """
    Returns list of dicts, each defining one experiment:
      name, model_key, in_channels, use_gabor, use_weighted_sampling, loss
    """
    exps = []

    # ---- Proposed ----
    exps.append(dict(
        name="proposed_full",
        model_key="proposed",
        in_channels=4,
        use_gabor=True,
        use_weighted_sampling=True,
        loss="stabilized",
        desc="Proposed (EfficientNet-B7 / UNet++ / SCSE / Gabor / Stabilized Loss)",
    ))

    # ---- SOTA baselines (all 3-ch, standard loss, no weighted sampling) ----
    for key in ["unet_resnet34", "unet_resnet50", "unetpp_resnet50",
                "attention_unet_resnet50", "deeplabv3p_resnet50",
                "manet_resnet50", "fpn_resnet50"]:
        exps.append(dict(
            name=f"sota_{key}",
            model_key=key,
            in_channels=3,
            use_gabor=False,
            use_weighted_sampling=False,
            loss="standard",
            desc=BASELINE_REGISTRY[key]["desc"],
        ))

    # ---- Ablations (remove one proposed component at a time) ----
    # A1: no Gabor edge channel (3-ch input)
    exps.append(dict(
        name="ablation_no_gabor",
        model_key="proposed",
        in_channels=3,
        use_gabor=False,
        use_weighted_sampling=True,
        loss="stabilized",
        desc="Ours w/o Gabor edge channel (3-ch input)",
    ))

    # A2: no weighted sampling
    exps.append(dict(
        name="ablation_no_weighted_sampling",
        model_key="proposed",
        in_channels=4,
        use_gabor=True,
        use_weighted_sampling=False,
        loss="stabilized",
        desc="Ours w/o complexity-weighted sampling",
    ))

    # A3: standard loss instead of stabilized
    exps.append(dict(
        name="ablation_standard_loss",
        model_key="proposed",
        in_channels=4,
        use_gabor=True,
        use_weighted_sampling=True,
        loss="standard",
        desc="Ours w/ standard Dice+BCE loss (no stabilisation)",
    ))

    # A4: no SCSE attention
    exps.append(dict(
        name="ablation_no_scse",
        model_key="ablation_no_scse",
        in_channels=4,
        use_gabor=True,
        use_weighted_sampling=True,
        loss="stabilized",
        desc="Ours w/o SCSE decoder attention",
    ))

    # A5: U-Net instead of UNet++
    exps.append(dict(
        name="ablation_unet_not_unetpp",
        model_key="ablation_unet_eb7",
        in_channels=4,
        use_gabor=True,
        use_weighted_sampling=True,
        loss="stabilized",
        desc="Ours w/ plain U-Net decoder (not UNet++)",
    ))

    return exps


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this experiment name (for debugging)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, just evaluate existing checkpoints")
    parser.add_argument("--existing_model", type=str, default=None,
                        help="Path to existing .pt for proposed model (skip its training)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    device = setup_device()
    results_root = Path("./results")
    results_root.mkdir(exist_ok=True)

    experiments = get_experiments()
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"No experiment named '{args.only}'")
            return

    all_results = []

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"  EXPERIMENT: {exp['name']}")
        print(f"  {exp['desc']}")
        print("=" * 80)

        save_dir = results_root / exp["name"]
        save_dir.mkdir(parents=True, exist_ok=True)

        ch = exp["in_channels"]
        use_gabor = exp["use_gabor"]

        # --- build model ---
        model, model_desc = build_model(
            exp["model_key"], in_channels=ch,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # --- loss ---
        criterion = (StabilizedCombinedLoss() if exp["loss"] == "stabilized"
                      else StandardDiceBCE())

        # --- loaders ---
        train_loader = build_loaders(data_root, "train", args.batch_size,
                                      use_gabor, is_training=True)
        val_loader = build_loaders(data_root, "val", args.batch_size,
                                    use_gabor, is_training=False)
        test_loader = build_loaders(data_root, "test", 1,
                                     use_gabor, is_training=False)

        # --- train or load ---
        trainer = Trainer(model, criterion, device, save_dir=str(save_dir),
                          accum_steps=2, patience=30)

        ckpt_path = save_dir / f"best_{exp['name']}.pt"

        if args.skip_train and ckpt_path.exists():
            print(f"  Loading existing checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.model = model.to(device)
        elif (exp["name"] == "proposed_full" and args.existing_model):
            print(f"  Loading provided model: {args.existing_model}")
            ckpt = torch.load(args.existing_model, map_location=device,
                              weights_only=False)
            # strict=False handles minor key mismatches from legacy checkpoints
            missing, unexpected = model.load_state_dict(
                ckpt["model_state_dict"], strict=False)
            if missing:
                print(f"  WARNING: missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
            if unexpected:
                print(f"  WARNING: unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
            trainer.model = model.to(device)
        else:
            best = trainer.train(train_loader, val_loader,
                                  epochs=args.epochs, tag=exp["name"])
            print(f"  Best val dice: {best:.4f}")
            # reload best
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device,
                                  weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                trainer.model = model.to(device)

        # --- evaluate on test set ---
        print(f"  Evaluating on test set...")
        summary = trainer.evaluate(test_loader, desc="Test")

        row = dict(
            name=exp["name"],
            description=exp["desc"],
            params=n_params,
            dice_mean=summary["dice"]["mean"],
            dice_std=summary["dice"]["std"],
            iou_mean=summary["iou"]["mean"],
            iou_std=summary["iou"]["std"],
            precision=summary["precision"]["mean"],
            recall=summary["recall"]["mean"],
            f1=summary["f1"]["mean"],
            dice_min=summary["dice"]["min"],
            dice_max=summary["dice"]["max"],
        )
        all_results.append(row)

        # save per-experiment results
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  TEST  Dice={row['dice_mean']:.4f}±{row['dice_std']:.4f}  "
              f"IoU={row['iou_mean']:.4f}  "
              f"Prec={row['precision']:.4f}  Rec={row['recall']:.4f}")

    # --- write comparison table ---
    csv_path = results_root / "comparison_table.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_results)

    print("\n" + "=" * 80)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nComparison table saved to: {csv_path}")
    print(f"\n{'Method':<50} {'Dice':>12} {'IoU':>12} {'Prec':>8} {'Rec':>8}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: -x["dice_mean"]):
        print(f"{r['description'][:50]:<50} "
              f"{r['dice_mean']:.4f}±{r['dice_std']:.4f} "
              f"{r['iou_mean']:.4f}±{r['iou_std']:.4f} "
              f"{r['precision']:.4f} {r['recall']:.4f}")


if __name__ == "__main__":
    main()
