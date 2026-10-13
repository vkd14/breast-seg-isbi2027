#!/usr/bin/env python3
"""Patient-separated TNBC cross-validation for the proposed model and matched controls."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from breastseg.core import Net, SegmentationLoss, features, metrics, sha256
from breastseg.external import load_external
from train_public_study import save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tnbc_grouped_cv"


class Cached(Dataset):
    def __init__(self, cases, edge, augment=False):
        self.cases = cases
        self.edge = edge
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        x = case["x4"] if self.edge else case["x3"]
        y = case["y"]
        if self.augment:
            k = np.random.randint(4)
            x, y = torch.rot90(x, int(k), (-2, -1)), torch.rot90(y, int(k), (-2, -1))
            if np.random.random() < .5:
                x, y = x.flip(-1), y.flip(-1)
        return x, y, index


class BaselineNet(torch.nn.Module):
    def __init__(self, spec, pretrained=False):
        super().__init__()
        cls = getattr(smp, spec["architecture"])
        kwargs = {
            "encoder_name": spec["encoder"],
            "encoder_weights": "imagenet" if pretrained else None,
            "in_channels": 3,
            "classes": 1,
            "activation": None,
        }
        if spec.get("attention") is not None:
            kwargs["decoder_attention_type"] = spec["attention"]
        self.model = cls(**kwargs)
        self.use_edge = False

    def forward(self, x):
        return self.model(x)


def make_model(spec, pretrained):
    if spec["kind"] == "proposed_family":
        return Net(spec["variant"], pretrained=pretrained)
    if spec["kind"] == "baseline":
        return BaselineNet(spec, pretrained=pretrained)
    raise ValueError(spec["kind"])


@torch.inference_mode()
def evaluate(model, loader, cases, full=False, out=None):
    model.eval()
    rows = []
    if out is not None:
        (out / "predictions").mkdir(parents=True, exist_ok=True)
    for x, _, indices in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda(non_blocking=True))
        probabilities = logits.float().sigmoid().cpu().numpy()[:, 0]
        for probability, index in zip(probabilities, indices.tolist()):
            case = cases[index]
            target = case["mask"].astype(bool)
            prediction = cv2.resize(
                probability, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR
            ) > .5
            if full:
                result = metrics(prediction, target)
            else:
                denominator = prediction.sum() + target.sum()
                result = {
                    "dice": float(2 * (prediction & target).sum() / denominator)
                    if denominator else 1.
                }
            rows.append({"id": case["id"], "group": case["group"], **result})
            if out is not None:
                destination = out / "predictions" / f"{case['id']}.png"
                if not cv2.imwrite(str(destination), prediction.astype(np.uint8) * 255):
                    raise IOError(f"Failed prediction write: {destination}")
    if len(rows) != len(cases):
        raise RuntimeError("Incomplete evaluation")
    return rows


def split_cases(cases, folds, test_fold):
    test_groups = set(folds[test_fold])
    validation_groups = set(folds[(test_fold + 1) % len(folds)])
    train_groups = set().union(*(
        set(groups) for index, groups in enumerate(folds)
        if index not in (test_fold, (test_fold + 1) % len(folds))
    ))
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise RuntimeError("Patient overlap across partitions")
    assigned = train_groups | validation_groups | test_groups
    observed = {case["group"] for case in cases}
    if assigned != observed:
        raise RuntimeError(f"Fold patients differ from dataset: {assigned ^ observed}")
    return (
        [case for case in cases if case["group"] in train_groups],
        [case for case in cases if case["group"] in validation_groups],
        [case for case in cases if case["group"] in test_groups],
    )


def loader(cases, edge, batch_size, augment, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Cached(cases, edge=edge, augment=augment),
        batch_size=batch_size,
        shuffle=augment,
        generator=generator if augment else None,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def run_one(config, name, spec, seed, fold, cases, protocol_hash):
    run_name = f"{name}_f{fold}_s{seed}"
    output = ROOT / "results" / STUDY / run_name
    checkpoint_dir = ROOT / "checkpoints" / STUDY / run_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        previous = json.loads((output / "summary.json").read_text())
        if previous["protocol_sha256"] != protocol_hash:
            raise RuntimeError(f"Protocol differs from completed run: {run_name}")
        print("SKIP completed", run_name, flush=True)
        return
    if (output / "history.json").exists():
        raise RuntimeError(f"Partial run exists; inspect before restarting: {output}")

    start = time.time()
    seed_all(seed)
    train, validation, test = split_cases(cases, config["outer_folds"], fold)
    model = make_model(spec, pretrained=True).cuda()
    initial_hash = state_digest(model)
    train_loader = loader(train, model.use_edge, config["batch_size"], True, seed)
    validation_loader = loader(validation, model.use_edge, config["batch_size"], False, seed)
    loss_function = SegmentationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * .01
    )

    history = []
    best = -1.
    best_epoch = None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        epoch_start = time.time()
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_function(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            losses.append((loss.item(), len(x)))
        validation_rows = evaluate(model, validation_loader, validation)
        score = float(np.mean([row["dice"] for row in validation_rows]))
        if score > best:
            best = score
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": name,
                "spec": spec,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "val_dice": best,
                "protocol_sha256": protocol_hash,
            }, checkpoint_dir / "best.pt")
        record = {
            "epoch": epoch,
            "train_loss": sum(value * count for value, count in losses) / sum(
                count for _, count in losses
            ),
            "val_dice": score,
            "best_val_dice": best,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        save_json(output / "history.json", history)
        save_json(ROOT / "results" / STUDY / "status.json", {
            "state": "training", "run": run_name, **record
        })
        print(json.dumps({"run": run_name, **record}), flush=True)
        scheduler.step()

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_rows = evaluate(
        model,
        loader(test, model.use_edge, config["batch_size"], False, seed),
        test,
        full=True,
        out=output,
    )
    with (output / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    summary = {
        "model": name,
        "spec": spec,
        "seed": seed,
        "fold": fold,
        "train_patients": sorted({case["group"] for case in train}),
        "validation_patients": sorted({case["group"] for case in validation}),
        "test_patients": sorted({case["group"] for case in test}),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_dice": best,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_hash,
        "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"),
        "protocol_sha256": protocol_hash,
        "seconds": time.time() - start,
        "image_mean": {
            key: float(np.mean([row[key] for row in test_rows]))
            for key in test_rows[0] if key not in ("id", "group")
        },
    }
    save_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "tnbc_grouped_cv.json"
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    protocol_hash = sha256(arguments.config)
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)

    cases = load_external(arguments.data, "tnbc")
    for case in cases:
        case["x3"] = features(case["image"], edge=False, size=config["image_size"])
        case["x4"] = features(case["image"], edge=True, size=config["image_size"])
        case["y"] = torch.from_numpy(cv2.resize(
            case["mask"], (config["image_size"], config["image_size"]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)[None])

    output = ROOT / "results" / STUDY
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases:
        test_fold = next(
            index for index, groups in enumerate(config["outer_folds"])
            if case["group"] in groups
        )
        manifest.append({
            "id": case["id"],
            "patient": case["group"],
            "test_fold": test_fold,
            "validation_fold_for_test": (test_fold + 1) % len(config["outer_folds"]),
            "height": case["mask"].shape[0],
            "width": case["mask"].shape[1],
            "foreground_fraction": float(case["mask"].mean()),
        })
    save_json(output / "split_manifest.json", manifest)
    save_json(output / "environment.json", {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "protocol_sha256": protocol_hash,
        "config": config,
        "archives": {
            path.name: sha256(path)
            for path in (arguments.data / "tnbc").glob("*.zip")
        },
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                Path(__file__),
                ROOT / "src/breastseg/core.py",
                ROOT / "src/breastseg/external.py",
            ]
        },
    })

    selected_models = arguments.models or list(config["models"])
    selected_seeds = arguments.seeds if arguments.seeds is not None else config["seeds"]
    selected_folds = arguments.folds if arguments.folds is not None else range(
        len(config["outer_folds"])
    )
    if any(name not in config["models"] for name in selected_models):
        raise ValueError("Unknown model")
    if any(fold < 0 or fold >= len(config["outer_folds"]) for fold in selected_folds):
        raise ValueError("Invalid fold")
    for seed in selected_seeds:
        for fold in selected_folds:
            for name in selected_models:
                run_one(
                    config, name, config["models"][name], seed, fold, cases, protocol_hash
                )
    save_json(output / "status.json", {
        "state": "complete",
        "protocol_sha256": protocol_hash,
        "expected_runs": len(config["models"]) * len(config["seeds"]) * len(
            config["outer_folds"]
        ),
    })


if __name__ == "__main__":
    main()
