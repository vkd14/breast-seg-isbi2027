#!/usr/bin/env python3
"""Train a three-class BBBC039 model for paper-native instance endpoints."""
import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from skimage import morphology, segmentation

from breastseg.core import Net, features, sha256
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import (
    bbbc039_reference_instances,
    classmap_to_instances,
    published_metrics,
)
from train_public_study import save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]
STUDY = os.environ.get("BREASTSEG_BBBC_INSTANCE_STUDY", "bbbc_instance_10seed")


def three_class_target(instances):
    """Create background/interior/three-pixel-boundary targets from instances."""
    instances = np.asarray(instances)
    inner_contour = segmentation.find_boundaries(instances, mode="inner", connectivity=1)
    boundary = morphology.dilation(inner_contour, np.ones((3, 3), bool))
    classes = np.zeros(instances.shape, np.int64)
    classes[instances > 0] = 1
    classes[boundary] = 2
    return classes


def annotation_maps(root):
    output = {}
    with zipfile.ZipFile(Path(root) / "masks.zip") as archive:
        for name in archive.namelist():
            if not name.startswith("masks/") or not name.endswith(".png"):
                continue
            raw = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
            instances = bbbc039_reference_instances(raw)
            # BBBC039 values 1--3 are graph colours that keep touching objects
            # distinguishable; they are not semantic classes.  Derive the
            # three-class target described by Caicedo et al.: background,
            # object interior, and a contour expanded one pixel to each side.
            classes = three_class_target(instances)
            output[Path(name).stem] = (classes, instances)
    return output


class Cached(Dataset):
    def __init__(self, cases, edge, augment=False):
        self.cases, self.edge, self.augment = cases, edge, augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        x = case["x4"] if self.edge else case["x3"]
        y = case["classes"]
        if self.augment:
            k = np.random.randint(4)
            x, y = torch.rot90(x, int(k), (-2, -1)), torch.rot90(y, int(k), (-2, -1))
            if np.random.random() < .5:
                x, y = x.flip(-1), y.flip(-1)
        return x, y, index


def segmentation_loss(logits, target, weights):
    probabilities = logits.float().softmax(1)
    one_hot = F.one_hot(target, 3).permute(0, 3, 1, 2).float()
    dice_terms = []
    for channel in (1, 2):
        p, y = probabilities[:, channel].flatten(1), one_hot[:, channel].flatten(1)
        dice_terms.append(1 - ((2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)).mean())
    return F.cross_entropy(logits.float(), target, weight=weights) + .5 * torch.stack(dice_terms).mean()


@torch.inference_mode()
def evaluate(model, loader, cases, expanded=False, output=None):
    model.eval()
    rows = []
    if output is not None:
        (output / "predictions").mkdir(parents=True, exist_ok=True)
    for x, _, indices in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda(non_blocking=True)).float().cpu().numpy()
        for values, index in zip(logits, indices.tolist()):
            case = cases[index]
            native = np.stack([
                cv2.resize(channel, (case["mask"].shape[1], case["mask"].shape[0]), interpolation=cv2.INTER_LINEAR)
                for channel in values
            ])
            classes = native.argmax(0).astype(np.uint8)
            instances = classmap_to_instances(classes, expand_boundary=expanded)
            rows.append({"id": case["id"], **published_metrics(case["instances"], instances)})
            if output is not None:
                cv2.imwrite(str(output / "predictions" / f"{case['id']}_classes.png"), classes)
                cv2.imwrite(str(output / "predictions" / f"{case['id']}_instances.tif"), instances.astype(np.uint16))
    return rows


def loader(cases, edge, batch_size, augment, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Cached(cases, edge, augment), batch_size=batch_size, shuffle=augment,
                      generator=generator if augment else None, num_workers=0, pin_memory=True)


def run_one(config, name, variant, seed, cases, protocol_hash):
    run_name = f"{name}_s{seed}"
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
        raise RuntimeError(f"Partial run exists: {output}")
    seed_all(seed)
    train = [case for case in cases if case["split"] == "train"]
    validation = [case for case in cases if case["split"] == "val"]
    test = [case for case in cases if case["split"] == "test"]
    model = Net(variant, pretrained=True, classes=3).cuda()
    initial_hash = state_digest(model)
    counts = np.bincount(np.concatenate([case["classes"].numpy().ravel() for case in train]), minlength=3)
    weights = torch.from_numpy(np.sqrt(counts.sum() / np.maximum(counts, 1))).float().cuda()
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * .01)
    train_loader = loader(train, model.use_edge, config["batch_size"], True, seed)
    validation_loader = loader(validation, model.use_edge, config["batch_size"], False, seed)
    start, best, best_epoch, history = time.time(), -1., None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        epoch_start = time.time()
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = segmentation_loss(model(x), y, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append((loss.item(), len(x)))
        rows = evaluate(model, validation_loader, validation)
        score = float(np.mean([row["caicedo_af1"] for row in rows]))
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "model": name, "variant": variant,
                        "seed": seed, "epoch": epoch, "val_caicedo_af1": best,
                        "protocol_sha256": protocol_hash}, checkpoint_dir / "best.pt")
        record = {"epoch": epoch, "train_loss": sum(v*n for v,n in losses)/sum(n for _,n in losses),
                  "val_caicedo_af1": score, "best_val_caicedo_af1": best,
                  "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time()-epoch_start}
        history.append(record)
        save_json(output / "history.json", history)
        save_json(ROOT / "results" / STUDY / "status.json", {"state": "training", "run": run_name, **record})
        print(json.dumps({"run": run_name, **record}), flush=True)
        scheduler.step()
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_loader = loader(test, model.use_edge, config["batch_size"], False, seed)
    rows = evaluate(model, test_loader, test, expanded=False, output=output)
    expanded_rows = evaluate(model, test_loader, test, expanded=True)
    for suffix, values in (("interior", rows), ("expanded", expanded_rows)):
        with (output / f"test_{suffix}_per_image.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(values)
    metric_keys = [key for key in rows[0] if key not in ("id", "n_reference", "n_prediction")]
    summary = {"model": name, "variant": variant, "seed": seed, "n_test": 50,
               "epochs": config["epochs"], "best_epoch": best_epoch, "best_val_caicedo_af1": best,
               "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial_hash,
               "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"), "protocol_sha256": protocol_hash,
               "seconds": time.time()-start,
               "interior_mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_keys},
               "expanded_mean": {key: float(np.mean([row[key] for row in expanded_rows])) for key in metric_keys}}
    save_json(output / "summary.json", summary)
    print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bbbc_instance_10seed.json")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol_hash = sha256(args.config)
    cv2.setNumThreads(1); torch.set_num_threads(8)
    cases = load_bbbc039(args.data)
    annotations = annotation_maps(args.data)
    for case in cases:
        case["x3"] = features(case["image"], edge=False, size=config["image_size"])
        case["x4"] = features(case["image"], edge=True, size=config["image_size"])
        classes, instances = annotations[case["id"]]
        case["classes"] = torch.from_numpy(cv2.resize(classes, (config["image_size"], config["image_size"]),
                                                        interpolation=cv2.INTER_NEAREST).astype(np.int64))
        case["instances"] = instances
    study = ROOT / "results" / STUDY
    study.mkdir(parents=True, exist_ok=True)
    save_json(study / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "protocol_sha256": protocol_hash, "config": config})
    selected_models = args.models or list(config["models"])
    selected_seeds = args.seeds if args.seeds is not None else config["seeds"]
    for seed in selected_seeds:
        for name in selected_models:
            run_one(config, name, config["models"][name], seed, cases, protocol_hash)
    save_json(study / "status.json", {"state": "complete", "protocol_sha256": protocol_hash})


if __name__ == "__main__":
    main()
