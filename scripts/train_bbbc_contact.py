#!/usr/bin/env python3
"""Train a foreground model with an auxiliary touching-contact boundary head."""
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
from scipy import ndimage as ndi
from skimage import measure, morphology
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from breastseg.core import Net, features, sha256, soft_dice
from breastseg.data import load_bbbc039
from breastseg.instance_metrics import bbbc039_reference_instances, published_metrics, remap_labels
from train_public_study import save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]
STUDY = os.environ.get("BREASTSEG_BBBC_CONTACT_STUDY", "bbbc_contact_10seed")


def touching_contact_target(instances, radius=1):
    """Mark only interfaces between adjacent non-zero instance IDs."""
    labels = np.asarray(instances)
    if labels.ndim != 2 or (labels < 0).any():
        raise ValueError("Expected a nonnegative 2D instance map")
    contact = np.zeros(labels.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        source_y = slice(max(0, -dy), labels.shape[0] - max(0, dy))
        source_x = slice(max(0, -dx), labels.shape[1] - max(0, dx))
        target_y = slice(max(0, dy), labels.shape[0] - max(0, -dy))
        target_x = slice(max(0, dx), labels.shape[1] - max(0, -dx))
        first, second = labels[source_y, source_x], labels[target_y, target_x]
        differs = (first > 0) & (second > 0) & (first != second)
        contact[source_y, source_x] |= differs
        contact[target_y, target_x] |= differs
    if radius:
        contact = morphology.dilation(contact, morphology.disk(radius))
    return contact & (labels > 0)


def instance_annotations(root):
    output = {}
    with zipfile.ZipFile(Path(root) / "masks.zip") as archive:
        for name in archive.namelist():
            if name.startswith("masks/") and name.endswith(".png"):
                raw = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_UNCHANGED)
                output[Path(name).stem] = bbbc039_reference_instances(raw)
    return output


class Cached(Dataset):
    def __init__(self, cases, edge, augment=False):
        self.cases, self.edge, self.augment = cases, edge, augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        x = case["x4"] if self.edge else case["x3"]
        target = case["target"]
        if self.augment:
            k = int(np.random.randint(4))
            x, target = torch.rot90(x, k, (-2, -1)), torch.rot90(target, k, (-2, -1))
            if np.random.random() < .5:
                x, target = x.flip(-1), target.flip(-1)
        return x, target, index


def contact_loss(logits, target, contact_weight):
    logits, target = logits.float(), target.float()
    foreground_loss = .5 * soft_dice(logits[:, :1].sigmoid(), target[:, :1]) + \
        .5 * F.binary_cross_entropy_with_logits(logits[:, :1], target[:, :1])
    contact = target[:, 1:2]
    ratio = contact.mean()
    positive_weight = torch.where(
        ratio > 0,
        ((1 - ratio) / ratio.clamp_min(1e-7)).clamp(1, 50),
        torch.ones_like(ratio),
    )
    boundary_loss = .5 * soft_dice(logits[:, 1:2].sigmoid(), contact) + \
        .5 * F.binary_cross_entropy_with_logits(logits[:, 1:2], contact, pos_weight=positive_weight)
    loss = foreground_loss + contact_weight * boundary_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite contact-head loss")
    return loss


def contact_instances(foreground, contact, min_size=16):
    foreground = morphology.remove_small_holes(np.asarray(foreground, bool), max_size=min_size - 1)
    foreground = morphology.remove_small_objects(foreground, max_size=min_size - 1)
    markers_binary = morphology.remove_small_objects(foreground & ~np.asarray(contact, bool), max_size=min_size - 1)
    markers = remap_labels(measure.label(markers_binary, connectivity=1))
    if markers.max() == 0:
        return remap_labels(measure.label(foreground, connectivity=1))
    _, indices = ndi.distance_transform_edt(markers == 0, return_indices=True)
    output = markers[tuple(indices)]
    output[~foreground] = 0
    return remap_labels(output)


def loader(cases, edge, batch_size, augment, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Cached(cases, edge, augment), batch_size=batch_size, shuffle=augment,
                      generator=generator if augment else None, num_workers=0, pin_memory=True)


@torch.inference_mode()
def predict_probabilities(model, data_loader, cases):
    model.eval()
    output = {}
    for x, _, indices in data_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(x.cuda(non_blocking=True)).float().sigmoid().cpu().numpy()
        for probability, index in zip(values, indices.tolist()):
            case = cases[index]
            output[case["id"]] = np.stack([
                cv2.resize(channel, (case["mask"].shape[1], case["mask"].shape[0]),
                           interpolation=cv2.INTER_LINEAR)
                for channel in probability
            ])
    return output


def evaluate(probabilities, cases, contact_threshold=.5, output=None):
    rows = []
    if output is not None:
        (output / "predictions").mkdir(parents=True, exist_ok=True)
    for case in cases:
        probability = probabilities[case["id"]]
        foreground, contact = probability[0] > .5, probability[1] > contact_threshold
        instances = contact_instances(foreground, contact)
        rows.append({"id": case["id"], **published_metrics(case["instances"], instances)})
        if output is not None:
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_foreground.png"), foreground.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_contact.png"), contact.astype(np.uint8) * 255)
            cv2.imwrite(str(output / "predictions" / f"{case['id']}_instances.tif"), instances.astype(np.uint16))
    return rows


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
    model = Net(variant, pretrained=True, classes=2).cuda()
    initial_hash = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * .01)
    train_loader = loader(train, model.use_edge, config["batch_size"], True, seed)
    validation_loader = loader(validation, model.use_edge, config["batch_size"], False, seed)
    start, best, best_epoch, history = time.time(), -1., None, []
    for epoch in range(1, config["epochs"] + 1):
        model.train(); losses = []; epoch_start = time.time()
        for x, target, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = contact_loss(model(x.cuda(non_blocking=True)), target.cuda(non_blocking=True), config["contact_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); losses.append((loss.item(), len(x)))
        probabilities = predict_probabilities(model, validation_loader, validation)
        rows = evaluate(probabilities, validation, contact_threshold=.5)
        score = float(np.mean([row["caicedo_af1"] for row in rows]))
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "model": name, "variant": variant,
                        "seed": seed, "epoch": epoch, "val_caicedo_af1": best,
                        "protocol_sha256": protocol_hash}, checkpoint_dir / "best.pt")
        record = {"epoch": epoch, "train_loss": sum(v*n for v,n in losses)/sum(n for _,n in losses),
                  "val_caicedo_af1": score, "best_val_caicedo_af1": best,
                  "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.time()-epoch_start}
        history.append(record); save_json(output / "history.json", history)
        save_json(ROOT / "results" / STUDY / "status.json", {"state": "training", "run": run_name, **record})
        print(json.dumps({"run": run_name, **record}), flush=True)
        scheduler.step()

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation_probabilities = predict_probabilities(model, validation_loader, validation)
    candidates = config["contact_thresholds"]
    threshold_scores = {}
    for threshold in candidates:
        rows = evaluate(validation_probabilities, validation, contact_threshold=threshold)
        threshold_scores[str(threshold)] = float(np.mean([row["caicedo_af1"] for row in rows]))
    selected_threshold = max(candidates, key=lambda value: threshold_scores[str(value)])
    test_probabilities = predict_probabilities(
        model, loader(test, model.use_edge, config["batch_size"], False, seed), test)
    rows = evaluate(test_probabilities, test, contact_threshold=selected_threshold, output=output)
    with (output / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    metric_keys = [key for key in rows[0] if key not in ("id", "n_reference", "n_prediction")]
    summary = {"model": name, "variant": variant, "seed": seed, "n_test": 50,
               "epochs": config["epochs"], "best_epoch": best_epoch, "best_val_caicedo_af1": best,
               "selected_contact_threshold": selected_threshold, "validation_threshold_scores": threshold_scores,
               "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial_hash,
               "checkpoint_sha256": sha256(checkpoint_dir / "best.pt"), "protocol_sha256": protocol_hash,
               "seconds": time.time()-start,
               "mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}}
    save_json(output / "summary.json", summary); print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer; torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(); config = json.loads(args.config.read_text()); protocol_hash = sha256(args.config)
    cv2.setNumThreads(1); torch.set_num_threads(8)
    cases = load_bbbc039(args.data); annotations = instance_annotations(args.data)
    for case in cases:
        case["x3"] = features(case["image"], edge=False, size=config["image_size"])
        case["x4"] = features(case["image"], edge=True, size=config["image_size"])
        case["instances"] = annotations[case["id"]]
        foreground = case["instances"] > 0
        contact = touching_contact_target(case["instances"], radius=config["contact_radius"])
        target = np.stack([foreground, contact]).astype(np.uint8)
        case["target"] = torch.from_numpy(np.stack([
            cv2.resize(channel, (config["image_size"], config["image_size"]), interpolation=cv2.INTER_NEAREST)
            for channel in target
        ]).astype(np.float32))
    study = ROOT / "results" / STUDY; study.mkdir(parents=True, exist_ok=True)
    save_json(study / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "protocol_sha256": protocol_hash, "config": config})
    models = args.models or list(config["models"]); seeds = args.seeds if args.seeds is not None else config["seeds"]
    for seed in seeds:
        for name in models:
            run_one(config, name, config["models"][name], seed, cases, protocol_hash)
    save_json(study / "status.json", {"state": "complete", "protocol_sha256": protocol_hash})


if __name__ == "__main__":
    main()
