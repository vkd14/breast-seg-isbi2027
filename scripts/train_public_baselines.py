#!/usr/bin/env python3
"""Matched-budget public architecture baselines requested during peer review."""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from breastseg.core import SegmentationLoss, features, metrics, sha256
from breastseg.data import load_bbbc039
from train_public_study import Cached, save_json, seed_all, state_digest


ROOT = Path(__file__).resolve().parents[1]


class BaselineNet(torch.nn.Module):
    def __init__(self, spec, pretrained=False):
        super().__init__()
        cls = getattr(smp, spec["architecture"])
        kwargs = dict(encoder_name=spec["encoder"], encoder_weights="imagenet" if pretrained else None,
                      in_channels=3, classes=1, activation=None)
        if spec.get("attention") is not None:
            kwargs["decoder_attention_type"] = spec["attention"]
        self.model = cls(**kwargs)

    def forward(self, x):
        return self.model(x)


@torch.inference_mode()
def evaluate(model, loader, cases, full=False, out=None):
    model.eval()
    rows = []
    if out:
        (out / "predictions").mkdir(parents=True, exist_ok=True)
    for x, _, indices in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x.cuda(non_blocking=True))
        probs = logits.float().sigmoid().cpu().numpy()[:, 0]
        for prob, index in zip(probs, indices.tolist()):
            case = cases[index]
            gt = case["mask"].astype(bool)
            pred = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) > .5
            result = metrics(pred, gt) if full else {
                "dice": float(2 * (pred & gt).sum() / (pred.sum() + gt.sum())) if pred.any() or gt.any() else 1.
            }
            rows.append(dict(id=case["id"], group=case["group"], **result))
            if out and not cv2.imwrite(str(out / "predictions" / f"{case['id']}.png"), pred.astype(np.uint8) * 255):
                raise IOError("Failed prediction write")
    if len(rows) != len(cases):
        raise RuntimeError("Incomplete evaluation")
    return rows


def run_one(cfg, name, spec, seed, cases, protocol_hash):
    study = "public_architecture_baselines"
    out = ROOT / "results" / study / f"{name}_s{seed}"
    ckptdir = ROOT / "checkpoints" / study / f"{name}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ckptdir.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text())
        if old["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Protocol differs from completed run")
        print("SKIP completed", name, seed, flush=True)
        return
    if (out / "history.json").exists():
        raise RuntimeError(f"Partial run exists: {out}; inspect before restarting")

    start = time.time()
    seed_all(seed)
    model = BaselineNet(spec, pretrained=True).cuda()
    initial_hash = state_digest(model)
    train = [c for c in cases if c["split"] == "train"]
    val = [c for c in cases if c["split"] == "val"]
    test = [c for c in cases if c["split"] == "test"]
    generator = torch.Generator().manual_seed(seed)

    def loader(items, augment=False):
        return DataLoader(Cached(items, edge=False, augment=augment), batch_size=cfg["batch_size"],
                          shuffle=augment, generator=generator if augment else None, num_workers=0,
                          pin_memory=True, drop_last=False)

    train_loader, val_loader = loader(train, True), loader(val)
    loss_fn = SegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["learning_rate"] * .01)
    history, best, best_epoch = [], -1., None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            losses.append((loss.item(), len(x)))
        rows = evaluate(model, val_loader, val)
        score = float(np.mean([r["dice"] for r in rows]))
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model_state_dict": model.state_dict(), "baseline": name, "spec": spec,
                        "seed": seed, "epoch": epoch, "val_dice": best,
                        "protocol_sha256": protocol_hash}, ckptdir / "best.pt")
        record = dict(epoch=epoch, train_loss=sum(loss * n for loss, n in losses) / sum(n for _, n in losses),
                      val_dice=score, best_val_dice=best, learning_rate=optimizer.param_groups[0]["lr"],
                      seconds=time.time() - t0)
        history.append(record)
        save_json(out / "history.json", history)
        save_json(ROOT / "results" / study / "status.json", dict(state="training", run=out.name, **record))
        print(json.dumps(dict(run=out.name, **record)), flush=True)
        scheduler.step()

    checkpoint = torch.load(ckptdir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = evaluate(model, loader(test), test, full=True, out=out)
    with (out / "test_per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(baseline=name, spec=spec, seed=seed, n_test=len(rows), epochs=cfg["epochs"],
                   best_epoch=best_epoch, best_val_dice=best,
                   parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial_hash,
                   checkpoint_sha256=sha256(ckptdir / "best.pt"), protocol_sha256=protocol_hash,
                   seconds=time.time() - start,
                   mean={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("id", "group")})
    save_json(out / "summary.json", summary)
    print("COMPLETE " + json.dumps(summary), flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "public_architecture_baselines.json")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol_hash = sha256(args.config)
    cv2.setNumThreads(1)
    torch.set_num_threads(8)
    cases = load_bbbc039(args.data)
    for case in cases:
        case["x4"] = features(case["image"], edge=False, size=cfg["image_size"])
        case["y"] = torch.from_numpy(cv2.resize(case["mask"], (cfg["image_size"], cfg["image_size"]),
                                                   interpolation=cv2.INTER_NEAREST).astype(np.float32)[None])
    study = ROOT / "results" / "public_architecture_baselines"
    save_json(study / "split_manifest.json", [{k: v for k, v in c.items() if k not in ("image", "mask", "x4", "y")}
                                                for c in cases])
    save_json(study / "environment.json", {
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "protocol_sha256": protocol_hash, "config": cfg,
        "archives": {p.name: sha256(p) for p in args.data.glob("*.zip")},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        [Path(__file__), ROOT / "src/breastseg/core.py", ROOT / "src/breastseg/data.py"]},
    })
    selected = args.models or list(cfg["models"])
    for name in selected:
        if name not in cfg["models"]:
            raise ValueError(name)
    for seed in (args.seeds if args.seeds is not None else cfg["seeds"]):
        for name in selected:
            run_one(cfg, name, cfg["models"][name], seed, cases, protocol_hash)
    save_json(study / "status.json", {"state": "complete", "protocol_sha256": protocol_hash})


if __name__ == "__main__":
    main()
