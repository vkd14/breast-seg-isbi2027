#!/usr/bin/env python3
"""Frozen-checkpoint evaluation on complete stated BBBC038, BBBC039, or TNBC sets.

No training, threshold selection, image selection, or morphology is performed.
Results are semantic nucleus foreground, not instance segmentation scores.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "data/bbbc038"
OUT = ROOT / "outputs/bbbc038_zero_shot"
CHECKPOINTS = {
    "released_2d": PROJECT / "data&code/configs/2d_configs/best_enhanced_model.pt",
    "revised_2d_s0": PROJECT / "data&code/outputs/revision/runs/proposed_s0/best.pt",
    "cellpose_sam": Path.home() / ".cellpose/models/cpsam_v2",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b7", encoder_weights=None, in_channels=3,
            classes=1, activation=None, decoder_attention_type="scse",
            decoder_channels=(256, 128, 64, 32, 16))
        self.input_proj = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.Conv2d(8, 3, 1))

    def forward(self, x):
        return self.base_model(self.input_proj(x))


def preprocess(rgb, revised):
    if not revised and rgb.dtype == np.uint16:
        rgb = (rgb // 256).astype(np.uint8)  # Match released cv2.IMREAD_COLOR loader.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if revised:
        lo, hi = np.percentile(gray.astype(np.float32), [1, 99.5])
        gray = (255 * np.clip((gray.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    responses = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        for f in np.linspace(.05, .25, 3):
            k = cv2.getGaborKernel((31, 31), 5., theta, 1 / f, .5, 0, ktype=cv2.CV_32F)
            k /= np.abs(k).sum() + 1e-8 if revised else k.sum()
            responses.append(np.abs(cv2.filter2D(gray.astype(np.float32) / 255, cv2.CV_32F, k)))
    edge = np.max(responses, axis=0)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    x = np.dstack([rgb, (edge * 255).astype(np.uint8)])
    x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    x = (x - np.array([.485, .456, .406, .5], dtype=np.float32)) / np.array([.229, .224, .225, .25], dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0)


def decode_rle(encoded, h, w):
    a = np.zeros(h * w, dtype=np.uint8)
    if encoded.strip():
        runs = np.array(encoded.split(), dtype=np.int64).reshape(-1, 2)
        for start, length in runs:
            if start < 1 or start - 1 + length > h * w:
                raise ValueError("Invalid RLE")
            a[start - 1:start - 1 + length] = 1
    return a.reshape((h, w), order="F").astype(bool)


def metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    denom = 2 * tp + fp + fn
    sp, sg = p & ~ndi.binary_erosion(p), g & ~ndi.binary_erosion(g)
    if sp.any() and sg.any():
        dp = ndi.distance_transform_edt(~sg)[sp]
        dg = ndi.distance_transform_edt(~sp)[sg]
        bp, br = float((dp <= 2).mean()), float((dg <= 2).mean())
        bf1 = 2 * bp * br / (bp + br) if bp + br else 0.
        nsd = float(((dp <= 2).sum() + (dg <= 2).sum()) / (len(dp) + len(dg)))
        hd95 = float(np.percentile(np.r_[dp, dg], 95))
    elif not sp.any() and not sg.any():
        bf1, nsd, hd95 = 1., 1., 0.
    else:
        bf1, nsd, hd95 = 0., 0., float(np.hypot(*g.shape))
    return dict(dice=2 * tp / denom if denom else 1., iou=tp / (tp + fp + fn) if tp + fp + fn else 1.,
                precision=tp / (tp + fp) if tp + fp else float(not g.any()), recall=tp / (tp + fn) if tp + fn else float(not p.any()),
                boundary_f1_2px=bf1, surface_dice_2px=nsd, hd95_px=hd95,
                empty_prediction=int(not p.any()), tp=tp, fp=fp, fn=fn)


def load_cases(dataset="bbbc038"):
    if dataset == "tnbc":
        archive = zipfile.ZipFile(DATA / "TNBC_NucleiSegmentation.zip")
        names = sorted(n for n in archive.namelist() if "/Slide_" in n and n.endswith(".png") and "__MACOSX" not in n)
        cases = []
        for name in names:
            rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask_name = name.replace("/Slide_", "/GT_")
            gt = cv2.imdecode(np.frombuffer(archive.read(mask_name), np.uint8), cv2.IMREAD_UNCHANGED) > 0
            assert gt.ndim == 2 and gt.shape == rgb.shape[:2]
            # Binary masks do not establish the true number of touching instances.
            cases.append((Path(name).stem, rgb, gt, -1))
        assert len(cases) == 50
        return cases
    if dataset == "bbbc039":
        iz, mz = zipfile.ZipFile(DATA / "images.zip"), zipfile.ZipFile(DATA / "masks.zip")
        metaz = zipfile.ZipFile(DATA / "metadata.zip")
        test_names = metaz.read("metadata/test.txt").decode().splitlines()
        cases = []
        for name in sorted(n.strip() for n in test_names if n.strip()):
            sid = Path(name).stem
            raw = cv2.imdecode(np.frombuffer(iz.read(f"images/{sid}.tif"), np.uint8), cv2.IMREAD_UNCHANGED)
            mask = cv2.imdecode(np.frombuffer(mz.read(f"masks/{sid}.png"), np.uint8), cv2.IMREAD_COLOR)
            gt = mask[..., 2] > 0  # Author's decoding: RGB red channel is nuclear foreground.
            rgb = np.repeat(raw[..., None], 3, axis=2)
            assert raw.dtype == np.uint16 and gt.shape == raw.shape
            cases.append((sid, rgb, gt, -1))
        assert len(cases) == 50
        return cases
    labels = defaultdict(list)
    with open(DATA / "stage1_solution.csv", newline="") as f:
        for row in csv.DictReader(f):
            labels[row["ImageId"]].append(row)
    archive = zipfile.ZipFile(DATA / "stage1_test.zip")
    names = sorted(n for n in archive.namelist() if "/images/" in n and n.endswith(".png"))
    cases = []
    for name in names:
        sid = Path(name).stem
        rgb = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
        if rgb is None or sid not in labels:
            raise ValueError(f"Missing image or annotation: {sid}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        gt = np.zeros((h, w), dtype=bool)
        for row in labels[sid]:
            assert (int(row["Height"]), int(row["Width"])) == (h, w)
            gt |= decode_rle(row["EncodedPixels"], h, w)
        cases.append((sid, rgb, gt, len(labels[sid])))
    assert len(cases) == 65, f"Expected all 65 test images, found {len(cases)}"
    assert set(labels) == {c[0] for c in cases}
    return cases


def summarize(rows):
    result = {}
    for method in sorted({r["method"] for r in rows}):
        rr = [r for r in rows if r["method"] == method]
        summary = {"n": len(rr)}
        for key in ("dice", "iou", "precision", "recall", "boundary_f1_2px", "surface_dice_2px", "hd95_px", "seconds"):
            a = np.array([r[key] for r in rr])
            summary[key] = float(a.mean())
            summary[key + "_median"] = float(np.median(a))
        a = np.array([r["dice"] for r in rr])
        rng = np.random.default_rng(20260928)
        ci = np.percentile(a[rng.integers(len(a), size=(2000, len(a)))].mean(1), [2.5, 97.5])
        summary["dice_descriptive_image_bootstrap_ci95"] = ci.tolist()
        summary["empty_predictions"] = sum(r["empty_prediction"] for r in rr)
        result[method] = summary
    return result


def main():
    global DATA, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bbbc038", "bbbc039", "tnbc"], default="bbbc038")
    ap.add_argument("--checkpoint-released", type=Path)
    ap.add_argument("--checkpoint-revised", type=Path)
    ap.add_argument("--checkpoint-cellpose", type=Path)
    ap.add_argument("--methods", nargs="+", default=["released_2d", "revised_2d_s0", "otsu_border_polarity", "cellpose_sam"])
    args = ap.parse_args()
    for option, method in ((args.checkpoint_released, "released_2d"), (args.checkpoint_revised, "revised_2d_s0"), (args.checkpoint_cellpose, "cellpose_sam")):
        if option is not None:
            CHECKPOINTS[method] = option
    DATA = ROOT / "data" / args.dataset
    OUT = ROOT / "outputs" / (args.dataset + "_zero_shot")
    torch.manual_seed(20260928)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset)
    manifest = {"dataset": args.dataset, "n": len(cases), "selected_before_evaluation": "Complete official test partition for BBBC038/039; all 50 images for TNBC v1.1; no exclusions", "training_on_external_data": False,
                "threshold": .5, "prediction_resolution": "512x512 then bilinear probability resize to native dimensions", "metric_space": "native pixels, foreground union", "tta": False,
                "prediction_resolution_note": "512x512 and threshold 0.5 apply to the two author networks only; Cellpose/Otsu use native images",
                "cellpose_settings": {"flow_threshold": .4, "cellprob_threshold": 0., "normalize": True, "do_3D": False, "channel_axis": -1, "augment": False},
                "otsu_settings": "Native grayscale; invert foreground if median border intensity exceeds Otsu threshold; no cleanup",
                "boundary_tolerance_pixels": 2, "empty_hd95_convention": "image diagonal if only one mask empty", "bootstrap": "Descriptive image bootstrap, not source-cluster inference",
                "python": platform.python_version(), "torch": torch.__version__, "smp": smp.__version__, "device": torch.cuda.get_device_name(0),
                "files": {p.name: sha256(p) for p in DATA.iterdir() if p.is_file()},
                "checkpoints": {k: {"path": str(v), "sha256": sha256(v)} for k, v in CHECKPOINTS.items() if k in args.methods},
                "cases": [{"id": sid, "height": rgb.shape[0], "width": rgb.shape[1], "annotated_instances": count} for sid, rgb, gt, count in cases]}
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for method in args.methods:
        model = None
        if method in ("released_2d", "revised_2d_s0"):
            model = Model()
            state = torch.load(CHECKPOINTS[method], map_location="cpu", weights_only=False)
            if method == "released_2d":
                state = state["model_state_dict"]
            else:
                state = {k.replace("base.", "base_model.", 1) if k.startswith("base.") else k.replace("proj.", "input_proj.", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            del state
            model.cuda().eval()
        elif method == "cellpose_sam":
            from cellpose import models
            model = models.CellposeModel(gpu=True, pretrained_model=str(CHECKPOINTS[method]))
        (OUT / "predictions" / method).mkdir(parents=True, exist_ok=True)
        for i, (sid, rgb, gt, count) in enumerate(cases):
            torch.cuda.synchronize()
            started = time.perf_counter()
            if method in ("released_2d", "revised_2d_s0"):
                x = preprocess(rgb, revised=method == "revised_2d_s0").cuda()
                with torch.inference_mode():
                    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                pred = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) > .5
            elif method == "cellpose_sam":
                inst, _, _ = model.eval(rgb, do_3D=False, channel_axis=-1, normalize=True, flow_threshold=.4, cellprob_threshold=0., augment=False)
                pred = inst > 0
            elif method == "otsu_border_polarity":
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                border = np.r_[gray[0], gray[-1], gray[:, 0], gray[:, -1]]
                pred = mask > 0
                if np.median(border) > threshold:
                    pred = ~pred
            else:
                raise ValueError(method)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            assert pred.shape == gt.shape
            row = dict(method=method, image_id=sid, height=gt.shape[0], width=gt.shape[1], n_instances=count,
                       seconds=seconds, **metrics(pred, gt))
            rows.append(row)
            cv2.imwrite(str(OUT / "predictions" / method / f"{sid}.png"), pred.astype(np.uint8) * 255)
            with open(OUT / "per_image.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{method} {i+1}/{len(cases)} Dice={row['dice']:.4f} BF1={row['boundary_f1_2px']:.4f} {seconds:.2f}s", flush=True)
        print(json.dumps({method: summarize(rows)[method]}, indent=2), flush=True)
        (OUT / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
