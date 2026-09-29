#!/usr/bin/env python3
"""Cellpose-SAM (zero-shot + leakage-clean fine-tuned) on the NEW volume-wise test set."""
import os, sys, json, glob, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_revision import image_metrics, summarize, SMD, ROOT, load_pair
from cellpose import models
OUT = os.path.join(ROOT, "outputs", "revision")
recs = [r for r in json.load(open(f"{OUT}/volume_split.json")) if r["new_split"] == "test"]
# fine-tuned fold model that never saw each test volume
FT = {"V05": "fold3", "V09": "fold4"}          # MCF10A-10 -> fold3 held-out; MCF10A-16 -> fold4 held-out
def ft_path(v):
    d = os.path.join(ROOT, "outputs", "cellpose_cv", FT.get(v, "fold1"), "models"); return glob.glob(d + "/*")[0]
# Pin the exact built-in checkpoint instead of relying on the package default.
zs = models.CellposeModel(gpu=True, pretrained_model="cpsam_v2")
ft_models = {}
res = {"cellpose_zeroshot": [], "cellpose_finetuned": []}
for r in recs:
    g, m = load_pair(r)
    for tag, mdl in [("cellpose_zeroshot", zs), ("cellpose_finetuned", None)]:
        if mdl is None:
            k = FT.get(r["volume"], "fold1")
            if k not in ft_models: ft_models[k] = models.CellposeModel(gpu=True, pretrained_model=ft_path(r["volume"]))
            mdl = ft_models[k]
        masks = mdl.eval(g, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)[0]
        row = image_metrics(masks > 0, m); row.update(file=r["file"], volume=r["volume"], n_instances=int(masks.max())); res[tag].append(row)
    print(f"  {r['file']:8s} {r['volume']} zs {res['cellpose_zeroshot'][-1]['dice']:.3f}  ft {res['cellpose_finetuned'][-1]['dice']:.3f}", flush=True)
out = {k: dict(test=summarize(v), per_image=v) for k, v in res.items()}
json.dump(out, open(f"{OUT}/specialist_baselines.json", "w"), indent=1)
for k, v in out.items():
    t = v["test"]; print(f"{k:20s} n={t['n']} Dice {t['dice']:.4f} (micro {t['dice_micro']:.4f}) IoU {t['iou']:.4f} P {t['precision']:.3f} R {t['recall']:.3f} HD95 {t['hd95_um']:.2f} ASSD {t['assd_um']:.2f} NSD2 {t['nsd2']:.3f}")
