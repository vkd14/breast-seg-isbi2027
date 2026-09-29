#!/usr/bin/env python3
"""Qualitative panel: image / GT / prediction / error map for held-out test slices (proposed_s0 best.pt)."""
import os, sys, json, csv, numpy as np, cv2, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_revision import Net, SliceDS, EdgeChannel, ROOT
OUT = os.path.join(ROOT, "outputs", "revision"); FIG = os.path.join(OUT, "figures"); dev = torch.device("cuda")
cfg = json.load(open(f"{OUT}/runs/proposed_s0/config.json"))
per = {r["file"]: float(r["dice"]) for r in csv.DictReader(open(f"{OUT}/runs/proposed_s0/per_image.csv"))}
recs = [r for r in json.load(open(f"{OUT}/volume_split.json")) if r["new_split"] == "test"]
# pick: best, median, a dense one, worst -> one per distinct volume where possible
recs_sorted = sorted(recs, key=lambda r: per[r["file"]])
picks = [recs_sorted[-1], recs_sorted[len(recs_sorted)//2], recs_sorted[len(recs_sorted)//4], recs_sorted[0]]
labels = ["best", "median", "lower quartile", "worst"]
edge = EdgeChannel("gabor"); ds = SliceDS(picks, cfg["size"], edge, False)
net = Net(cfg["arch"], cfg["encoder"], 4, bool(cfg["scse"]), bool(cfg["proj"])).to(dev); net.load_state_dict(torch.load(f"{OUT}/runs/proposed_s0/best.pt", map_location=dev)); net.eval()
fig, ax = plt.subplots(4, 5, figsize=(10, 8.2))
with torch.no_grad():
    for i in range(4):
        x, _, _ = ds[i]; g, m = ds.cache[i]; H, W = m.shape
        p = torch.sigmoid(net(x[None].to(dev)).float()); p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        pr = p > 0.5; gt = m > 0
        err = np.zeros((H, W, 3)); err[pr & gt] = (0.2, 0.8, 0.2); err[pr & ~gt] = (0.9, 0.2, 0.2); err[~pr & gt] = (0.2, 0.3, 0.95)
        panels = [(g, "gray", f"{labels[i]}: {picks[i]['volume']} / {picks[i]['file']}"), (gt, "viridis", "ground truth"), (p, "magma", "probability"), (pr, "viridis", f"prediction (Dice {per[picks[i]['file']]:.3f})"), (err, None, "TP green / FP red / FN blue")]
        for j, (im, cm, t) in enumerate(panels):
            a = ax[i, j]; a.imshow(im, cmap=cm) if cm else a.imshow(im); a.set_title(t, fontsize=7); a.axis("off")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_qualitative.pdf"); plt.savefig(f"{FIG}/fig_qualitative.png", dpi=150); print("ok", [(p["file"], round(per[p["file"]], 3)) for p in picks])
