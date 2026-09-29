#!/usr/bin/env python3
"""Build the prediction-sample gallery: private confocal fields, public BBBC039, TNBC transfer.

Private predictions use the retained Scientific-Reports-revision checkpoint
(breast-3d .../runs/proposed_s0/best.pt); public predictions use the new
matched-budget checkpoints in this repository. Every panel is a real image,
a real reference mask and a real model output. Nothing is synthesised.
"""
import os, sys, json, csv, argparse
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PRIV = "/home/vdasoju/Desktop/breast-3d/data&code"
FIG = os.path.join(REPO, "reports", "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

def err_rgb(pred, gt, img):
    base = np.dstack([img]*3) * 0.55
    base[pred & gt] = (0.15, 0.85, 0.25)
    base[pred & ~gt] = (0.95, 0.15, 0.15)
    base[~pred & gt] = (0.20, 0.45, 1.00)
    return np.clip(base, 0, 1)

def norm01(a):
    a = a.astype(np.float32); lo, hi = np.percentile(a, [1, 99.5])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

# ---------------------------------------------------------------- private
def private_panel():
    sys.path.insert(0, os.path.join(PRIV, "src", "revision"))
    from train_revision import Net, SliceDS, EdgeChannel, load_pair
    run = os.path.join(PRIV, "outputs", "revision", "runs", "proposed_s0")
    cfg = json.load(open(f"{run}/config.json"))
    per = {(r["volume"], r["file"]): float(r["dice"]) for r in csv.DictReader(open(f"{run}/per_image.csv"))}
    recs = [r for r in json.load(open(os.path.join(PRIV, "outputs", "revision", "volume_split.json")))
            if r["new_split"] == "test"]
    order = sorted(recs, key=lambda r: per[(r["volume"], r["file"])])
    picks = [order[-1], order[len(order)//2], order[len(order)//4], order[0]]
    tags = ["best", "median", "lower quartile", "worst"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge = EdgeChannel("gabor"); ds = SliceDS(picks, cfg["size"], edge, False)
    net = Net(cfg["arch"], cfg["encoder"], 4, bool(cfg["scse"]), bool(cfg["proj"])).to(dev)
    net.load_state_dict(torch.load(f"{run}/best.pt", map_location=dev)); net.eval()
    fig, ax = plt.subplots(4, 4, figsize=(8.6, 8.8))
    with torch.no_grad():
        for i in range(4):
            x, _, _ = ds[i]; g, m = ds.cache[i]; H, W = m.shape
            p = torch.sigmoid(net(x[None].to(dev)).float())
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            img = norm01(g); pr = p > 0.5; gt = m > 0
            d = per[(picks[i]["volume"], picks[i]["file"])]
            for j, (im, cm, t) in enumerate([
                    (img, "gray", f"{tags[i]} — field {picks[i]['volume']}, {picks[i]['file']}"),
                    (gt, "gray", "manual annotation"),
                    (pr, "gray", f"prediction (Dice {d:.3f})"),
                    (err_rgb(pr, gt, img), None, "green TP / red FP / blue FN")]):
                a = ax[i, j]; a.imshow(im, cmap=cm) if cm else a.imshow(im)
                a.set_title(t, fontsize=7); a.axis("off")
    fig.suptitle("Private confocal breast-organoid fields — held-out acquisition fields\n"
                 "(EfficientNet-B7 UNet++ + Gabor bundle, Scientific Reports revision checkpoint)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out = f"{FIG}/private_predictions.png"; plt.savefig(out); plt.savefig(out.replace(".png", ".pdf")); plt.close()
    return out, [(t, picks[i]["volume"], picks[i]["file"], per[(picks[i]["volume"], picks[i]["file"])])
                 for i, t in enumerate(tags)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--private-only", action="store_true"); a = ap.parse_args()
    out, rows = private_panel()
    print("wrote", out)
    for r in rows: print("  ", r)
