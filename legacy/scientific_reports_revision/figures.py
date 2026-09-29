#!/usr/bin/env python3
"""All manuscript figures -> outputs/revision/figures/. Robust to missing runs."""
import os, sys, json, glob, csv, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_revision import EdgeChannel, load_pair, SMD, ROOT
OUT = os.path.join(ROOT, "outputs", "revision"); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = dict(prop="#1565C0", base="#9E9E9E", cp="#2E7D32", abl="#E07B00")
recs = json.load(open(f"{OUT}/volume_split.json"))
def save(name): plt.tight_layout(); plt.savefig(f"{FIG}/{name}.pdf"); plt.savefig(f"{FIG}/{name}.png", dpi=180); plt.close()

# ---------------- 1. leakage audit ----------------
def thumb(r):
    im = cv2.imread(f"{SMD}/{r['orig_split']}/images/{r['file']}", -1).astype(np.float32); t = cv2.resize(im, (128, 128)); t = (t - t.mean()) / (t.std() + 1e-6); return t.ravel()
def maxcorr(key):
    tr = [r for r in recs if r[key] == "train"]; te = [r for r in recs if r[key] == "test"]
    A = np.stack([thumb(r) for r in tr]); B = np.stack([thumb(r) for r in te]); return ((B @ A.T) / A.shape[1]).max(1)
mo, mn = maxcorr("orig_split"), maxcorr("new_split")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax[0].hist(mo, bins=np.linspace(0, 1, 41), color="#C62828", alpha=.75, label=f"original split (n={len(mo)})")
ax[0].hist(mn, bins=np.linspace(0, 1, 41), color=C["prop"], alpha=.75, label=f"source-group split (n={len(mn)})")
ax[0].axvline(0.95, ls="--", c="k", lw=.8); ax[0].set_xlabel("max NCC of each test image vs. any training image"); ax[0].set_ylabel("test images"); ax[0].legend(fontsize=7, frameon=False)
ax[0].set_title("(a) Train/test similarity audit", loc="left", fontsize=9)
pairs = [("test", "110.tif", "train", "358.tif"), ("test", "48.tif", "train", "75.tif")]
ax[1].axis("off"); ax[1].set_title("(b) Leaked pair (original split)", loc="left", fontsize=9)
tiles = []
for sa, na, sb, nb in pairs[:1]:
    for s, n in [(sa, na), (sb, nb)]:
        im = cv2.imread(f"{SMD}/{s}/images/{n}", -1).astype(np.float32); lo, hi = np.percentile(im, [1, 99.5]); tiles.append(np.clip((im - lo) / (hi - lo), 0, 1))
ax[1].imshow(np.hstack([cv2.resize(t, (200, 200)) for t in tiles]), cmap="gray"); ax[1].text(100, 215, "test/110", ha="center", fontsize=7); ax[1].text(300, 215, "train/358", ha="center", fontsize=7)
save("fig_leakage_audit")

# ---------------- 2. real Gabor bank ----------------
e = EdgeChannel("gabor"); r = next(x for x in recs if x["new_split"] == "test" and x["volume"] == "V05")
g, m = load_pair(r)
fig = plt.figure(figsize=(7.2, 4.6)); gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.35])
for i in range(6):
    a = fig.add_subplot(gs[0, i]); k = e.kernels[i * 4]; a.imshow(k, cmap="RdBu_r", vmin=-np.abs(k).max(), vmax=np.abs(k).max()); a.axis("off")
    th = (i * 4) // 3; sc = (i * 4) % 3; a.set_title(f"$\\theta$={np.degrees(np.linspace(0, np.pi, 8, endpoint=False)[th]):.0f}°  $\\lambda$={[20, 6.67, 4][sc]:.3g}px", fontsize=7)
panels = [("input", g / 255.0), ("|resp.| kernel 1", np.abs(cv2.filter2D(g / 255.0, cv2.CV_32F, e.kernels[0]))), ("|resp.| kernel 13", np.abs(cv2.filter2D(g / 255.0, cv2.CV_32F, e.kernels[12]))),
          ("max of 24 (edge channel)", e(g)), ("Sobel", EdgeChannel("sobel")(g)), ("ground truth", m)]
for i, (t, im) in enumerate(panels):
    a = fig.add_subplot(gs[1, i]); a.imshow(im, cmap="gray" if t != "ground truth" else "viridis"); a.set_title(t, fontsize=7); a.axis("off")
fig.suptitle("Multi-scale Gabor edge channel: 6 of 24 kernels (top) and actual responses on a held-out test slice (bottom)", fontsize=8)
save("fig_gabor_bank")

# ---------------- results-dependent figures ----------------
if not os.path.exists(f"{OUT}/stats_clustered/numbers.json"): sys.exit(0)
N = json.load(open(f"{OUT}/stats_clustered/numbers.json")); S = N["summary"]
S_SLICE = json.load(open(f"{OUT}/stats/numbers.json"))["summary"]
def bar(names, labels, colors, fname, title, ylim=(0.0, 1.0)):
    names = [n for n in names if n in S]; 
    if not names: return
    x = np.arange(len(names)); d = [S[n]["dice"] for n in names]; lo = [S[n]["dice"] - S[n]["dice_lo"] for n in names]; hi = [S[n]["dice_hi"] - S[n]["dice"] for n in names]
    fig, ax = plt.subplots(figsize=(max(3.6, 0.55 * len(names) + 1), 2.8))
    ax.bar(x, d, yerr=[lo, hi], color=[colors[n] for n in names], capsize=2, width=0.7)
    for i, v in enumerate(d): ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([labels.get(n, n) for n in names], rotation=35, ha="right", fontsize=7); ax.set_ylim(*ylim); ax.set_ylabel("source-group macro Dice (95% CI)"); ax.set_title(title, loc="left", fontsize=9)
    save(fname)
L = {"proposed": "Full config. (EffB7 UNet++)", "base_unet_r34": "U-Net R34", "base_unet_r50": "U-Net R50", "base_attunet_r50": "Attention U-Net R50", "base_fpn_r50": "FPN R50",
     "base_deeplabv3plus_r50": "DeepLabV3+ R50", "base_manet_r50": "MA-Net R50", "base_unetpp_r50": "UNet++ R50", "cellpose_zeroshot": "Cellpose-SAM (zero-shot)", "cellpose_finetuned": "Cellpose-SAM (fine-tuned)",
     "abl_no_gabor": "w/o Gabor channel", "abl_no_scse": "w/o SCSE", "abl_uniform_sampling": "w/o weighted sampling", "abl_loss_dicebce": "plain Dice+BCE loss", "abl_loss_tversky_boundary": "Tversky+boundary loss",
     "abl_plain_unet": "plain U-Net decoder", "vanilla_effb7_unetpp": "vanilla EffB7 UNet++", "edge_sobel": "Sobel channel", "edge_scharr": "Scharr channel", "edge_log": "LoG channel", "edge_canny": "Canny channel",
     "ladder_resnet34": "ResNet-34", "ladder_resnet50": "ResNet-50", "ladder_effb0": "EffNet-B0", "ladder_effb3": "EffNet-B3", "ladder_effb5": "EffNet-B5",
     "gabor_orient4": "4 orientations", "gabor_orient16": "16 orientations", "gabor_scale2": "2 scales", "gabor_scale5": "5 scales", "gabor_sigma2": "σ=2", "gabor_sigma10": "σ=10"}
col = {n: C["base"] for n in L}; col.update({"proposed": C["prop"], "cellpose_zeroshot": C["cp"], "cellpose_finetuned": C["cp"]})
for n in L:
    if n.startswith(("abl_", "edge_", "ladder_", "gabor_", "vanilla")): col[n] = C["abl"]
bar(["proposed", "base_attunet_r50", "base_unetpp_r50", "base_deeplabv3plus_r50", "base_fpn_r50", "base_manet_r50", "base_unet_r50", "base_unet_r34", "cellpose_zeroshot", "cellpose_finetuned"], L, col, "fig_sota", "Budget-matched comparison (206 slices in 7 source groups)", (0.5, 1.0))
bar(["proposed", "abl_no_gabor", "edge_sobel", "edge_scharr", "edge_log", "edge_canny", "abl_no_scse", "abl_uniform_sampling", "abl_loss_dicebce", "abl_loss_tversky_boundary", "abl_plain_unet", "vanilla_effb7_unetpp"], L, col, "fig_ablation", "Component ablation (one change from the full configuration)", (0.5, 1.0))
bar(["proposed", "gabor_orient4", "gabor_orient16", "gabor_scale2", "gabor_scale5", "gabor_sigma2", "gabor_sigma10"], L, col, "fig_gabor_sensitivity", "Gabor parameter sensitivity", (0.5, 1.0))
# encoder ladder: dice vs params
lad = [n for n in ["ladder_resnet34", "ladder_resnet50", "ladder_effb0", "ladder_effb3", "ladder_effb5", "proposed"] if n in S]
if lad:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for n in lad:
        meta = S_SLICE["proposed_s0" if n == "proposed" else n]
        params = meta["params_M"]
        ax.errorbar(params, S[n]["dice"], yerr=[[S[n]["dice"] - S[n]["dice_lo"]], [S[n]["dice_hi"] - S[n]["dice"]]], fmt="o", color=C["prop"] if n == "proposed" else C["abl"], capsize=2)
        ax.annotate("EffNet-B7" if n == "proposed" else L[n], (params, S[n]["dice"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("parameters (M)"); ax.set_ylabel("source-group macro Dice"); ax.set_title("Encoder ladder (UNet++ decoder, all additions fixed)", loc="left", fontsize=8); save("fig_encoder_ladder")
# training curves (seeds)
seeds = sorted(glob.glob(f"{OUT}/runs/proposed_s*/result.json"))
if seeds:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for p in seeds:
        h = json.load(open(p))["history"]; ep = [x["epoch"] for x in h]
        ax[0].plot(ep, [x["train_loss"] for x in h], lw=1); ax[1].plot(ep, [x["val_dice"] for x in h], lw=1, label=os.path.basename(os.path.dirname(p)))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation Dice (held-out source groups)"); ax[1].set_ylim(0.5, 1); ax[1].legend(fontsize=7, frameon=False)
    ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left"); save("fig_training_curves")
# Seven independent source-group means per method (not 206 correlated slices).
names = [n for n in ["proposed", "base_attunet_r50", "base_unetpp_r50", "base_deeplabv3plus_r50", "base_fpn_r50", "base_manet_r50", "base_unet_r50", "base_unet_r34", "cellpose_zeroshot"] if n in S]
data = [[S[n]["per_group"][g]["dice"] for g in sorted(S[n]["per_group"])] for n in names]
if data:
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
    for b, n in zip(bp["boxes"], names): b.set_facecolor(col[n]); b.set_alpha(.7)
    for i, values in enumerate(data, 1): ax.scatter(np.full(len(values), i), values, s=9, c="black", alpha=.65, zorder=3)
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels([L[n] for n in names], rotation=30, ha="right", fontsize=7); ax.set_ylabel("source-group mean Dice"); ax.set_ylim(0.75, 1.0); save("fig_boxplot")
print("figures written:", sorted(os.listdir(FIG)))
