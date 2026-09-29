#!/usr/bin/env python3
"""Budget-matched training/evaluation harness for the revision.

One script, one protocol, every model. Volume-wise split from
outputs/revision/volume_split.json. Metrics at NATIVE resolution.

python train_revision.py --name proposed --arch unetplusplus --encoder efficientnet-b7 \
    --edge gabor --scse 1 --sampling complexity --loss stabilized --epochs 40 --seed 0
"""
import os, sys, json, time, math, argparse, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SMD  = os.path.join(ROOT, "data", "All ZIP Files", "SuperMasterDataset")
SPLIT= os.path.join(ROOT, "outputs", "revision", "volume_split.json")
RUNS = os.path.join(ROOT, "outputs", "revision", "runs")
PX_UM = 0.2506

# ------------------------------------------------------------------ edge channels
class EdgeChannel:
    def __init__(self, kind, n_orient=8, n_scale=3, sigma=5.0, gamma=0.5, ksize=31, fmin=0.05, fmax=0.25):
        self.kind = kind; self.kernels = []
        if kind == "gabor":
            for th in np.linspace(0, np.pi, n_orient, endpoint=False):
                for f in np.linspace(fmin, fmax, n_scale):
                    k = cv2.getGaborKernel((ksize, ksize), sigma=sigma, theta=th, lambd=1.0/f,
                                           gamma=gamma, psi=0, ktype=cv2.CV_32F)
                    self.kernels.append(k / (np.abs(k).sum() + 1e-8))     # L1 norm (orig: k/k.sum(), unstable)
    def __call__(self, gray_u8):
        g = gray_u8.astype(np.float32) / 255.0
        if self.kind == "gabor":
            e = np.max(np.stack([np.abs(cv2.filter2D(g, cv2.CV_32F, k)) for k in self.kernels], -1), -1)
        elif self.kind == "sobel":
            e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        elif self.kind == "scharr":
            e = np.hypot(cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1))
        elif self.kind == "log":
            e = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2.0), cv2.CV_32F, ksize=5))
        elif self.kind == "canny":
            e = cv2.Canny(gray_u8, 50, 150).astype(np.float32) / 255.0
        else: raise ValueError(kind)
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        return e.astype(np.float32)

# ------------------------------------------------------------------ data
LEGACY_PREP = False
def load_pair(rec):
    m  = cv2.imread(f"{SMD}/{rec['orig_split']}/labels/{rec['file']}", cv2.IMREAD_UNCHANGED)
    m  = ((m > 127) if m.max() > 1 else (m > 0)).astype(np.uint8)
    if LEGACY_PREP:   # original pipeline: cv2.imread default flag -> 16-bit truncated to high byte (nearly black)
        return cv2.cvtColor(cv2.imread(f"{SMD}/{rec['orig_split']}/images/{rec['file']}"), cv2.COLOR_BGR2GRAY), m
    im = cv2.imread(f"{SMD}/{rec['orig_split']}/images/{rec['file']}", cv2.IMREAD_UNCHANGED).astype(np.float32)
    lo, hi = np.percentile(im, [1, 99.5]); im = np.clip((im - lo) / (hi - lo + 1e-6), 0, 1)
    return (im * 255).astype(np.uint8), m

class SliceDS(Dataset):
    def __init__(self, recs, size, edge, train):
        self.recs, self.size, self.edge, self.train = recs, size, edge, train
        nch = 4 if edge else 3
        mean = [0.485, 0.456, 0.406] + ([0.5] if edge else []); std = [0.229, 0.224, 0.225] + ([0.25] if edge else [])
        aug = [A.Resize(size, size)]
        if train:
            aug += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-15, 15), p=0.5),
                    A.RandomBrightnessContrast(0.2, 0.2, p=0.5)]
        aug += [A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()]
        self.tf = A.Compose(aug)
        self.cache = [load_pair(r) for r in recs]
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        g, m = self.cache[i]
        img = np.dstack([g, g, g])
        if self.edge: img = np.dstack([img, (self.edge(g) * 255).astype(np.uint8)])
        out = self.tf(image=img, mask=m)
        return out["image"], out["mask"][None].float(), i

def sample_weights(recs, cache, mode):
    if mode == "uniform": return None
    w = []
    for (g, m) in cache:
        r = m.mean()
        if mode == "complexity":                                   # paper's scheme, CORRECT binarisation
            if r > 0.001:
                wt = 2.0 + 3.0 * r
                if r < 0.05: wt *= 1.5
                e = cv2.Canny(m * 255, 50, 150); wt *= (1 + (e > 0).mean())
            else: wt = 0.3
        elif mode == "fgratio": wt = 1.0 + 3.0 * r
        w.append(wt)
    w = np.array(w); return (w / w.sum() * len(w)).tolist()

# ------------------------------------------------------------------ model
class Net(nn.Module):
    def __init__(self, arch, encoder, in_ch, scse, proj):
        super().__init__()
        kw = dict(encoder_name=encoder, encoder_weights="imagenet", classes=1, activation=None)
        smp_in = 3 if proj else in_ch
        if arch in ("unet", "unetplusplus"):
            kw["decoder_attention_type"] = "scse" if scse else None
            kw["decoder_channels"] = (256, 128, 64, 32, 16)
        cls = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN,
               "deeplabv3plus": smp.DeepLabV3Plus, "manet": smp.MAnet, "linknet": smp.Linknet}[arch]
        self.base = cls(in_channels=smp_in, **kw)
        self.proj = None
        if proj and in_ch == 4:                                    # paper's learnable 4->3 projection
            self.proj = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
                                      nn.Conv2d(16, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(True),
                                      nn.Conv2d(8, 3, 1))
    def forward(self, x):
        if self.proj is not None: x = self.proj(x)
        return self.base(x)

# ------------------------------------------------------------------ losses
def dice_per_sample(p, t, smooth=1.0):
    p = p.flatten(1); t = t.flatten(1)
    return 1 - ((2 * (p * t).sum(1) + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()

class StabilizedLoss(nn.Module):          # what trained the released model: 0.7 Dice + 0.3 adaptive BCE
    def forward(self, logits, t):
        logits = logits.clamp(-10, 10); p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        r = t.mean()
        pw = torch.tensor(float(min(50.0, max(1.0, (1 - r) / r))) if r > 0 else 1.0, device=t.device)
        bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)
        d = dice_per_sample(p, t)
        if not torch.isfinite(d): d = torch.ones((), device=t.device)
        if not torch.isfinite(bce): bce = torch.ones((), device=t.device)
        return 0.7 * d + 0.3 * bce

class DiceBCE(nn.Module):                 # plain, un-stabilised
    def forward(self, logits, t):
        return 0.5 * dice_per_sample(torch.sigmoid(logits), t) + 0.5 * F.binary_cross_entropy_with_logits(logits, t)

class TverskyBoundary(nn.Module):         # the loss the paper *described* (Tversky a=.3 b=.7 + boundary)
    def bnd(self, m):
        return F.max_pool2d(m, 3, 1, 1) - (-F.max_pool2d(-m, 3, 1, 1))
    def forward(self, logits, t):
        p = torch.sigmoid(logits.clamp(-10, 10))
        tp = (p * t).sum(); fp = (p * (1 - t)).sum(); fn = ((1 - p) * t).sum()
        tv = 1 - (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
        bd = dice_per_sample(self.bnd(p), self.bnd(t))
        r = t.mean(); pw = torch.tensor(float(min(50.0, max(1.0, (1 - r) / r))) if r > 0 else 1.0, device=t.device)
        bce = F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)
        return 0.5 * (0.5 * bce + 0.5 * dice_per_sample(p, t)) + 0.3 * bd + 0.2 * tv

# ------------------------------------------------------------------ metrics (native res)
def surface_dists(a, b):
    """symmetric surface distances (px) between binary masks a,b"""
    if a.sum() == 0 and b.sum() == 0: return np.array([0.0])
    if a.sum() == 0 or b.sum() == 0:  return np.array([np.hypot(*a.shape)])
    sa = a & ~ndimage.binary_erosion(a); sb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sb)[sa]; db = ndimage.distance_transform_edt(~sa)[sb]
    return np.concatenate([da, db])
def image_metrics(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum(); tn = (~pred & ~gt).sum()
    d = surface_dists(pred, gt)
    # NSD@2px: fraction of surface points within tolerance
    sp = pred & ~ndimage.binary_erosion(pred); sg = gt & ~ndimage.binary_erosion(gt)
    if sp.sum() and sg.sum():
        dp = ndimage.distance_transform_edt(~sg)[sp]; dg = ndimage.distance_transform_edt(~sp)[sg]
        nsd2 = ((dp <= 2).sum() + (dg <= 2).sum()) / (sp.sum() + sg.sum())
    else: nsd2 = float(sp.sum() == sg.sum())
    return dict(dice=2*tp/(2*tp+fp+fn+1e-9), iou=tp/(tp+fp+fn+1e-9), precision=tp/(tp+fp+1e-9),
                recall=tp/(tp+fn+1e-9), specificity=tn/(tn+fp+1e-9), fp_px=int(fp),
                hd95_um=float(np.percentile(d, 95) * PX_UM), assd_um=float(d.mean() * PX_UM), nsd2=float(nsd2),
                tp=int(tp), fp=int(fp), fn=int(fn))

@torch.no_grad()
def evaluate(model, ds, dev, bs=8, tta=False):
    model.eval(); rows = []
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    for x, _, idx in dl:
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.flip(torch.sigmoid(model(torch.flip(x, [3])).float()), [3])
                       + torch.flip(torch.sigmoid(model(torch.flip(x, [2])).float()), [2])) / 3
        for k, i in enumerate(idx.tolist()):
            g, m = ds.cache[i]; H, W = m.shape
            pr = F.interpolate(p[k:k+1], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0.5
            r = image_metrics(pr, m); r.update(file=ds.recs[i]["file"], volume=ds.recs[i]["volume"]); rows.append(r)
    return rows

def summarize(rows):
    keys = ["dice", "iou", "precision", "recall", "specificity", "hd95_um", "assd_um", "nsd2"]
    s = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    TP = sum(r["tp"] for r in rows); FP = sum(r["fp"] for r in rows); FN = sum(r["fn"] for r in rows)
    s["dice_micro"] = 2*TP/(2*TP+FP+FN); s["iou_micro"] = TP/(TP+FP+FN); s["n"] = len(rows)
    return s

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--arch", default="unetplusplus")
    ap.add_argument("--encoder", default="efficientnet-b7"); ap.add_argument("--edge", default="gabor")
    ap.add_argument("--scse", type=int, default=1); ap.add_argument("--proj", type=int, default=1)
    ap.add_argument("--sampling", default="complexity"); ap.add_argument("--loss", default="stabilized")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save_ckpt", type=int, default=0)
    ap.add_argument("--gabor", default="8,3,5.0,0.5,31", help="n_orient,n_scale,sigma,gamma,ksize")
    ap.add_argument("--fold", default="", help="optional: comma list of test volume ids for CV")
    ap.add_argument("--legacy_prep", type=int, default=0)
    ap.add_argument("--split_file", default=SPLIT,
                    help="JSON split/provenance file (default: inferred 855-slice split)")
    a = ap.parse_args()
    global LEGACY_PREP; LEGACY_PREP = bool(a.legacy_prep)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda"); out = os.path.join(RUNS, a.name); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f"{out}/config.json", "w"), indent=1)

    recs = json.load(open(os.path.expanduser(a.split_file)))
    if a.fold:
        tv = set(a.fold.split(",")); te = [r for r in recs if r["volume"] in tv]
        rest = [r for r in recs if r["volume"] not in tv]
        vols = sorted(set(r["volume"] for r in rest)); rng = random.Random(a.seed); rng.shuffle(vols)
        vv = set(vols[:max(1, len(vols)//6)])
        tr = [r for r in rest if r["volume"] not in vv]; va = [r for r in rest if r["volume"] in vv]
    else:
        tr = [r for r in recs if r["new_split"] == "train"]; va = [r for r in recs if r["new_split"] == "val"]
        te = [r for r in recs if r["new_split"] == "test"]
    no, ns, sg, gm, ks = a.gabor.split(","); 
    edge = EdgeChannel(a.edge, int(no), int(ns), float(sg), float(gm), int(ks)) if a.edge != "none" else None
    dtr = SliceDS(tr, a.size, edge, True); dva = SliceDS(va, a.size, edge, False); dte = SliceDS(te, a.size, edge, False)
    w = sample_weights(tr, dtr.cache, a.sampling)
    sampler = WeightedRandomSampler(w, len(w), replacement=True) if w else None
    dl = DataLoader(dtr, batch_size=a.bs, sampler=sampler, shuffle=sampler is None, num_workers=4, drop_last=True)
    print(f"[{a.name}] train {len(tr)} / val {len(va)} / test {len(te)}  edge={a.edge} sampling={a.sampling} loss={a.loss}", flush=True)

    model = Net(a.arch, a.encoder, 4 if edge else 3, bool(a.scse), bool(a.proj)).to(dev).to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    crit = {"stabilized": StabilizedLoss, "dicebce": DiceBCE, "tversky_boundary": TverskyBoundary}[a.loss]()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, final_div_factor=1000)
    best = -1; hist = []; t0 = time.time(); nonfinite = 0
    for ep in range(1, a.epochs + 1):
        model.train(); tl = 0; n = 0
        for x, y, _ in dl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).float(), y)
            if not torch.isfinite(loss): nonfinite += 1; opt.zero_grad(); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); sched.step()
            tl += loss.item() * len(x); n += len(x)
        vs = summarize(evaluate(model, dva, dev))
        hist.append(dict(epoch=ep, train_loss=tl / max(n, 1), val_dice=vs["dice"], val_iou=vs["iou"], t=time.time() - t0))
        if vs["dice"] > best:
            best = vs["dice"]; best_ep = ep
            torch.save(model.state_dict(), f"{out}/best.pt")
        print(f"  ep {ep:3d}/{a.epochs} loss {tl/max(n,1):.4f}  val dice {vs['dice']:.4f} (best {best:.4f} @ {best_ep})  {time.time()-t0:.0f}s", flush=True)
    model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    rows = evaluate(model, dte, dev); rows_tta = evaluate(model, dte, dev, tta=True)
    res = dict(name=a.name, config=vars(a), params_M=nparam, best_epoch=best_ep, best_val_dice=best,
               train_time_s=time.time() - t0, nonfinite_steps=nonfinite,
               test=summarize(rows), test_tta=summarize(rows_tta), history=hist)
    json.dump(res, open(f"{out}/result.json", "w"), indent=1)
    with open(f"{out}/per_image.csv", "w") as f:
        ks = list(rows[0].keys()); f.write(",".join(ks) + "\n")
        for r in rows: f.write(",".join(str(r[k]) for k in ks) + "\n")
    if not a.save_ckpt: os.remove(f"{out}/best.pt")
    t = res["test"]
    print(f"[{a.name}] DONE  test Dice {t['dice']:.4f} (micro {t['dice_micro']:.4f}) IoU {t['iou']:.4f} "
          f"P {t['precision']:.3f} R {t['recall']:.3f} HD95 {t['hd95_um']:.2f}um ASSD {t['assd_um']:.2f}um "
          f"NSD2 {t['nsd2']:.3f} | TTA dice {res['test_tta']['dice']:.4f} | {nparam:.1f}M params | best ep {best_ep} | {res['train_time_s']/60:.1f} min", flush=True)

if __name__ == "__main__": main()
