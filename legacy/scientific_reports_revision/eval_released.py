#!/usr/bin/env python3
"""Re-evaluate the RELEASED checkpoint (best_enhanced_model.pt) with the ORIGINAL preprocessing
on the ORIGINAL test split, stratified by whether each test image has a near-duplicate in the
original training set (leakage stratum). Also reports boundary metrics and the per-volume view."""
import os, sys, json, numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_revision import image_metrics, summarize, SMD, ROOT
CK = os.path.join(ROOT, "configs", "2d_configs", "best_enhanced_model.pt")
OUT = os.path.join(ROOT, "outputs", "revision"); dev = torch.device("cuda")

# --- original Gabor (orig normalisation k/k.sum()) ---
ks = []
for th in np.linspace(0, np.pi, 8, endpoint=False):
    for f in np.linspace(0.05, 0.25, 3):
        k = cv2.getGaborKernel((31, 31), sigma=5.0, theta=th, lambd=1/f, gamma=0.5, psi=0, ktype=cv2.CV_32F); ks.append(k / k.sum())
def gabor(gray):
    g = gray.astype(np.float32) / 255.0
    e = np.max(np.stack([np.abs(cv2.filter2D(g, cv2.CV_32F, k)) for k in ks], -1), -1)
    return (e - e.min()) / (e.max() - e.min() + 1e-8)
tf = A.Compose([A.Resize(512, 512), A.Normalize(mean=[.485,.456,.406,.5], std=[.229,.224,.225,.25], max_pixel_value=255.0), ToTensorV2()])

# --- model: EffB7 UNet++ scse + input_proj, load released weights ---
class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.base_model = smp.UnetPlusPlus(encoder_name="efficientnet-b7", encoder_weights=None, in_channels=3, classes=1,
                                        activation=None, decoder_attention_type="scse", decoder_channels=(256,128,64,32,16))
        s.input_proj = nn.Sequential(nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
                                     nn.Conv2d(16,8,3,padding=1), nn.BatchNorm2d(8), nn.ReLU(True), nn.Conv2d(8,3,1))
    def forward(s, x): return s.base_model(s.input_proj(x))
m = M().to(dev); sd = torch.load(CK, map_location="cpu", weights_only=False)["model_state_dict"]
missing, unexpected = m.load_state_dict(sd, strict=False); print("missing", len(missing), "unexpected", len(unexpected)); m.eval()

recs = json.load(open(f"{OUT}/volume_split.json"))
tr = [r for r in recs if r["orig_split"] == "train"]; te = [r for r in recs if r["orig_split"] == "test"]
def thumb(r):
    im = cv2.imread(f"{SMD}/{r['orig_split']}/images/{r['file']}", -1).astype(np.float32)
    t = cv2.resize(im, (128, 128)); t = (t - t.mean()) / (t.std() + 1e-6); return t.ravel()
A_ = np.stack([thumb(r) for r in tr]); B_ = np.stack([thumb(r) for r in te]); mx = ((B_ @ A_.T) / A_.shape[1]).max(1)

rows = []
with torch.no_grad():
    for i, r in enumerate(te):
        img = cv2.imread(f"{SMD}/test/images/{r['file']}")                # ORIGINAL: 8-bit BGR conversion
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x4 = np.dstack([img, (gabor(gray) * 255).astype(np.uint8)])
        mk = cv2.imread(f"{SMD}/test/labels/{r['file']}", -1); mk = ((mk > 127) if mk.max() > 1 else (mk > 0)).astype(np.uint8)
        x = tf(image=x4, mask=mk)["image"][None].to(dev)
        p = torch.sigmoid(m(x))[0, 0].cpu().numpy()
        H, W = mk.shape
        pn = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        p512 = p > 0.5; m512 = cv2.resize(mk, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
        rn = image_metrics(pn, mk); r5 = image_metrics(p512, m512)
        rn.update(file=r["file"], volume=r["volume"], maxcorr_train=float(mx[i]), leaked=bool(mx[i] > 0.95),
                  dice_512=r5["dice"], iou_512=r5["iou"]); rows.append(rn)
        print(f"  {r['file']:8s} vol {r['volume']} corr {mx[i]:.3f} dice(native) {rn['dice']:.3f} dice(512) {r5['dice']:.3f}", flush=True)

def S(sel):
    s = summarize(sel); s["dice_512"] = float(np.mean([r["dice_512"] for r in sel])); s["iou_512"] = float(np.mean([r["iou_512"] for r in sel])); return s
res = dict(all=S(rows), leaked=S([r for r in rows if r["leaked"]]), not_leaked=S([r for r in rows if not r["leaked"]]),
           per_volume={v: S([r for r in rows if r["volume"] == v]) for v in sorted(set(r["volume"] for r in rows))})
json.dump(dict(summary=res, per_image=rows), open(f"{OUT}/eval_released_ckpt.json", "w"), indent=1)
print("\nRELEASED CHECKPOINT on ORIGINAL test split (n=%d)" % len(rows))
for k in ["all", "leaked", "not_leaked"]:
    s = res[k]; print(f"  {k:11s} n={s['n']:3d}  Dice@512 {s['dice_512']:.4f}  Dice@native {s['dice']:.4f}  IoU {s['iou']:.4f}  P {s['precision']:.3f} R {s['recall']:.3f}  HD95 {s['hd95_um']:.2f}um ASSD {s['assd_um']:.2f}um NSD2 {s['nsd2']:.3f}")
