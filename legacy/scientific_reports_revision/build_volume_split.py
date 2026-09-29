#!/usr/bin/env python3
"""Recover source-volume provenance for every SuperMasterDataset image and
build a leakage-free VOLUME-WISE train/val/test split.

Method: images are full z-slices of per-organoid stacks, so (1) frame shape
identifies the imaging field; (2) within a shape group, adjacent z-slices are
strongly correlated, so connected components of a correlation graph recover
individual stacks. Whole components are assigned to one split.
"""
import os, glob, json, hashlib, collections
import numpy as np, cv2
ROOT = "data/All ZIP Files"
SMD  = f"{ROOT}/SuperMasterDataset"
OUT  = "outputs/revision"
os.makedirs(OUT, exist_ok=True)

# ---- known named volumes (AnnotatedDATA) keyed by shape -------------------
named = {}
for base in ["AnnotatedDATA_REVISED", "AnnotatedDATA"]:
    for d in sorted(os.listdir(f"{ROOT}/{base}")):
        p = f"{ROOT}/{base}/{d}"
        if not os.path.isdir(p) or "MASK" in d or "split" in d.lower(): continue
        fs = sorted(glob.glob(p + "/*.tif"))
        if not fs: continue
        im = cv2.imread(fs[0], cv2.IMREAD_UNCHANGED)
        named.setdefault(tuple(im.shape), d.replace(" Joy", ""))

# ---- load all SMD images -------------------------------------------------
recs = []
for split in ["train", "val", "test"]:
    for f in sorted(glob.glob(f"{SMD}/{split}/images/*.tif"),
                    key=lambda s: int(os.path.basename(s).split(".")[0])):
        n = os.path.basename(f)
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        m  = cv2.imread(f"{SMD}/{split}/labels/{n}", cv2.IMREAD_UNCHANGED)
        if m.dtype != np.uint8 or len(np.unique(m)) > 2:      # corrupt label (image saved as mask)
            print(f"  EXCLUDING corrupt label {split}/{n} dtype={m.dtype} nuniq={len(np.unique(m))}"); continue
        m = (m > 127) if m.max() > 1 else (m > 0)              # handle 0/255 AND 0/1 encodings
        t = cv2.resize(im.astype(np.float32), (64, 64)); t = (t - t.mean()) / (t.std() + 1e-6)
        recs.append(dict(orig_split=split, name=n, shape=tuple(im.shape),
                         has_nuc=bool(m.sum() > 0), fg=float(m.mean()),
                         thumb=t.ravel(), md5=hashlib.md5(im.tobytes()).hexdigest()))
print(f"loaded {len(recs)} images")

# exact duplicates across the whole set
dup = collections.Counter(r["md5"] for r in recs)
ndup = sum(v - 1 for v in dup.values() if v > 1)
print(f"exact pixel-duplicate images: {ndup}")

# ---- cluster within shape groups ------------------------------------------
by_shape = collections.defaultdict(list)
for i, r in enumerate(recs): by_shape[r["shape"]].append(i)

THR = 0.60          # edge threshold on 64x64 NCC; adjacent z-slices are >>0.6
groups = []          # list of lists of record indices
for shp, idx in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
    T = np.stack([recs[i]["thumb"] for i in idx]); C = (T @ T.T) / T.shape[1]
    n = len(idx); parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a in range(n):
        for b in range(a + 1, n):
            if C[a, b] > THR: parent[find(a)] = find(b)
    comp = collections.defaultdict(list)
    for a in range(n): comp[find(a)].append(idx[a])
    for c in comp.values(): groups.append(c)
    print(f"  shape {str(shp):12s} n={n:3d} -> {len(comp)} component(s) "
          f"{sorted(len(c) for c in comp.values())[::-1][:6]}  {named.get(shp,'(unnamed)')[-9:]}")

GROUP_BY_SHAPE = True
if GROUP_BY_SHAPE:
    groups = [list(v) for v in by_shape.values()]
groups.sort(key=len, reverse=True)
print(f"\n{len(groups)} source-volume groups from {len(by_shape)} frame shapes")

# ---- assign groups to splits (greedy, stratified by nucleus count) --------
rng = np.random.RandomState(0)
target = {"train": 0.60, "val": 0.15, "test": 0.25}
tot_img = len(recs); tot_nuc = sum(r["has_nuc"] for r in recs)
cnt = {k: [0, 0] for k in target}          # [images, nucleus-images]
assign = {}
order = list(range(len(groups))); rng.shuffle(order)
order.sort(key=lambda g: -len(groups[g]))  # big groups first
for g in order:
    gi = groups[g]; gn = sum(recs[i]["has_nuc"] for i in gi)
    # pick the split most below its target on nucleus-images (then images)
    def deficit(k): return (target[k] - cnt[k][1] / max(tot_nuc, 1)) + 0.3 * (target[k] - cnt[k][0] / tot_img)
    k = max(target, key=deficit)
    assign[g] = k; cnt[k][0] += len(gi); cnt[k][1] += gn

# ---- write split ---------------------------------------------------------
rows = []
for g, gi in enumerate(groups):
    shp = recs[gi[0]]["shape"]
    vol_id = f"V{g:02d}"
    for i in gi:
        r = recs[i]
        rows.append(dict(volume=vol_id, named_volume=named.get(shp, ""), shape=list(shp),
                         orig_split=r["orig_split"], file=r["name"], new_split=assign[g],
                         has_nucleus=r["has_nuc"], fg_fraction=round(r["fg"], 5)))
json.dump(rows, open(f"{OUT}/volume_split.json", "w"), indent=1)
with open(f"{OUT}/volume_split.csv", "w") as f:
    f.write("volume,named_volume,shape,orig_split,file,new_split,has_nucleus,fg_fraction\n")
    for r in rows:
        f.write(f"{r['volume']},{r['named_volume']},{r['shape'][0]}x{r['shape'][1]},{r['orig_split']},"
                f"{r['file']},{r['new_split']},{int(r['has_nucleus'])},{r['fg_fraction']}\n")

print("\nNEW VOLUME-WISE SPLIT")
for k in ["train", "val", "test"]:
    rr = [r for r in rows if r["new_split"] == k]
    vols = sorted(set(r["volume"] for r in rr))
    nn = sum(r["has_nucleus"] for r in rr)
    print(f"  {k:5s}: {len(rr):3d} images  {nn:3d} with nuclei ({100*nn/len(rr):.1f}%)  "
          f"{len(vols)} volumes")

# ---- leakage audit on the NEW split (nucleus-containing, 128x128) ---------
def thumbs128(sel):
    out = []
    for r in sel:
        f = f"{SMD}/{r['orig_split']}/images/{r['file']}"
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32)
        t = cv2.resize(im, (128, 128)); t = (t - t.mean()) / (t.std() + 1e-6)
        out.append(t.ravel())
    return np.stack(out)
def audit(key, tag):
    tr = [r for r in rows if r[key] == "train"]; te = [r for r in rows if r[key] == "test"]
    A = thumbs128(tr); B = thumbs128(te); C = (B @ A.T) / A.shape[1]; mx = C.max(1)
    print(f"\nLEAKAGE AUDIT [{tag}] test(n={len(te)}) vs train(n={len(tr)}), 128x128 NCC, all images have nuclei")
    for thr in [0.95, 0.90, 0.85, 0.80]:
        print(f"  test images with train match corr>{thr}: {(mx>thr).sum()}/{len(mx)} ({100*(mx>thr).mean():.1f}%)")
    print(f"  median max-corr = {np.median(mx):.3f}   max = {mx.max():.3f}")
    json.dump(dict(median_maxcorr=float(np.median(mx)), max_maxcorr=float(mx.max()),
                   frac_gt_0p95=float((mx>0.95).mean()), frac_gt_0p90=float((mx>0.90).mean()),
                   frac_gt_0p80=float((mx>0.80).mean()), n_test=len(te), n_train=len(tr)),
              open(f"{OUT}/leakage_audit_{tag}.json", "w"), indent=1)
audit("orig_split", "original_split"); audit("new_split", "volume_split")
import sys; sys.exit(0)
tr = [r for r in rows if r["new_split"] == "train" and r["has_nucleus"]]
te = [r for r in rows if r["new_split"] == "test" and r["has_nucleus"]]
A = thumbs128(tr); B = thumbs128(te); C = (B @ A.T) / A.shape[1]; mx = C.max(1)
print("\nLEAKAGE AUDIT (new split, nucleus-containing test vs train, 128x128 NCC)")
for thr in [0.95, 0.90, 0.85, 0.80]:
    print(f"  test images with train match corr>{thr}: {(mx>thr).sum()}/{len(mx)}")
print(f"  median max-corr = {np.median(mx):.3f}   max = {mx.max():.3f}")
json.dump(dict(median_maxcorr=float(np.median(mx)), max_maxcorr=float(mx.max()),
               frac_gt_0p95=float((mx>0.95).mean()), frac_gt_0p90=float((mx>0.90).mean()),
               n_test_nuc=len(te), n_train_nuc=len(tr)),
          open(f"{OUT}/leakage_audit_new_split.json", "w"), indent=1)
