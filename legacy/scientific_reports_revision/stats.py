#!/usr/bin/env python3
"""Aggregate all runs -> bootstrap CIs, paired Wilcoxon vs proposed (Holm-corrected), effect sizes.
Writes outputs/revision/stats/{summary.csv,tests.csv,numbers.json}."""
import os, json, glob, numpy as np, csv
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "outputs", "revision"); ST = os.path.join(OUT, "stats"); os.makedirs(ST, exist_ok=True)
KEYS = ["dice", "iou", "precision", "recall", "hd95_um", "assd_um", "nsd2"]
rng = np.random.default_rng(0)

def load_runs():
    runs = {}
    for d in sorted(glob.glob(f"{OUT}/runs/*/result.json")):
        r = json.load(open(d)); name = r["name"]
        rows = list(csv.DictReader(open(os.path.join(os.path.dirname(d), "per_image.csv"))))
        runs[name] = dict(meta=r, per=rows)
    sp = os.path.join(OUT, "specialist_baselines.json")
    if os.path.exists(sp):
        for k, v in json.load(open(sp)).items(): runs[k] = dict(meta=dict(name=k, test=v["test"], params_M=None, best_epoch=None, train_time_s=None), per=v["per_image"])
    return runs

def ci(x, n=10000):
    x = np.asarray(x, float)
    if len(x) < 3 or np.allclose(x, x[0]): return float(x.mean()), float(x.mean()), float(x.mean())
    try: b = stats.bootstrap((x,), np.mean, n_resamples=n, method="BCa", random_state=0).confidence_interval; return float(x.mean()), float(b.low), float(b.high)
    except Exception: b = stats.bootstrap((x,), np.mean, n_resamples=n, method="percentile", random_state=0).confidence_interval; return float(x.mean()), float(b.low), float(b.high)

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b); return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))

def main():
    runs = load_runs(); names = list(runs)
    # ---- summary with CIs (per-image mean, CI over images) ----
    summ = {}
    for n in names:
        per = runs[n]["per"]; s = {}
        for k in KEYS:
            m, lo, hi = ci([float(r[k]) for r in per]); s[k] = m; s[k + "_lo"] = lo; s[k + "_hi"] = hi
        s["n"] = len(per); s["dice_micro"] = runs[n]["meta"]["test"].get("dice_micro"); s["params_M"] = runs[n]["meta"].get("params_M")
        s["best_epoch"] = runs[n]["meta"].get("best_epoch"); s["train_min"] = (runs[n]["meta"].get("train_time_s") or 0) / 60
        summ[n] = s
    # ---- proposed: seed aggregate ----
    seeds = [n for n in names if n.startswith("proposed_s")]
    if seeds:
        d = [summ[n]["dice"] for n in seeds]; summ["proposed_seeds"] = dict(n_seeds=len(seeds), dice_mean=float(np.mean(d)), dice_sd=float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                                                                     iou_mean=float(np.mean([summ[n]["iou"] for n in seeds])), seeds=seeds)
    # ---- paired tests vs proposed_s0 ----
    tests = []
    if "proposed_s0" in runs:
        ref = {r["file"]: float(r["dice"]) for r in runs["proposed_s0"]["per"]}
        for n in names:
            if n == "proposed_s0": continue
            per = {r["file"]: float(r["dice"]) for r in runs[n]["per"]}
            common = [f for f in ref if f in per]
            if len(common) < 10: continue
            a = np.array([ref[f] for f in common]); b = np.array([per[f] for f in common]); diff = a - b
            if np.allclose(diff, 0): p = 1.0; W = 0.0
            else: W, p = stats.wilcoxon(a, b, zero_method="wilcox")
            md, mlo, mhi = ci(diff)
            tests.append(dict(comparison=f"proposed_s0 vs {n}", n=len(common), mean_diff=md, diff_lo=mlo, diff_hi=mhi, median_diff=float(np.median(diff)),
                              wilcoxon_W=float(W), p_raw=float(p), cliffs_delta=cliffs_delta(a, b)))
        # Holm-Bonferroni
        ps = np.array([t["p_raw"] for t in tests]); order = np.argsort(ps); m = len(ps); adj = np.empty(m)
        run = 0
        for rank, i in enumerate(order):
            run = max(run, ps[i] * (m - rank)); adj[i] = min(1.0, run)
        for t, a in zip(tests, adj): t["p_holm"] = float(a)
    # ---- write ----
    with open(f"{ST}/summary.csv", "w") as f:
        cols = ["name"] + [c for c in next(iter(summ.values())).keys() if c != "seeds"]
        f.write(",".join(cols) + "\n")
        for n, s in summ.items():
            if n == "proposed_seeds": continue
            f.write(",".join([n] + [f"{s.get(c, ''):.4f}" if isinstance(s.get(c), float) else str(s.get(c, "")) for c in cols[1:]]) + "\n")
    with open(f"{ST}/tests.csv", "w") as f:
        if tests:
            f.write(",".join(tests[0].keys()) + "\n")
            for t in tests: f.write(",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in t.values()) + "\n")
    json.dump(dict(summary=summ, tests=tests), open(f"{ST}/numbers.json", "w"), indent=1)
    print(f"{'run':26s} {'Dice [95% CI]':22s} {'IoU':7s} {'Prec':6s} {'Rec':6s} {'HD95':6s} {'ASSD':6s} {'NSD2':6s} {'p_holm':8s} {'delta':6s}")
    tmap = {t["comparison"].split(" vs ")[1]: t for t in tests}
    for n, s in summ.items():
        if n == "proposed_seeds": continue
        t = tmap.get(n, {})
        print(f"{n:26s} {s['dice']:.4f} [{s['dice_lo']:.3f},{s['dice_hi']:.3f}]  {s['iou']:.4f} {s['precision']:.3f} {s['recall']:.3f} {s['hd95_um']:6.2f} {s['assd_um']:6.2f} {s['nsd2']:.3f} "
              f"{t.get('p_holm', float('nan')):8.2e} {t.get('cliffs_delta', float('nan')):6.2f}")
    if seeds: print("proposed over seeds:", summ["proposed_seeds"])
if __name__ == "__main__": main()
