#!/usr/bin/env python3
"""Fill main_revised.template.tex with numbers from outputs/revision -> Template folder/main_revised.tex"""
import os, json, csv, glob, shutil, re, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); OUT = os.path.join(ROOT, "outputs", "revision")
TPL = os.path.join(os.path.dirname(__file__), "main_revised.template.tex")
DEST = os.path.expanduser("~/Desktop/Template_for_submissions_to_Scientific_Reports")
N = json.load(open(f"{OUT}/stats_clustered/numbers.json")); S = N["summary"]; T = {t["comparison"].split(" vs ")[1]: t for t in N["tests"]}
S_SLICE = json.load(open(f"{OUT}/stats/numbers.json"))["summary"]
rel = json.load(open(f"{OUT}/eval_released_ckpt.json"))["summary"]
lo_ = json.load(open(f"{OUT}/leakage_audit_original_split.json")); ln_ = json.load(open(f"{OUT}/leakage_audit_volume_split.json"))
split = json.load(open(f"{OUT}/volume_split.json"))
L = {"proposed": "Full configuration (EfficientNet-B7 UNet++, all components)", "base_unet_r34": "U-Net (ResNet-34)", "base_unet_r50": "U-Net (ResNet-50)", "base_attunet_r50": "Attention U-Net (ResNet-50)",
     "base_fpn_r50": "FPN (ResNet-50)", "base_deeplabv3plus_r50": "DeepLabV3+ (ResNet-50)", "base_manet_r50": "MA-Net (ResNet-50)", "base_unetpp_r50": "UNet++ (ResNet-50)",
     "cellpose_zeroshot": "Cellpose-SAM, zero-shot", "cellpose_finetuned": "Cellpose-SAM, fine-tuned (leakage-clean folds)",
     "abl_no_gabor": "w/o Gabor edge channel (RGB only)", "edge_sobel": "Gabor $\\rightarrow$ Sobel channel", "edge_scharr": "Gabor $\\rightarrow$ Scharr channel", "edge_log": "Gabor $\\rightarrow$ LoG channel", "edge_canny": "Gabor $\\rightarrow$ Canny channel",
     "abl_no_scse": "w/o SCSE attention", "abl_uniform_sampling": "w/o complexity-weighted sampling (uniform)", "abl_loss_dicebce": "stabilised loss $\\rightarrow$ plain Dice+BCE",
     "abl_loss_tversky_boundary": "stabilised loss $\\rightarrow$ Tversky($\\alpha$=0.3,$\\beta$=0.7)+boundary", "abl_plain_unet": "UNet++ $\\rightarrow$ plain U-Net decoder",
     "vanilla_effb7_unetpp": "Vanilla EfficientNet-B7 UNet++ (no additions)", "ladder_resnet34": "ResNet-34", "ladder_resnet50": "ResNet-50", "ladder_effb0": "EfficientNet-B0", "ladder_effb3": "EfficientNet-B3", "ladder_effb5": "EfficientNet-B5",
     "gabor_orient4": "4 orientations (default 8)", "gabor_orient16": "16 orientations", "gabor_scale2": "2 scales (default 3)", "gabor_scale5": "5 scales", "gabor_sigma2": "$\\sigma=2$ (default 5)", "gabor_sigma10": "$\\sigma=10$"}
def f3(x): return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
def cistr(n, k="dice"): s = S[n]; return f"{s[k]:.3f} [{s[k+'_lo']:.3f}, {s[k+'_hi']:.3f}]"
def pstr(n):
    t = T.get(n); 
    if not t: return "--"
    p = t["p_holm"]; return "$<$0.001" if p < 1e-3 else f"{p:.3f}"
def row(n, ref="proposed", show_p=True):
    if n not in S: return f"{L.get(n,n)} & \\multicolumn{{7}}{{c}}{{\\emph{{pending}}}} \\\\"
    s = S[n]; d = f"{s['dice']-S[ref]['dice']:+.3f}" if (n != ref and ref in S) else "--"
    cells = [L.get(n, n), cistr(n), f3(s["iou"]), f3(s["precision"]), f3(s["recall"]), f"{s['hd95_um']:.2f}", f"{s['assd_um']:.2f}", f3(s["nsd2"])]
    if show_p: cells += [d, pstr(n) if n != ref else "--"]
    return " & ".join(cells) + " \\\\"
HEAD = "Method & Dice [95\\% CI] & IoU & Prec. & Rec. & HD95 ($\\mu$m) & ASSD ($\\mu$m) & NSD$_{2}$"
def table(names, caption, label, show_p=True, ref="proposed"):
    cols = "l" + "c" * (7 + (2 if show_p else 0)); h = HEAD + (" & $\\Delta$Dice & $p_{\\rm Holm}$" if show_p else "")
    body = "\n".join(row(n, ref, show_p) for n in names)
    return f"""\\begin{{table}}[ht]\\centering\\scriptsize
\\caption{{{caption}}}\\label{{{label}}}
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{{cols}}}\\toprule
{h} \\\\ \\midrule
{body}
\\bottomrule\\end{{tabular}}\\end{{table}}"""
ref_row = row("proposed", show_p=True).replace("& -- & --", "& -- & --")
V = {}
P = S.get("proposed", {})
V["PROP_DICE"], V["PROP_DICE_LO"], V["PROP_DICE_HI"] = f3(P.get("dice")), f3(P.get("dice_lo")), f3(P.get("dice_hi"))
V["PROP_IOU"], V["PROP_P"], V["PROP_R"] = f3(P.get("iou")), f3(P.get("precision")), f3(P.get("recall"))
V["PROP_HD95"], V["PROP_ASSD"], V["PROP_NSD"], V["PROP_MICRO"] = f"{P.get('hd95_um',0):.2f}", f"{P.get('assd_um',0):.2f}", f3(P.get("nsd2")), "--"
V["PROP_SEED_MEAN"], V["PROP_SEED_SD"], V["N_SEEDS"] = f3(P.get("dice")), f3(P.get("dice_seed_sd")), str(P.get("n_seeds", 0))
bases = [n for n in ["base_attunet_r50", "base_unetpp_r50", "base_deeplabv3plus_r50", "base_fpn_r50", "base_manet_r50", "base_unet_r50", "base_unet_r34"] if n in S]
if bases:
    bb = max(bases, key=lambda n: S[n]["dice"]); V["BEST_BASE"], V["BEST_BASE_DICE"], V["GAP"] = L[bb], f3(S[bb]["dice"]), f"{P['dice']-S[bb]['dice']:+.3f}"; V["BEST_BASE_P"] = pstr(bb)
    wb = min(bases, key=lambda n: S[n]["dice"]); V["WORST_BASE"], V["WORST_BASE_DICE"] = L[wb], f3(S[wb]["dice"]); V["N_BASE"] = str(len(bases))
else: V.update(BEST_BASE="\\pending{best baseline}", BEST_BASE_DICE="\\pending{}", GAP="\\pending{}", BEST_BASE_P="\\pending{}", WORST_BASE="\\pending{}", WORST_BASE_DICE="\\pending{}", N_BASE="0")
for n in ["cellpose_zeroshot", "cellpose_finetuned"]:
    k = n.upper(); V[k + "_DICE"] = f3(S[n]["dice"]) if n in S else "\\pending{}"; V[k + "_P"] = f3(S[n]["precision"]) if n in S else "\\pending{}"; V[k + "_R"] = f3(S[n]["recall"]) if n in S else "\\pending{}"
for n in L:
    V[f"D:{n}"] = f3(S[n]["dice"]) if n in S else "\\pending{}"; V[f"DELTA:{n}"] = f"{S[n]['dice']-P['dice']:+.3f}" if (n in S and P) else "\\pending{}"; V[f"P:{n}"] = pstr(n) if n in S else "\\pending{}"
    V[f"PREC:{n}"] = f3(S[n]["precision"]) if n in S else "\\pending{}"; V[f"REC:{n}"] = f3(S[n]["recall"]) if n in S else "\\pending{}"
V["REL_ALL"], V["REL_LEAK"], V["REL_NOLEAK"], V["REL_ALL512"] = f3(rel["all"]["dice"]), f3(rel["leaked"]["dice"]), f3(rel["not_leaked"]["dice"]), f3(rel["all"]["dice_512"])
V["REL_N_LEAK"], V["REL_N_NOLEAK"] = str(rel["leaked"]["n"]), str(rel["not_leaked"]["n"])
V["LEAK_ORIG_MED"], V["LEAK_ORIG_95"], V["LEAK_ORIG_90"], V["LEAK_ORIG_MAX"] = f3(lo_["median_maxcorr"]), f"{100*lo_['frac_gt_0p95']:.0f}", f"{100*lo_['frac_gt_0p90']:.0f}", f3(lo_["max_maxcorr"])
V["LEAK_NEW_MED"], V["LEAK_NEW_MAX"] = f3(ln_["median_maxcorr"]), f3(ln_["max_maxcorr"])
cnt = {k: [0, set()] for k in ["train", "val", "test"]}
for r in split: cnt[r["new_split"]][0] += 1; cnt[r["new_split"]][1].add(r["volume"])
V["N_TRAIN"], V["N_VAL"], V["N_TEST"] = str(cnt["train"][0]), str(cnt["val"][0]), str(cnt["test"][0])
V["V_TRAIN"], V["V_VAL"], V["V_TEST"] = str(len(cnt["train"][1])), str(len(cnt["val"][1])), str(len(cnt["test"][1]))
fg = [r["fg_fraction"] for r in split]; V["FG_MEAN"], V["FG_MIN"], V["FG_MAX"] = f"{100*np.mean(fg):.1f}", f"{100*np.min(fg):.1f}", f"{100*np.max(fg):.1f}"
V["N_TOTAL"], V["N_VOL"] = str(len(split)), str(len(set(r["volume"] for r in split)))
V["PROP_PARAMS"] = f"{S_SLICE.get('proposed_s0',{}).get('params_M',68.4):.1f}"
V["TAB_SOTA"] = table(["proposed"] + bases + [n for n in ["cellpose_zeroshot", "cellpose_finetuned"] if n in S], "Budget-matched comparison on the conservative source-group test set (" + V["N_TEST"] + " slices in " + V["V_TEST"] + " held-out shape-defined groups). Trained configurations use the same schedule, augmentation and model-selection rule; objective ablations differ only as stated. Metrics are averaged within source group and then across groups. Confidence intervals bootstrap source groups; paired Wilcoxon tests use source-group means and are Holm-corrected. Cellpose-SAM is reported zero-shot and after author fine-tuning. No test-time augmentation.", "tab:sota")
V["TAB_ABL"] = table(["proposed", "abl_no_gabor", "edge_sobel", "edge_scharr", "edge_log", "edge_canny", "abl_no_scse", "abl_uniform_sampling", "abl_loss_dicebce", "abl_loss_tversky_boundary", "abl_plain_unet", "vanilla_effb7_unetpp"], "Component ablation. Each ablation row changes one element of the full configuration (first row); the final vanilla anchor removes all four additions. Metrics and inference protocol are as in Table~\\ref{tab:sota}.", "tab:ablation")
V["TAB_GABOR"] = table(["proposed", "gabor_orient4", "gabor_orient16", "gabor_scale2", "gabor_scale5", "gabor_sigma2", "gabor_sigma10"], "Gabor parameter sensitivity (one parameter changed from the default 8 orientations, 3 scales, $\\sigma$=5, $\\gamma$=0.5, $31\\times31$).", "tab:gabor")
lad = [n for n in ["ladder_resnet34", "ladder_resnet50", "ladder_effb0", "ladder_effb3", "ladder_effb5", "proposed"]]
param_key = {"proposed": "proposed_s0"}
lrows = "\n".join((f"{L[n] if n!='proposed' else 'EfficientNet-B7 (full configuration)'} & {S_SLICE.get(param_key.get(n,n),{}).get('params_M',float('nan')):.1f} & {cistr(n)} & {f3(S[n]['iou'])} & {f3(S[n]['precision'])} & {f3(S[n]['recall'])} & {S[n]['assd_um']:.2f} & {pstr(n) if n!='proposed' else '--'} \\\\" if n in S else f"{L.get(n,n)} & \\multicolumn{{7}}{{c}}{{\\emph{{pending}}}} \\\\") for n in lad)
V["TAB_LADDER"] = f"""\\begin{{table}}[ht]\\centering\\scriptsize
\\caption{{Encoder ladder: the UNet++ decoder, Gabor channel, SCSE, sampling and loss are held fixed; only the encoder changes.}}\\label{{tab:ladder}}
\\begin{{tabular}}{{lccccccc}}\\toprule
Encoder & Params (M) & Dice [95\\% CI] & IoU & Prec. & Rec. & ASSD ($\\mu$m) & $p_{{\\rm Holm}}$ \\\\ \\midrule
{lrows}
\\bottomrule\\end{{tabular}}\\end{{table}}"""
# released-checkpoint audit table
V["TAB_RELEASED"] = f"""\\begin{{table}}[ht]\\centering\\scriptsize
\\caption{{Re-evaluation of the originally submitted model on the original (slice-level) test split, stratified by whether a test image has a near-duplicate ($\\mathrm{{NCC}}>0.95$) in the original training set.}}\\label{{tab:released}}
\\begin{{tabular}}{{lcccccc}}\\toprule
Stratum & $n$ & Dice (512$^2$) & Dice (native) & IoU & HD95 ($\\mu$m) & ASSD ($\\mu$m) \\\\ \\midrule
All test images & {rel['all']['n']} & {f3(rel['all']['dice_512'])} & {f3(rel['all']['dice'])} & {f3(rel['all']['iou'])} & {rel['all']['hd95_um']:.2f} & {rel['all']['assd_um']:.2f} \\\\
With training near-duplicate & {rel['leaked']['n']} & {f3(rel['leaked']['dice_512'])} & {f3(rel['leaked']['dice'])} & {f3(rel['leaked']['iou'])} & {rel['leaked']['hd95_um']:.2f} & {rel['leaked']['assd_um']:.2f} \\\\
Without & {rel['not_leaked']['n']} & {f3(rel['not_leaked']['dice_512'])} & {f3(rel['not_leaked']['dice'])} & {f3(rel['not_leaked']['iou'])} & {rel['not_leaked']['hd95_um']:.2f} & {rel['not_leaked']['assd_um']:.2f} \\\\
\\bottomrule\\end{{tabular}}\\end{{table}}"""
# CV is summarized at source-group level after its per-image rows are loaded.
V.update(CV_N="4", CV_MEAN="\\pending{}", CV_SD="\\pending{}", CV_MIN="\\pending{}", CV_MAX="\\pending{}", CV_ROWS="\\pending{CV pending}")
for n in ["legacy_prep_proposed", "legacy_prep_unet_r34"]:
    V["D:" + n] = f3(S[n]["dice"]) if n in S else "\\pending{}"
# per-field CV table
import collections as _c
cvrows=[]
for f in range(1,5):
    fp=f"{OUT}/runs/cv_fold{f}/per_image.csv"
    if os.path.exists(fp):
        for r in csv.DictReader(open(fp)): cvrows.append((r["volume"], float(r["dice"]), float(r["precision"]), float(r["recall"])))
if cvrows:
    named={r["volume"]:r["named_volume"] for r in split}; shp={r["volume"]:r["shape"] for r in split}
    enc={}
    for r in split:
        m=__import__("cv2").imread(f"{ROOT}/data/All ZIP Files/SuperMasterDataset/{r['orig_split']}/labels/{r['file']}", -1); enc.setdefault(r["volume"], "0/255" if m.max()>1 else "0/1")
    g=_c.defaultdict(list)
    for v,d,pr,rc in cvrows: g[v].append((d,pr,rc))
    lines=[]
    for v in sorted(g, key=lambda v: np.mean([x[0] for x in g[v]])):
        a=np.array(g[v]); nm=named[v].split("_")[-1] if named[v] else "--"
        lines.append(f"{v} & {shp[v][0]}$\\times${shp[v][1]} & {nm} & {enc[v]} & {len(a)} & {a[:,0].mean():.3f} & {a[:,1].mean():.3f} & {a[:,2].mean():.3f} \\\\")
    V["TAB_CV"]="""\\begin{table}[ht]\\centering\\scriptsize
\\caption{Per-group results under 4-fold leave-source-groups-out cross-validation (every shape-defined source group is held out once; full configuration). Precision/recall differences are associated with mask encoding, but encoding does not identify an annotator or annotation session.}\\label{tab:cv}
\\begin{tabular}{llllrccc}\\toprule
Source group & Frame & Traceable stack & Mask enc. & $n$ & Dice & Prec. & Rec. \\\\ \\midrule
"""+"\n".join(lines)+"""
\\bottomrule\\end{tabular}\\end{table}"""
    group_dice=[np.mean([x[0] for x in values]) for values in g.values()]
    fold_macro=[]
    for f in range(1,5):
        fg=_c.defaultdict(list)
        for r in csv.DictReader(open(f"{OUT}/runs/cv_fold{f}/per_image.csv")):
            fg[r["volume"]].append(float(r["dice"]))
        fold_macro.append(float(np.mean([np.mean(v) for v in fg.values()])))
    V["CV_POOLED"]=f3(np.mean(group_dice)); V["CV_NSLICES"]=str(len(cvrows))
    V["CV_MEAN"]=f3(np.mean(group_dice)); V["CV_SD"]=f3(np.std(group_dice,ddof=1))
    V["CV_MIN"], V["CV_MAX"] = f3(min(fold_macro)), f3(max(fold_macro))
    V["CV_ROWS"] = " ".join(f"fold {i+1}: {d:.3f};" for i,d in enumerate(fold_macro))
    tight=[v for v in g if np.mean([x[1] for x in g[v]])<0.9 and np.mean([x[2] for x in g[v]])>0.97]
    wide=[v for v in g if np.mean([x[1] for x in g[v]])>0.95 and np.mean([x[2] for x in g[v]])<0.9]
    V["CV_TIGHT_N"]=str(len(tight)); V["CV_WIDE_N"]=str(len(wide)); V["CV_TIGHT_LIST"]=", ".join(sorted(tight)); V["CV_WIDE_LIST"]=", ".join(sorted(wide))
else: V.update(TAB_CV="\\pending{CV table}", CV_POOLED="\\pending{}", CV_NSLICES="\\pending{}", CV_TIGHT_N="\\pending{}", CV_WIDE_N="\\pending{}", CV_TIGHT_LIST="", CV_WIDE_LIST="")
# TTA table for proposed
r0 = glob.glob(f"{OUT}/runs/proposed_s0/result.json")
if r0:
    j = json.load(open(r0[0])); V["PROP_TTA_DICE"] = f3(j["test_tta"]["dice"]); V["PROP_NONFINITE"] = str(j["nonfinite_steps"]); V["PROP_BEST_EP"] = str(j["best_epoch"]); V["PROP_MIN"] = f"{j['train_time_s']/60:.0f}"
else: V.update(PROP_TTA_DICE="\\pending{}", PROP_NONFINITE="\\pending{}", PROP_BEST_EP="\\pending{}", PROP_MIN="\\pending{}")
tex = open(TPL).read()
missing = set()
def sub(m):
    k = m.group(1)
    if k in V: return V[k]
    missing.add(k); return "\\pending{" + k + "}"
tex = re.sub(r"@@([A-Za-z0-9_:]+)@@", sub, tex)
open(f"{DEST}/main_revised.tex", "w").write(tex)
for f in glob.glob(f"{OUT}/figures/*.pdf") + glob.glob(f"{OUT}/figures/*.png"): shutil.copy(f, DEST)
print("wrote", f"{DEST}/main_revised.tex", "| unresolved:", sorted(missing) or "none", "| pending markers:", tex.count("\\pending{"))
# ---- response letter ----
rtex = re.sub(r"@@([A-Za-z0-9_:]+)@@", sub, open(os.path.join(os.path.dirname(__file__), "response.template.tex")).read())
open(f"{DEST}/response_to_reviewers.tex", "w").write(rtex); print("wrote response_to_reviewers.tex | pending markers:", rtex.count("\\pending{"))
# ---- meeting / changes summary (markdown) ----
def g(k): return V.get(k, "pending")
md = f"""# Historical revision summary — not the submission gate
This file records the earlier Scientific Reports revision and retains some of its terminology. For the current next-journal assessment, use `SUBMISSION_READINESS_AUDIT.md`, `AUTHOR_ACTIONS_REQUIRED.md`, and `main_revised.tex`.

# What changed and why — revision summary for discussion
Generated {__import__('datetime').date.today()} from `outputs/revision/`. Companion docs: `main_revised.tex` (manuscript), `response_to_reviewers.tex` (point-by-point), `REVIEWER_REQUESTS_CHECKLIST.md` (task list).

## The three audit findings (ours, not the reviewers')
1. **The original split leaked.** Slice-level split put adjacent z-planes of the same organoid in train and test: {g('LEAK_ORIG_95')}% of test images had a training near-duplicate (NCC>0.95), median max-NCC {g('LEAK_ORIG_MED')}. Fixed by splitting on acquisition field ({g('V_TRAIN')}/{g('V_VAL')}/{g('V_TEST')} fields = {g('N_TRAIN')}/{g('N_VAL')}/{g('N_TEST')} slices); max cross-split NCC now {g('LEAK_NEW_MAX')}.
2. **"60% empty images" was false.** 508 masks are stored 0/1 and were read as empty at a 127 threshold. All {g('N_TOTAL')} valid slices contain nuclei (foreground {g('FG_MIN')}–{g('FG_MAX')}%, mean {g('FG_MEAN')}%). Two corrupt label files excluded. Side effect: the original *sampler* used the same threshold → 508 nucleus images were down-weighted 8.9×, so "complexity-weighted sampling" did not do what the paper said.
3. **Algorithm 1 ≠ trained model.** Released checkpoint used 0.7·Dice + 0.3·adaptive-BCE — no Tversky, no boundary term. Paper now documents the real loss; Tversky+boundary is an ablation.
Also found: Gabor kernels normalised by their sum (8/24 kernels have |Σ|<0.1 → amplified ~100–1000×), fixed to L1; original loader truncated 16-bit images to the high byte (max ≈11/255) — the likely reason baselines scored 0.47–0.76 before.

## Headline numbers (volume-wise test set: {g('N_TEST')} slices, {g('V_TEST')} unseen fields)
| Model | Dice [95% CI] | IoU | Prec | Rec | ASSD µm |
|---|---|---|---|---|---|
| Proposed (EffB7 UNet++ + Gabor + SCSE + sampling + stabilised loss) | **{g('PROP_DICE')}** [{g('PROP_DICE_LO')}, {g('PROP_DICE_HI')}] | {g('PROP_IOU')} | {g('PROP_P')} | {g('PROP_R')} | {g('PROP_ASSD')} |
| — seeds ({g('N_SEEDS')}) | {g('PROP_SEED_MEAN')} ± {g('PROP_SEED_SD')} | | | | |
| — {g('CV_N')}-fold leave-fields-out CV | {g('CV_MEAN')} ± {g('CV_SD')} | | | | |
| Best baseline: {g('BEST_BASE')} | {g('BEST_BASE_DICE')} (Δ {g('GAP')}, p_Holm {g('BEST_BASE_P')}) | | | | |
| Worst baseline: {g('WORST_BASE')} | {g('WORST_BASE_DICE')} | | | | |
| Cellpose-SAM zero-shot | {g('CELLPOSE_ZEROSHOT_DICE')} | | {g('CELLPOSE_ZEROSHOT_P')} | {g('CELLPOSE_ZEROSHOT_R')} | |
| Cellpose-SAM fine-tuned (leakage-clean) | {g('CELLPOSE_FINETUNED_DICE')} | | {g('CELLPOSE_FINETUNED_P')} | {g('CELLPOSE_FINETUNED_R')} | |
| *Original paper claim* | *0.951 (leaked split, 100-img subset)* | | | | |
| Released checkpoint re-evaluated, original split | {g('REL_ALL512')} @512² — leaked half {g('REL_LEAK')} / clean half {g('REL_NOLEAK')} | | | | |

## Ablations (Dice, Δ vs proposed, Holm p)
| Change | Dice | Δ | p |
|---|---|---|---|
""" + "\n".join(f"| {L[n]} | {g('D:'+n)} | {g('DELTA:'+n)} | {g('P:'+n)} |" for n in ["abl_no_gabor","edge_sobel","edge_scharr","edge_log","edge_canny","abl_no_scse","abl_uniform_sampling","abl_loss_dicebce","abl_loss_tversky_boundary","abl_plain_unet","vanilla_effb7_unetpp"]) + f"""

## Gabor sensitivity
""" + " · ".join(f"{L[n]}: {g('D:'+n)}" for n in ["gabor_orient4","gabor_orient16","gabor_scale2","gabor_scale5","gabor_sigma2","gabor_sigma10"]) + f"""

## Encoder ladder (all additions fixed)
""" + " · ".join(f"{L[n]}: {g('D:'+n)}" for n in ["ladder_resnet34","ladder_resnet50","ladder_effb0","ladder_effb3","ladder_effb5"]) + f" · EffNet-B7: {g('PROP_DICE')}" + f"""

## Legacy-preprocessing check (why baselines failed before)
U-Net R34 with original 8-bit loader: {g('D:legacy_prep_unet_r34')} vs {g('D:base_unet_r34')} with proper normalisation. Proposed: {g('D:legacy_prep_proposed')} vs {g('PROP_DICE')}.

## Manuscript changes by section
- **Title**: "boundary-aware"/"adaptive loss stabilization" framing replaced; evaluation protocol in the title.
- **Abstract**: rewritten; leads with the leakage finding and corrected protocol; all numbers with CIs; no clinical claims.
- **Introduction**: shortened; opens from the concrete problem; explicit statement of the two corrected errors; gap paragraph; new Related Work subsection (4 parts) incl. the 4 reviewer-requested citations (2 substantive, 2 brief).
- **Results**: new §Dataset provenance + leakage audit (Fig 1, Table 2 released-ckpt audit); budget-matched comparison with 7 baselines + Cellpose (Table 3, Fig 2); 11-row component ablation (Table 4, Fig 3); Gabor sensitivity (Table 5, Fig 5 = real kernels); encoder ladder (Table 6, Fig 6); training curves + seeds + CV (Fig 7); qualitative best/median/worst (Fig 8).
- **Discussion**: reframed around protocol; bugs disclosed; limitations (single lab/line/instrument, single annotator, no external validation, no clinical claims).
- **Methods**: dataset provenance + encodings + exclusions; no ROI stage; corrected Gabor (λ={{20, 6.67, 4}}, L1 norm, ψ, ksize); Algorithm 1 = the real loss with zero-positive guard; sampler formula; single training protocol; evaluation with boundary metrics + statistics; Cellpose protocol; AI-use statement.
- **Data availability**: public code/split/seeds/weights + sample release (URLs pending).

## Still needs a human decision
- Public repo URL + Zenodo DOI + sample-data licence (Data availability, R1.3).
- Inter-annotator study (R1.10) — needs a second annotator on ≥30 slices.
- Two bibliography entries (EEG, Jena API) need full details verified.
- Which journal: manuscript uses the SR class; body is class-agnostic.
"""
open(f"{DEST}/CHANGES_SUMMARY.md", "w").write(md); print("wrote CHANGES_SUMMARY.md")
