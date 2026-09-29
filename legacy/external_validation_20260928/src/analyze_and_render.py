#!/usr/bin/env python3
"""Build measured figures, patient-level analysis, and reviewable draft PDFs."""
import csv
import html
import json
import importlib.metadata
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
import evaluate_bbbc038 as ev

ROOT = ev.ROOT
REPORT = ROOT / "reports"
FIG = REPORT / "figures"
METHODS = ["released_2d", "revised_2d_s0", "otsu_border_polarity", "cellpose_sam"]
LABELS = {"released_2d": "Released 2D", "revised_2d_s0": "Revised 2D (s0)", "otsu_border_polarity": "Otsu", "cellpose_sam": "Cellpose-SAM"}
DATASETS = ["bbbc038", "bbbc039", "tnbc"]


def load_all():
    summaries, rows, cases = {}, {}, {}
    for ds in DATASETS:
        out = ROOT / "outputs" / (ds + "_zero_shot")
        summaries[ds] = json.loads((out / "summary.json").read_text())
        with open(out / "per_image.csv") as f:
            rr = list(csv.DictReader(f))
        for row in rr:
            for k in ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px", "seconds"):
                row[k] = float(row[k])
        rows[ds] = rr
        ev.DATA = ROOT / "data" / ds
        cases[ds] = ev.load_cases(ds)
    return summaries, rows, cases


def bootstrap(a, n=10000):
    a = np.array(a)
    rng = np.random.default_rng(20260928)
    means = a[rng.integers(len(a), size=(n, len(a)))].mean(1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def analyze(rows, cases):
    patients = sorted({r["image_id"].split("_")[0] for r in rows["tnbc"]})
    assert len(patients) == 11
    patient_stats = {}
    for method in METHODS:
        per = {}
        for p in patients:
            pp = [r for r in rows["tnbc"] if r["method"] == method and r["image_id"].split("_")[0] == p]
            per[p] = {k: float(np.mean([r[k] for r in pp])) for k in ("dice", "iou", "boundary_f1_2px")}
        a = [v["dice"] for v in per.values()]
        patient_stats[method] = {"n_patients": 11, "dice": float(np.mean(a)), "dice_ci95": bootstrap(a), "per_patient": per}
    tests = []
    for a, b in (("revised_2d_s0", "released_2d"), ("revised_2d_s0", "cellpose_sam"), ("released_2d", "cellpose_sam")):
        diff = np.array([patient_stats[a]["per_patient"][p]["dice"] - patient_stats[b]["per_patient"][p]["dice"] for p in patients])
        w, pv = stats.wilcoxon(diff, alternative="two-sided", method="auto")
        tests.append({"comparison": a + " minus " + b, "n_patients": 11, "mean_difference": float(diff.mean()), "difference_ci95": bootstrap(diff), "p_raw": float(pv), "W": float(w)})
    last = 0.
    for rank, i in enumerate(np.argsort([t["p_raw"] for t in tests])):
        last = max(last, min(1., tests[i]["p_raw"] * (len(tests) - rank)))
        tests[i]["p_holm"] = last
    lookup = {sid: "dark" if np.median(rgb) < 128 else "bright" for sid, rgb, gt, n in cases["bbbc038"]}
    strata = {}
    for group in ("dark", "bright"):
        strata[group] = {"n": sum(v == group for v in lookup.values())}
        for method in METHODS:
            strata[group][method] = float(np.mean([r["dice"] for r in rows["bbbc038"] if r["method"] == method and lookup[r["image_id"]] == group]))
    return {"tnbc_patient_macro": patient_stats, "tnbc_paired_tests_holm_family_of_three": tests,
            "bbbc038_posthoc_intensity_strata": strata, "strata_rule": "median native RGB uint8 intensity <128; not a verified modality label"}


def display_rgb(rgb):
    if rgb.dtype == np.uint16:
        lo, hi = np.percentile(rgb, [1, 99.5])
        return np.clip((rgb.astype(float) - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def figures(summaries, rows, cases):
    FIG.mkdir(exist_ok=True, parents=True)
    palette = ["#8594a3", "#087e8b", "#e6a23c", "#5645a5"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), layout="constrained")
    for ax, key, label in zip(axes, ("dice", "boundary_f1_2px"), ("Foreground Dice", "Boundary F1 (2 pixels)")):
        x = np.arange(3)
        for j, method in enumerate(METHODS):
            vals = [summaries[ds][method][key] for ds in DATASETS]
            ax.bar(x + (j - 1.5) * .19, vals, width=.18, color=palette[j], label=LABELS[method])
        ax.set_xticks(x, ["BBBC038\nn=65", "BBBC039\nn=50", "TNBC\nn=50"])
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(fontsize=8, loc="upper left", ncol=2)
    fig.savefig(FIG / "external_comparison.png", dpi=180)
    fig.savefig(FIG / "external_comparison.pdf")
    plt.close(fig)
    selections = {}
    for ds in DATASETS:
        revised = sorted([r for r in rows[ds] if r["method"] == "revised_2d_s0"], key=lambda r: (r["dice"], r["image_id"]))
        selected = [revised[round((len(revised) - 1) * q)] for q in (.1, .5, .9)]
        selections[ds] = [{"quantile": q, "id": r["image_id"], "dice": r["dice"]} for q, r in zip((.1, .5, .9), selected)]
        case_map = {s: (rgb, gt) for s, rgb, gt, n in cases[ds]}
        fig, axes = plt.subplots(3, 5, figsize=(11, 6.6), layout="constrained")
        for i, row in enumerate(selected):
            sid = row["image_id"]
            rgb, gt = case_map[sid]
            axes[i, 0].imshow(display_rgb(rgb))
            axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f"q{[10,50,90][i]}\n{sid[:12]}", fontsize=8)
            for j, method in enumerate(("released_2d", "revised_2d_s0", "cellpose_sam"), 2):
                pred = cv2.imread(str(ROOT / "outputs" / (ds + "_zero_shot") / "predictions" / method / (sid + ".png")), 0)
                axes[i, j].imshow(pred, cmap="gray", vmin=0, vmax=255)
                metric = next(r["dice"] for r in rows[ds] if r["method"] == method and r["image_id"] == sid)
                axes[i, j].set_xlabel(f"Dice {metric:.3f}", fontsize=8)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        for j, title in enumerate(("Input", "Reference foreground", "Released", "Revised seed 0", "Cellpose-SAM")):
            axes[0, j].set_title(title, fontsize=10)
        fig.suptitle(ds.upper() + " | actual masks; rows selected by revised Dice quantiles", fontsize=12)
        fig.savefig(FIG / (ds + "_predictions.png"), dpi=160)
        plt.close(fig)
    # Actual, reproducible fourth-channel example, not an illustrative drawing.
    sid = selections["bbbc039"][1]["id"]
    rgb = next(im for s, im, gt, n in cases["bbbc039"] if s == sid)
    x = ev.preprocess(rgb, True)[0].numpy()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    for ax, im, title in zip(axes, (display_rgb(rgb), x[0] * .229 + .485, x[3] * .25 + .5), ("Native fluorescence", "Network intensity input", "Actual Gabor fourth channel")):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.savefig(FIG / "gabor_example.png", dpi=180)
    plt.close(fig)
    (REPORT / "figure_selection.json").write_text(json.dumps(selections, indent=2))


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<link href="\2" color="#087e8b">\1</link>', text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def render_pdf(md_path):
    fontroot = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(fontroot / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fontroot / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
    styles = getSampleStyleSheet()
    for name in ("Normal", "Title", "Heading1", "Heading2", "Heading3"):
        styles[name].fontName = "DejaVu-Bold" if name != "Normal" else "DejaVu"
    styles["Normal"].fontSize, styles["Normal"].leading = 9, 12.5
    styles["Normal"].spaceAfter = 6
    styles["Title"].fontSize, styles["Title"].leading = 18, 22
    styles["Heading1"].fontSize, styles["Heading1"].leading = 13, 17
    styles["Heading2"].fontSize, styles["Heading2"].leading = 10.5, 14
    for name in ("Title", "Heading1", "Heading2", "Heading3"):
        styles[name].keepWithNext = True
    cellstyle = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.2, leading=10, spaceAfter=0)
    width = 7.15 * inch
    doc = SimpleDocTemplate(str(md_path.with_suffix(".pdf")), pagesize=(8.5 * inch, 11 * inch),
                            rightMargin=.675*inch, leftMargin=.675*inch, topMargin=.55*inch, bottomMargin=.6*inch,
                            title=md_path.stem.replace("_", " "), author="Research working draft")
    story, i = [], 0
    lines = md_path.read_text().splitlines()
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in values):
                    table_rows.append([Paragraph(inline(v), cellstyle) for v in values])
                i += 1
            nc = len(table_rows[0])
            col_widths = [width / nc] * nc
            if nc == 3:
                col_widths = [width*.2, width*.4, width*.4]
            if nc == 4:
                col_widths = [width*.14, width*.2, width*.23, width*.43]
            table = Table(table_rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e4edf2")),
                                      ("VALIGN", (0,0), (-1,-1), "TOP"), ("LINEBELOW", (0,0), (-1,0), .5, colors.HexColor("#087e8b")),
                                      ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fa")]),
                                      ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
                                      ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
            story += [table, Spacer(1, 10)]
            continue
        match = re.match(r"!\[.*?\]\((.*?)\)", line)
        if match:
            path = (md_path.parent / match.group(1)).resolve()
            im = Image(str(path))
            ratio = min(width / im.imageWidth, 5*inch / im.imageHeight)
            im.drawWidth, im.drawHeight = im.imageWidth*ratio, im.imageHeight*ratio
            story += [im, Spacer(1, 8)]
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["Heading2"]))
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(("#", "|", "![")):
                i += 1
                paragraph.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(paragraph)), styles["Normal"]))
        i += 1
    def footer(canvas, document):
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(.675*inch, .3*inch, "Research discussion draft | measured results | 28 September 2026")
        canvas.drawRightString(7.825*inch, .3*inch, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    environment = {}
    for package in ("torch", "torchvision", "segmentation-models-pytorch", "cellpose", "numpy", "scipy", "opencv-python", "opencv-python-headless", "matplotlib", "reportlab", "albumentations", "openpyxl"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    (ROOT / "outputs/environment.json").write_text(json.dumps(environment, indent=2))
    from download_data import URLS
    # Clarify per-method settings in the original manifests without altering measured outputs.
    for ds in DATASETS:
        path = ROOT / "outputs" / (ds + "_zero_shot") / "protocol.json"
        protocol = json.loads(path.read_text())
        protocol["source_urls"] = URLS[ds]
        protocol["prediction_resolution_note"] = "512x512 and threshold 0.5 apply to the two author networks only; Cellpose/Otsu use native images"
        protocol["cellpose_settings"] = {"flow_threshold": .4, "cellprob_threshold": 0., "normalize": True, "do_3D": False, "channel_axis": -1, "augment": False}
        protocol["otsu_settings"] = "Native grayscale; invert foreground if median border intensity exceeds Otsu threshold; no cleanup"
        path.write_text(json.dumps(protocol, indent=2))
    s, rows, cases = load_all()
    analysis = analyze(rows, cases)
    (REPORT / "analysis.json").write_text(json.dumps(analysis, indent=2))
    figures(s, rows, cases)
    table = "| Dataset | Method | n | Dice | IoU | Boundary F1 | Surface Dice |\n| --- | --- | --- | --- | --- | --- | --- |\n"
    flat = []
    for ds in DATASETS:
        for m in METHODS:
            v = s[ds][m]
            table += f"| {ds.upper()} | {LABELS[m]} | {v['n']} | {v['dice']:.4f} | {v['iou']:.4f} | {v['boundary_f1_2px']:.4f} | {v['surface_dice_2px']:.4f} |\n"
            flat.append(dict(dataset=ds, method=m, n=v["n"], **{k:v[k] for k in ("dice", "iou", "boundary_f1_2px", "surface_dice_2px", "hd95_px")}, dice_ci_low=v["dice_descriptive_image_bootstrap_ci95"][0], dice_ci_high=v["dice_descriptive_image_bootstrap_ci95"][1]))
    with open(REPORT / "external_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    internal_file = ev.PROJECT / "data&code/outputs/revision/stats_clustered/numbers.json"
    snapshot = ROOT / "outputs/internal_reference_summary.json"
    if internal_file.exists():
        internal = json.loads(internal_file.read_text())["summary"]
        snapshot.write_text(json.dumps({"source": str(internal_file), "source_sha256": ev.sha256(internal_file), "summary": internal}, indent=2))
    else:
        internal = json.loads(snapshot.read_text())["summary"]
    itable = "| Internal configuration | Group-macro Dice | 95% group CI | Seeds |\n| --- | --- | --- | --- |\n"
    for key, title in (("proposed", "Full configuration"), ("vanilla_effb7_unetpp", "Vanilla B7 UNet++"), ("base_unet_r50", "U-Net ResNet50"), ("abl_loss_dicebce", "Full model, plain Dice-BCE"), ("cellpose_zeroshot", "Cellpose-SAM reference")):
        v = internal[key]
        itable += f"| {title} | {v['dice']:.4f} | {v['dice_lo']:.4f}-{v['dice_hi']:.4f} | {v['n_seeds']} |\n"
    patient = analysis["tnbc_patient_macro"]
    pp = "Equal weighting of the 11 TNBC patient folders gives revised-model Dice " + f"{patient['revised_2d_s0']['dice']:.4f} (95% patient-bootstrap CI {patient['revised_2d_s0']['dice_ci95'][0]:.4f}-{patient['revised_2d_s0']['dice_ci95'][1]:.4f}), versus {patient['cellpose_sam']['dice']:.4f} for Cellpose-SAM."
    test = next(t for t in analysis["tnbc_paired_tests_holm_family_of_three"] if t["comparison"] == "revised_2d_s0 minus cellpose_sam")
    pp += f" The paired two-sided Wilcoxon comparison has Holm-adjusted p={test['p_holm']:.5f} across the three specified TNBC comparisons. This indicates a difference between the deployed pipelines on these patients, without isolating architecture from training exposure."
    values = {"INTERNAL_TABLE": itable, "EXTERNAL_TABLE": table, "TNBC_PARAGRAPH": pp,
              "D038": f"{s['bbbc038']['revised_2d_s0']['dice']:.3f}", "D039": f"{s['bbbc039']['revised_2d_s0']['dice']:.3f}", "DTNBC": f"{s['tnbc']['revised_2d_s0']['dice']:.3f}",
              "C038": f"{s['bbbc038']['cellpose_sam']['dice']:.3f}", "C039": f"{s['bbbc039']['cellpose_sam']['dice']:.3f}", "CTNBC": f"{s['tnbc']['cellpose_sam']['dice']:.3f}",
              "R039": f"{s['bbbc039']['released_2d']['dice']:.3f}", "O039": f"{s['bbbc039']['otsu_border_polarity']['dice']:.3f}", "RTNBC": f"{s['tnbc']['released_2d']['dice']:.3f}",
              "DELTA038": f"{100*(s['bbbc038']['revised_2d_s0']['dice']-s['bbbc038']['released_2d']['dice']):.1f}",
              "DARK": f"{analysis['bbbc038_posthoc_intensity_strata']['dark']['revised_2d_s0']:.3f}", "BRIGHT": f"{analysis['bbbc038_posthoc_intensity_strata']['bright']['revised_2d_s0']:.3f}"}
    draft = (ROOT / "paper/draft_template.md").read_text()
    for k, v in values.items():
        draft = draft.replace("{{" + k + "}}", v)
    assert "{{" not in draft
    (ROOT / "paper/first_draft.md").write_text(draft)
    brief = f"""# Professor meeting brief: 2D nucleus segmentation

Prepared September 28 for the September 29, 2026 discussion. Recommended conference: **IEEE ISBI 2027, October 26 submission deadline**. The existing journal and IEEE submissions were rejected or withdrawn, as confirmed by the author.

## What was completed tonight

Both supplied PDFs and the later corrected manuscript/code were inspected. Four frozen methods were evaluated on 165 image entries across three public collections, producing 660 scored predictions. These are not 165 guaranteed-independent acquisitions: the two BBBC collections partly overlap. No new training was performed. Every dataset has complete stated test coverage; there was no favorable-image selection, threshold tuning, or test-time augmentation.

{table}

## What the results support

The revised checkpoint improves BBBC038 Dice by {values['DELTA038']} percentage points compared with the released pipeline. This measures the combined effect of corrected training/preprocessing/checkpoint selection, not an isolated Gabor effect. On fluorescence BBBC039, both author checkpoints achieve about 0.92 Dice, but Otsu and Cellpose-SAM are better. On TNBC histology, both checkpoints fail to transfer reliably. The revised pipeline is not uniformly better than the released one.

{pp}

The current evidence supports research-level nuclear foreground segmentation within a restricted imaging domain. It does not demonstrate cancer detection, clinical readiness, a universal model, or superiority over Cellpose. The large historical baseline advantage disappears under the existing corrected internal protocol.

![Measured public benchmark performance](figures/external_comparison.png)

## What changed compared with the submitted work

| Area | Submitted claim or limitation | Evidence now |
| --- | --- | --- |
| Generalization | No public external test | Three complete public evaluations, including explicit histology failures |
| Baseline advantage | Large margin over weak baselines | Corrected internal methods cluster around 0.93; vanilla B7 slightly exceeds the full configuration |
| Boundaries | Boundary-aware loss claimed | Actual loss identified as Dice-BCE; native foreground-boundary F1, surface Dice and HD95 added |
| Dataset statistics | 60% empty images | Existing audit identifies mask-encoding error; 855 valid pairs all contain foreground |
| Independence | Unclear source-volume split | 334 traceable slices from eight stacks; missing provenance for 521 explicitly acknowledged |
| Statistics | Image-level comparisons | Existing group-level corrected analysis; new patient-level TNBC inference and descriptive BBBC intervals |
| Scientific figures | Inconsistent illustrative figures | Actual images, masks, computed Gabor map, and documented quantile-selected predictions |

## Figures for discussion

![Fluorescence examples](figures/bbbc039_predictions.png)

BBBC039: q10/median/q90 images by revised Dice. Inputs are display-normalized only in this panel; evaluation uses the recorded model-specific preprocessing. References and predictions retain native geometry.

![Heterogeneous microscopy examples](figures/bbbc038_predictions.png)

BBBC038: the same quantile-selection rule. The full-dataset average includes all failures. A post-hoc median-intensity split gives revised Dice {values['DARK']} on 53 dark images and {values['BRIGHT']} on 12 bright images; this is a diagnostic, not a verified microscopy-modality classification.

![Breast histology failures](figures/tnbc_predictions.png)

TNBC: actual breast-histology failure examples. Image and mask source: Naylor et al., dataset v1.1, CC BY 4.0, https://doi.org/10.5281/zenodo.2579118. Prediction panels are derived outputs; quantile sampling and display layout are our modifications.

![Actual Gabor input](figures/gabor_example.png)

The fourth channel is computed with the same filter code as the revised inference. No figure is generated as simulated scientific evidence.

## The 3D paragraph to retain

Existing results on 334 annotated planes from eight source stacks: Cellpose-SAM 0.9181 Dice; fixed distilled 3D U-Net 0.8580; nested selected processing 0.9137; nested Cellpose-plus 0.9177. These are selected-plane semantic scores, not dense 3D instance accuracy. The QC student has trained, but its pseudo-label validation score is not an independent human-reference result and is excluded from the comparison table.

## Recommended next experiment and decisions

Prioritize controlled domain adaptation and intensity/polarity handling. Use BBBC039's official 100 training and 50 validation images, a same-encoder vanilla baseline, smaller U-Net, equal budgets, and three seeds. Add explicit boundary/distance supervision only as an ablation. For TNBC, partition by patient, never by random tiles. Use a new untouched cohort for confirmation because the current benchmarks have now been inspected.

For the professor: approve ISBI versus the longer MIDL schedule; confirm acquisition and annotation details; arrange repeat annotation; authorize appropriate private-data and model-weight release. The companion reviewer matrix identifies every reviewer concern and what remains unresolved. No manuscript was submitted automatically.

Official venue evidence: [ISBI October 26](https://biomedicalimaging.org/2027/papers/), [MIDL December 4 full-paper deadline](https://2027.midl.io/important-dates), [IPMI December 14 full-paper deadline](https://2027.ipmi-conf.org/). Detailed journal/dataset choices and the four-week plan are in venues_and_dataset_plan.md.
"""
    (REPORT / "meeting_brief.md").write_text(brief)
    for path in (ROOT / "paper/first_draft.md", REPORT / "meeting_brief.md", REPORT / "venues_and_dataset_plan.md", REPORT / "reviewer_action_matrix.md"):
        render_pdf(path)
        print("Created", path.with_suffix(".pdf"), flush=True)


if __name__ == "__main__":
    main()
