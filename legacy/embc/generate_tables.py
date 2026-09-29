#!/usr/bin/env python3
"""
generate_tables.py — Read results/comparison_table.csv and produce
LaTeX tables ready for the EMBC paper.

Usage:
    python generate_tables.py
    # or specify a different CSV:
    python generate_tables.py --csv results/comparison_table.csv
"""

import argparse
import csv
from pathlib import Path


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def latex_sota_table(rows):
    """Table II replacement: SOTA comparison."""
    sota = [r for r in rows if r["name"].startswith("sota_") or r["name"] == "proposed_full"]
    sota.sort(key=lambda r: -float(r["dice_mean"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison with state-of-the-art segmentation methods on the breast cell nucleus test set (100 images).}",
        r"\label{tab:sota}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Dice (\%) & IoU (\%) & Precision & Recall \\",
        r"\midrule",
    ]
    for r in sota:
        dm = float(r["dice_mean"]) * 100
        ds = float(r["dice_std"]) * 100
        im = float(r["iou_mean"]) * 100
        p = float(r["precision"])
        rc = float(r["recall"])
        desc = r["description"]
        bold = r["name"] == "proposed_full"
        if bold:
            lines.append(
                rf"\textbf{{{desc}}} & \textbf{{{dm:.1f} $\pm$ {ds:.1f}}} "
                rf"& \textbf{{{im:.1f}}} & \textbf{{{p:.3f}}} & \textbf{{{rc:.3f}}} \\"
            )
        else:
            lines.append(
                rf"{desc} & {dm:.1f} $\pm$ {ds:.1f} & {im:.1f} & {p:.3f} & {rc:.3f} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_ablation_table(rows):
    """Table III: ablation study."""
    abl = [r for r in rows if r["name"].startswith("ablation_") or r["name"] == "proposed_full"]
    abl.sort(key=lambda r: -float(r["dice_mean"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: contribution of each proposed component.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & Dice (\%) & $\Delta$ \\",
        r"\midrule",
    ]
    proposed_dice = None
    for r in abl:
        if r["name"] == "proposed_full":
            proposed_dice = float(r["dice_mean"])
            break

    for r in abl:
        dm = float(r["dice_mean"]) * 100
        ds = float(r["dice_std"]) * 100
        delta = (float(r["dice_mean"]) - proposed_dice) * 100 if proposed_dice else 0
        desc = r["description"]
        if r["name"] == "proposed_full":
            lines.append(rf"\textbf{{{desc}}} & \textbf{{{dm:.1f} $\pm$ {ds:.1f}}} & --- \\")
        else:
            lines.append(rf"{desc} & {dm:.1f} $\pm$ {ds:.1f} & {delta:+.1f} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/comparison_table.csv")
    args = parser.parse_args()

    rows = load_csv(args.csv)
    out_dir = Path(args.csv).parent

    sota = latex_sota_table(rows)
    abl = latex_ablation_table(rows)

    (out_dir / "table_sota.tex").write_text(sota)
    (out_dir / "table_ablation.tex").write_text(abl)

    print("SOTA comparison table:")
    print(sota)
    print("\nAblation table:")
    print(abl)
    print(f"\nSaved to {out_dir / 'table_sota.tex'} and {out_dir / 'table_ablation.tex'}")


if __name__ == "__main__":
    main()
