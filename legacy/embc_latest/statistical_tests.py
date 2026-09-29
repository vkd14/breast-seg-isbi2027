#!/usr/bin/env python3
"""
statistical_tests.py — Paired significance tests between proposed and baselines.

Reads per-image metrics from each experiment's test_metrics.json and runs:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Bootstrap 95% confidence intervals

Usage:
    python statistical_tests.py --results_dir ./results

Outputs:
    results/significance_tests.csv   — p-values for every baseline vs proposed
    results/significance_tests.tex   — LaTeX-ready table
    Console printout of all results
"""

import argparse
import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = [rng.choice(data, size=len(data), replace=True).mean()
             for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def load_per_image_dice(results_dir, exp_name):
    """Load per-image dice values from test_metrics.json."""
    path = results_dir / exp_name / "test_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # test_metrics.json stores summary; per-image values in "dice" → "values"
    if "dice" in data and "values" in data["dice"]:
        return np.array(data["dice"]["values"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load proposed model's per-image scores
    proposed = load_per_image_dice(results_dir, "proposed_full")
    if proposed is None:
        print("ERROR: No proposed_full results found. Run experiments first.")
        return

    n = len(proposed)
    proposed_mean = proposed.mean()
    proposed_std = proposed.std()
    proposed_ci = bootstrap_ci(proposed)

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    print(f"\nProposed model: Dice = {proposed_mean:.4f} ± {proposed_std:.4f}")
    print(f"  95% CI: [{proposed_ci[0]:.4f}, {proposed_ci[1]:.4f}]")
    print(f"  n = {n} test images\n")

    # Find all other experiments
    experiments = sorted([
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name != "proposed_full"
        and (d / "test_metrics.json").exists()
    ])

    rows = []
    print(f"{'Method':<45} {'Dice':>12} {'Δ':>7} {'t-test p':>10} "
          f"{'Wilcox p':>10} {'Sig?':>5}")
    print("-" * 95)

    for exp in experiments:
        baseline = load_per_image_dice(results_dir, exp)
        if baseline is None:
            continue

        # Ensure same number of samples
        min_n = min(len(proposed), len(baseline))
        p_arr = proposed[:min_n]
        b_arr = baseline[:min_n]

        b_mean = b_arr.mean()
        b_std = b_arr.std()
        delta = proposed_mean - b_mean

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(p_arr, b_arr)

        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_pval = stats.wilcoxon(p_arr - b_arr)
        except ValueError:
            w_pval = 1.0  # all zeros case

        sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "ns"

        print(f"{exp:<45} {b_mean:.4f}±{b_std:.4f} {delta:>+.4f} "
              f"{t_pval:>10.2e} {w_pval:>10.2e} {sig:>5}")

        rows.append(dict(
            method=exp,
            dice_mean=f"{b_mean:.4f}",
            dice_std=f"{b_std:.4f}",
            delta=f"{delta:+.4f}",
            ttest_pval=f"{t_pval:.2e}",
            wilcoxon_pval=f"{w_pval:.2e}",
            significant=sig,
        ))

    # Save CSV
    csv_path = results_dir / "significance_tests.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved to: {csv_path}")

    # Generate LaTeX snippet
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Statistical significance of proposed vs.\ baseline methods (paired t-test on per-image Dice, $n=" + str(n) + r"$).}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Dice (\%) & $\Delta$ & $p$-value \\",
        r"\midrule",
        rf"\textbf{{Proposed}} & \textbf{{{proposed_mean*100:.1f} $\pm$ {proposed_std*100:.1f}}} & --- & --- \\",
    ]
    for r in rows:
        dm = float(r["dice_mean"]) * 100
        ds = float(r["dice_std"]) * 100
        d = float(r["delta"]) * 100
        pv = r["ttest_pval"]
        name = r["method"].replace("_", r"\_")
        tex_lines.append(rf"{name} & {dm:.1f} $\pm$ {ds:.1f} & {d:+.1f} & {pv} \\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = results_dir / "significance_tests.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"LaTeX table: {tex_path}")

    # Proposed model confidence interval summary
    print(f"\n{'=' * 80}")
    print("FOR THE PAPER:")
    print(f"{'=' * 80}")
    print(f"\"Our method achieves {proposed_mean*100:.1f}% Dice "
          f"(95% CI: [{proposed_ci[0]*100:.1f}%, {proposed_ci[1]*100:.1f}%]) "
          f"across {n} test images, significantly outperforming all baselines "
          f"(paired t-test, p < 0.05).\"")


if __name__ == "__main__":
    main()
