#!/usr/bin/env python3
"""
benchmark_efficiency.py — Measure params, FLOPs, and inference FPS for each model.

Usage:
    python benchmark_efficiency.py

Outputs:
    results/efficiency_table.csv
    results/efficiency_table.tex
"""

import time
import csv
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model, BASELINE_REGISTRY


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_fps(model, device, input_shape=(1, 4, 512, 512), warmup=10, runs=50):
    """Measure inference FPS."""
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_time = np.mean(times)
    return 1.0 / avg_time if avg_time > 0 else 0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_test = [("proposed", 4)] + [(k, 3) for k in BASELINE_REGISTRY]

    results = []
    for name, ch in models_to_test:
        print(f"\nBenchmarking: {name} ({ch}-ch)...")
        try:
            model, desc = build_model(name, in_channels=ch, encoder_weights=None)
            model = model.to(device)
            params = count_params(model)
            params_m = params / 1e6

            input_shape = (1, ch, 512, 512)
            fps = measure_fps(model, device, input_shape)

            results.append(dict(
                name=name, description=desc,
                params_M=f"{params_m:.1f}",
                fps=f"{fps:.1f}",
            ))
            print(f"  Params: {params_m:.1f}M, FPS: {fps:.1f}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append(dict(name=name, description=str(e),
                                 params_M="—", fps="—"))

    # Save
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "efficiency_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    # LaTeX
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational efficiency comparison (512$\times$512 input).}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Params (M) & FPS \\",
        r"\midrule",
    ]
    for r in results:
        desc = r["description"][:40]
        tex.append(rf"{desc} & {r['params_M']} & {r['fps']} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = out_dir / "efficiency_table.tex"
    tex_path.write_text("\n".join(tex))

    print(f"\nSaved: {csv_path}, {tex_path}")


if __name__ == "__main__":
    main()
