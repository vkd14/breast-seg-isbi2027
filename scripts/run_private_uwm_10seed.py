#!/usr/bin/env python3
"""Resume an equal-budget 10-seed UWM primary/control experiment.

The private images remain outside this repository. Completed, protocol-matched
runs are validated and skipped, so the queue is safe to restart.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = {
    "proposed": {
        "arch": "unetplusplus", "encoder": "efficientnet-b7", "edge": "gabor",
        "scse": 1, "proj": 1, "sampling": "complexity", "loss": "stabilized",
    },
    "vanilla_effb7_unetpp": {
        "arch": "unetplusplus", "encoder": "efficientnet-b7", "edge": "none",
        "scse": 0, "proj": 1, "sampling": "uniform", "loss": "dicebce",
    },
}
FIXED = {"epochs": 30, "bs": 8, "lr": 3e-4, "size": 512,
         "gabor": "8,3,5.0,0.5,31", "fold": "", "legacy_prep": 0}


def run_name(variant: str, seed: int) -> str:
    # The historical vanilla seed-0 directory predates explicit seed suffixes.
    if variant == "vanilla_effb7_unetpp" and seed == 0:
        return variant
    return f"{variant}_s{seed}"


def validate_completed(run_dir: Path, variant: str, seed: int) -> bool:
    result_path = run_dir / "result.json"
    per_image_path = run_dir / "per_image.csv"
    if not result_path.is_file() or not per_image_path.is_file():
        return False
    result = json.loads(result_path.read_text())
    config = result.get("config", {})
    expected = {**VARIANTS[variant], **FIXED, "seed": seed}
    mismatches = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key, 0 if key == "legacy_prep" else None) != value}
    if mismatches:
        raise RuntimeError(f"Protocol mismatch in {run_dir}: {mismatches}")
    if result.get("test", {}).get("n") != 206:
        raise RuntimeError(f"Incomplete test evaluation in {run_dir}")
    if len(per_image_path.read_text().splitlines()) != 207:
        raise RuntimeError(f"Unexpected per-image row count in {run_dir}")
    return True


def command(python: Path, train: Path, name: str, variant: str, seed: int) -> list[str]:
    values = {**VARIANTS[variant], **FIXED}
    args = [str(python), str(train), "--name", name, "--seed", str(seed), "--save_ckpt", "0"]
    for key, value in values.items():
        args.extend((f"--{key}", str(value)))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    private_root = args.private_root.expanduser().resolve()
    train = private_root / "src/revision/train_revision.py"
    runs = private_root / "outputs/revision/runs"
    split = private_root / "outputs/revision/volume_split.json"
    if not train.is_file() or not split.is_file():
        raise FileNotFoundError("Private training script or fixed split is missing")

    jobs = []
    completed = []
    for variant in VARIANTS:
        for seed in range(10):
            name = run_name(variant, seed)
            run_dir = runs / name
            if validate_completed(run_dir, variant, seed):
                completed.append(name)
            else:
                jobs.append((name, variant, seed))
    print(f"20 required runs: {len(completed)} complete, {len(jobs)} pending", flush=True)

    running: list[tuple[str, subprocess.Popen, object, float]] = []
    while jobs or running:
        while jobs and len(running) < args.workers:
            name, variant, seed = jobs.pop(0)
            log_handle = (runs / f"private10seed_{name}.log").open("w")
            process = subprocess.Popen(
                command(args.python, train, name, variant, seed),
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
            running.append((name, process, log_handle, time.time()))
            print(f"START {name}", flush=True)
        time.sleep(5)
        for item in list(running):
            name, process, log_handle, started = item
            if process.poll() is None:
                continue
            running.remove(item)
            log_handle.close()
            elapsed = (time.time() - started) / 60
            print(f"END   {name} rc={process.returncode} {elapsed:.1f} min", flush=True)
            if process.returncode:
                raise SystemExit(f"{name} failed; inspect private10seed_{name}.log")

    analyzer = Path(__file__).with_name("analyze_private_uwm_10seed.py")
    subprocess.run(
        [str(args.python), str(analyzer), "--private-root", str(private_root)],
        check=True,
    )
    print("PRIVATE UWM 10-SEED PRIMARY/CONTROL STUDY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
