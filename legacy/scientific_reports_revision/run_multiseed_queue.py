#!/usr/bin/env python3
"""Complete seeds 1 and 2 for every hold-out comparison in the revision.

The original queue ran three seeds only for ``proposed`` and one seed for all
other configurations.  This queue uses the same training entry point and
hyperparameters, changing only the random seed.  Existing completed runs are
left untouched and are skipped on restart.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


PYTHON = os.path.expanduser("~/Desktop/breast-3d/breast_qubo/.venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_revision.py")
RUNS = os.path.abspath(os.path.join(HERE, "..", "..", "outputs", "revision", "runs"))
EPOCHS = "30"
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

PROPOSED = dict(
    arch="unetplusplus",
    encoder="efficientnet-b7",
    edge="gabor",
    scse=1,
    proj=1,
    sampling="complexity",
    loss="stabilized",
)


def config(name: str, **changes):
    values = dict(PROPOSED)
    values.update(changes)
    return name, values


CONFIGS = [
    config("abl_no_gabor", edge="none"),
    config("abl_no_scse", scse=0),
    config("abl_uniform_sampling", sampling="uniform"),
    config("abl_loss_dicebce", loss="dicebce"),
    config("abl_loss_tversky_boundary", loss="tversky_boundary"),
    config("abl_plain_unet", arch="unet"),
    config("vanilla_effb7_unetpp", edge="none", scse=0, sampling="uniform", loss="dicebce"),
    config("edge_sobel", edge="sobel"),
    config("edge_scharr", edge="scharr"),
    config("edge_log", edge="log"),
    config("edge_canny", edge="canny"),
    config("ladder_resnet34", encoder="resnet34"),
    config("ladder_resnet50", encoder="resnet50"),
    config("ladder_effb0", encoder="efficientnet-b0"),
    config("ladder_effb3", encoder="efficientnet-b3"),
    config("ladder_effb5", encoder="efficientnet-b5"),
    config("base_unet_r34", arch="unet", encoder="resnet34", edge="none", scse=0, proj=0),
    config("base_unet_r50", arch="unet", encoder="resnet50", edge="none", scse=0, proj=0),
    config("base_attunet_r50", arch="unet", encoder="resnet50", edge="none", scse=1, proj=0),
    config("base_fpn_r50", arch="fpn", encoder="resnet50", edge="none", scse=0, proj=0),
    config("base_deeplabv3plus_r50", arch="deeplabv3plus", encoder="resnet50", edge="none", scse=0, proj=0),
    config("base_manet_r50", arch="manet", encoder="resnet50", edge="none", scse=0, proj=0),
    config("base_unetpp_r50", arch="unetplusplus", encoder="resnet50", edge="none", scse=0, proj=0),
    config("gabor_orient4", gabor="4,3,5.0,0.5,31"),
    config("gabor_orient16", gabor="16,3,5.0,0.5,31"),
    config("gabor_scale2", gabor="8,2,5.0,0.5,31"),
    config("gabor_scale5", gabor="8,5,5.0,0.5,31"),
    config("gabor_sigma2", gabor="8,3,2.0,0.5,31"),
    config("gabor_sigma10", gabor="8,3,10.0,0.5,31"),
]


def command(base_name: str, values: dict, seed: int):
    run_name = f"{base_name}_s{seed}"
    args = [PYTHON, TRAIN, "--name", run_name, "--epochs", EPOCHS, "--seed", str(seed)]
    for key, value in values.items():
        args.extend((f"--{key}", str(value)))
    return run_name, args


def main():
    os.makedirs(RUNS, exist_ok=True)
    jobs = [command(name, values, seed) for name, values in CONFIGS for seed in (1, 2)]
    pending = [job for job in jobs if not os.path.exists(os.path.join(RUNS, job[0], "result.json"))]
    print(f"{len(jobs)} jobs, {len(pending)} pending, {N_WORKERS} workers", flush=True)
    running = []
    while pending or running:
        while pending and len(running) < N_WORKERS:
            name, args = pending.pop(0)
            log = open(os.path.join(RUNS, f"{name}.log"), "w")
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, process, log, time.time()))
            print(f"START {name}", flush=True)
        for item in list(running):
            name, process, log, started = item
            if process.poll() is not None:
                running.remove(item)
                log.close()
                minutes = (time.time() - started) / 60
                print(f"END   {name} rc={process.returncode} {minutes:.1f}min", flush=True)
                if process.returncode:
                    raise SystemExit(f"run failed: {name}; see {name}.log")
        time.sleep(5)
    print("MULTISEED QUEUE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
