#!/usr/bin/env python3
"""Run the full budget-matched experiment queue with N parallel GPU workers."""
import subprocess, sys, os, time, json
PY_ = os.path.expanduser("~/Desktop/breast-3d/breast_qubo/.venv/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "..", "..", "outputs", "revision", "runs")
EP = "30"; NW = int(sys.argv[1]) if len(sys.argv) > 1 else 2
P = dict(arch="unetplusplus", encoder="efficientnet-b7", edge="gabor", scse="1", proj="1", sampling="complexity", loss="stabilized")
def job(name, **kw):
    c = dict(P); c.update(kw); args = [PY_, os.path.join(HERE, "train_revision.py"), "--name", name, "--epochs", EP]
    for k, v in c.items(): args += [f"--{k}", str(v)]
    return name, args
Q2 = [
  job("legacy_prep_proposed", legacy_prep=1), job("legacy_prep_unet_r34", arch="unet", encoder="resnet34", edge="none", scse=0, proj=0, legacy_prep=1),
] + [job(f"cv_fold{i+1}", fold=",".join(f)) for i, f in enumerate(json.load(open(os.path.join(RUNS, "..", "cv_folds.json"))))]
Q = [
  job("proposed_s0", seed=0, save_ckpt=1), job("proposed_s1", seed=1), job("proposed_s2", seed=2),
  job("abl_no_gabor", edge="none"), job("abl_no_scse", scse=0), job("abl_uniform_sampling", sampling="uniform"),
  job("abl_loss_dicebce", loss="dicebce"), job("abl_loss_tversky_boundary", loss="tversky_boundary"),
  job("abl_plain_unet", arch="unet"),
  job("vanilla_effb7_unetpp", edge="none", scse=0, sampling="uniform", loss="dicebce"),
  job("edge_sobel", edge="sobel"), job("edge_scharr", edge="scharr"), job("edge_log", edge="log"), job("edge_canny", edge="canny"),
  job("ladder_resnet34", encoder="resnet34"), job("ladder_resnet50", encoder="resnet50"),
  job("ladder_effb0", encoder="efficientnet-b0"), job("ladder_effb3", encoder="efficientnet-b3"), job("ladder_effb5", encoder="efficientnet-b5"),
  job("base_unet_r34", arch="unet", encoder="resnet34", edge="none", scse=0, proj=0),
  job("base_unet_r50", arch="unet", encoder="resnet50", edge="none", scse=0, proj=0),
  job("base_attunet_r50", arch="unet", encoder="resnet50", edge="none", scse=1, proj=0),
  job("base_fpn_r50", arch="fpn", encoder="resnet50", edge="none", scse=0, proj=0),
  job("base_deeplabv3plus_r50", arch="deeplabv3plus", encoder="resnet50", edge="none", scse=0, proj=0),
  job("base_manet_r50", arch="manet", encoder="resnet50", edge="none", scse=0, proj=0),
  job("base_unetpp_r50", arch="unetplusplus", encoder="resnet50", edge="none", scse=0, proj=0),
  job("gabor_orient4", gabor="4,3,5.0,0.5,31"), job("gabor_orient16", gabor="16,3,5.0,0.5,31"),
  job("gabor_scale2", gabor="8,2,5.0,0.5,31"), job("gabor_scale5", gabor="8,5,5.0,0.5,31"),
  job("gabor_sigma2", gabor="8,3,2.0,0.5,31"), job("gabor_sigma10", gabor="8,3,10.0,0.5,31"),
]
Q = Q2 if (len(sys.argv) > 2 and sys.argv[2] == "q2") else Q
os.makedirs(RUNS, exist_ok=True)
pending = [(n, a) for n, a in Q if not os.path.exists(os.path.join(RUNS, n, "result.json"))]
print(f"{len(Q)} jobs, {len(pending)} pending, {NW} workers", flush=True)
running = []
while pending or running:
    while pending and len(running) < NW:
        n, a = pending.pop(0); log = open(os.path.join(RUNS, f"{n}.log"), "w")
        running.append((n, subprocess.Popen(a, stdout=log, stderr=subprocess.STDOUT), time.time())); print(f"START {n}", flush=True)
    for r in list(running):
        if r[1].poll() is not None:
            running.remove(r); print(f"END   {r[0]} rc={r[1].returncode} {((time.time()-r[2])/60):.1f}min", flush=True)
    time.sleep(5)
print("QUEUE COMPLETE", flush=True)
