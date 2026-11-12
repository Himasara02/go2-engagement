#!/usr/bin/env python3
"""Build the portable runtime benchmark folder (bench_pkg/) to copy to the Raspberry Pi / workstation.

    ~/miniconda3/envs/go2/bin/python make_bench_pkg.py

Contents: bench.py (bench_pkg_src/), the pipeline modules it imports (verbatim copies), the four models,
frames.mp4 (first 170 frames of D4_S3_r5, 3 people, re-encoded H.264), depth_frame.npy (one aligned depth
frame of the same recording), decision_inputs.npz (5,000 random evaluated person-frames: cues, gesture,
confidence; 5,000 random frames' distances of everyone in view; logistic regression weights fitted on all
episodes; median fitted thresholds from results/full/eval.json). frames.mp4 shows unblurred faces: delete the
copies after benchmarking.
"""
import glob
import gzip
import json
import os
import shutil
import subprocess

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from full_eval import FEATS, OUT, ROOT, Data

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "bench_pkg")
TAKE, NFR = "D4_S3_r5", 170
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(os.path.join(DST, "engage"))
shutil.copy(os.path.join(HERE, "bench_pkg_src", "bench.py"), DST)
for f in ("full_cues.py", "find_gestures_crop.py", "find_gestures_centered.py"):
    shutil.copy(os.path.join(HERE, f), DST)
for f in ("__init__.py", "perception.py", "fuzzy.py", "methods.py", "takeio.py"):
    shutil.copy(os.path.join(HERE, "engage", f), os.path.join(DST, "engage"))
shutil.copytree(os.path.join(HERE, "models"), os.path.join(DST, "models"))
import full_cues
shutil.copy(full_cues.POSE_MODEL, os.path.join(DST, "models"))
# full_cues.POSE_MODEL points into the labels pack: make the copy use the bundled model
s = open(os.path.join(DST, "full_cues.py")).read()
old = 'POSE_MODEL = os.path.join(ROOT, "data", "labels_pack_2026-11-10", "code", "models", "pose_landmarker_full.task")'
assert s.count(old) == 1
open(os.path.join(DST, "full_cues.py"), "w").write(s.replace(old, 'POSE_MODEL = os.path.join(HERE, "models", "pose_landmarker_full.task")'))

take_dir = os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3", TAKE)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(take_dir, "raw.mp4"), "-frames:v", str(NFR),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", os.path.join(DST, "frames.mp4")], check=True)
z = np.load(sorted(glob.glob(os.path.join(ROOT, "data", "depth_2026-11-10", TAKE, "*.npz")))[0])
np.save(os.path.join(DST, "depth_frame.npy"), z["depth"][100])

d = Data()
rng = np.random.default_rng(0)
X = d.X()
rows = [i for e in d.eps for i in e["idx"] if i is not None]
pick = rng.choice(rows, 5000, replace=False)
F = np.stack([d.x[k] for k in FEATS], 1)
y = np.array([lab for e, lab in zip(d.eps, d.lab) for i in e["idx"] if i is not None])
sc = StandardScaler().fit(F[rows])
lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(sc.transform(F[rows]), y)
E = json.load(open(os.path.join(OUT, "eval.json")))
fit = E["fitted"]
near = []
for f in sorted(glob.glob(os.path.join(ROOT, "data", "person_distance_2026-11-12", "*.json.gz"))):
    for fr, people in json.load(gzip.open(f, "rt")).items():
        if people:
            near.append([v[0] for v in people.values()])
near = [near[i] for i in rng.choice(len(near), 5000, replace=False)]
off = np.cumsum([0] + [len(g) for g in near])
np.savez(os.path.join(DST, "decision_inputs.npz"), X=X[pick],
         gesture=np.array(["palm" if d.stop[i] else ("fist" if d.cand[i] else "none") for i in pick]),
         conf=np.where(d.cand[pick], 0.9, 0.2), near_d=np.concatenate(near), near_off=off,
         lr_mean=sc.mean_, lr_scale=sc.scale_, lr_w=lr.coef_[0], lr_b=lr.intercept_[0],
         lr_tau=np.median([v["tau"] for v in fit["logreg"].values()]),
         tau_distance=np.median(list(fit["distance_only"].values())),
         tau_facing=np.median(list(fit["facing_only"].values())), tau_gaze=np.median(list(fit["gaze_only"].values())))
open(os.path.join(DST, "README.txt"), "w").write(
    "Runtime benchmark. Same code, models and frames on every machine.\n\n"
    "  python3 -m venv venv && ./venv/bin/pip install mediapipe==1.2.3 opencv-python-headless numpy ai-edge-litert\n"
    "  ./venv/bin/python bench.py            # CPU\n"
    "  ./venv/bin/python bench.py --gpu      # MediaPipe GPU delegate (may be unsupported)\n\n"
    "Send back results_<hostname>*.json. frames.mp4 shows participants' faces unblurred: delete this folder after the run.\n")
print("built", DST, subprocess.run(["du", "-sh", DST], capture_output=True, text=True).stdout.split()[0])
