#!/usr/bin/env python3
"""Per-frame runtime of each pipeline stage on this laptop's CPU (single process).

    ~/miniconda3/envs/go2/bin/python bench_runtime.py [take] [frames]

Stages per frame: decode; people = pose on 6 overlapping strips (dataset pipeline) and, for comparison,
one whole-frame pose pass (live perception.py); then per person: hand landmarks on the 640x640 upper-body
crop + keypoint classifier, face/iris on the 256 px head crop (gaze), IT2 engagement with 11 rules.
Writes results/full/runtime.{json,md}.
"""
import json
import os
import platform
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

from engage.perception import GestureClassifier
from engage.takeio import proc_size
from find_gestures_centered import SIZE, centred_crop
from find_gestures_crop import hands_model, owner
from full_cues import POSE_MODEL, detect_people, gaze_on, models
from full_eval import RULES4, fls

HERE = os.path.dirname(os.path.abspath(__file__))
TAKES = os.path.join(HERE, "..", "..", "data", "dataset_2026-11-11", "takes_crop3")
take = sys.argv[1] if len(sys.argv) > 1 else "D4_S3_r5"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
WARM = 10

pose_strip, face = models()
pose_whole = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL), running_mode=mpv.RunningMode.IMAGE, num_poses=6,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
hl, clf = hands_model(), GestureClassifier()
cap = cv2.VideoCapture(os.path.join(TAKES, take, "raw.mp4"))
W, H = int(cap.get(3)), int(cap.get(4))
pw, ph = proc_size(W, H)
T = {k: [] for k in ("decode", "people_strips", "people_whole_frame", "hands_per_person", "gaze_per_person",
                     "gate_per_person")}
npeople = []
for f in range(N + WARM):
    t0 = time.perf_counter()
    ok, raw = cap.read()
    if not ok:
        break
    full = cv2.flip(raw, 1)
    t1 = time.perf_counter()
    people = detect_people(pose_strip, full)
    t2 = time.perf_counter()
    small = cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA)
    pose_whole.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    t3 = time.perf_counter()
    boxes = [c[0] for c in people]
    th = tg = tf = 0.0
    for k, (b, _, lm, _) in enumerate(people):
        a = time.perf_counter()
        crop, sx, sy, sc = centred_crop(full, b)
        res = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        for hlm in res.hand_landmarks or []:
            if owner(sx + hlm[0].x * SIZE / sc, sy + hlm[0].y * SIZE / sc, boxes) == k:
                clf([[min(int(q.x * SIZE), SIZE - 1), min(int(q.y * SIZE), SIZE - 1)] for q in hlm])
        b_ = time.perf_counter()
        gaze_on(face, full, lm, W, H, 1.0)
        c_ = time.perf_counter()
        fls([2.5, 0.1, 0.8, 0.05, 0.0, 0.7, 0.1], True, 1.0, RULES4)
        d_ = time.perf_counter()
        th, tg, tf = th + b_ - a, tg + c_ - b_, tf + d_ - c_
    if f >= WARM and people:
        T["decode"].append(t1 - t0)
        T["people_strips"].append(t2 - t1)
        T["people_whole_frame"].append(t3 - t2)
        T["hands_per_person"].append(th / len(people))
        T["gaze_per_person"].append(tg / len(people))
        T["gate_per_person"].append(tf / len(people))
        npeople.append(len(people))
ms = {k: 1000 * float(np.median(v)) for k, v in T.items()}
p = float(np.mean(npeople))
per_person = ms["hands_per_person"] + ms["gaze_per_person"] + ms["gate_per_person"]
total_strips = ms["decode"] + ms["people_strips"] + p * per_person
total_whole = ms["decode"] + ms["people_whole_frame"] + p * per_person
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
out = {"take": take, "frames": len(npeople), "mean_people": p, "cpu": cpu, "median_ms": ms,
       "per_person_ms": per_person, "frame_total_ms_strips": total_strips, "frame_total_ms_whole_frame": total_whole,
       "fps_strips": 1000 / total_strips, "fps_whole_frame": 1000 / total_whole}
os.makedirs(os.path.join(HERE, "results", "full"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "results", "full", "runtime.json"), "w"), indent=1)
md = [f"# Runtime per frame ({cpu}, one process, {len(npeople)} frames of {take}, {p:.1f} people/frame)", "",
      "| Stage | Median ms |", "|---|---|"] + [f"| {k} | {v:.1f} |" for k, v in ms.items()] + [
      "", f"Per person (hands + gaze + gate): {per_person:.1f} ms. Frame total: {total_strips:.0f} ms with strip "
      f"people detection ({1000 / total_strips:.1f} fps), {total_whole:.0f} ms with one whole-frame pass "
      f"({1000 / total_whole:.1f} fps)."]
open(os.path.join(HERE, "results", "full", "runtime.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
