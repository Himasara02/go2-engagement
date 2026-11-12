#!/usr/bin/env python3
"""Portable runtime benchmark (same code, models and frames on every machine).

    python bench.py [--gpu] [--frames 150]

Part A, perception (shared by every policy), per frame of a 3-person recording, median ms:
  people (pose on 6 overlapping strips = dataset pipeline; and one whole-frame pass = live system),
  per person: hand landmarks on the upper-body crop + gesture MLP, face/iris on the head crop (gaze),
  facing from pose + depth distance at 5 landmarks.
Part B, decision step of every policy on 5,000 logged person-frames of the dataset, median microseconds
per person-frame: gesture only, nearest person, distance / facing / gaze threshold, logistic regression,
type-1 and IT2 gates (with and without gaze).
Part C, "our end" only: everything after the neural networks (MediaPipe pose / hands / face and the gesture
MLP), timed on their saved outputs for the same frames, per method: tracking (greedy IoU, as the dataset),
cue extraction (facing from pose, depth distance, approach, gaze from iris landmarks), measured uncertainty
and its temporal smoothing, the gate, 3-frame sustain and arbitration (STOP wins, else most engaged).
Writes results_<hostname>[_gpu].json next to this file. Faces in frames.mp4 are not blurred: delete the
folder after the run.
"""
import argparse
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

import full_cues
from engage.fuzzy import RULES, _km, mf_approach, mf_distance, mf_facing
from engage.methods import COMMAND_MAP
from engage.perception import L_HIP, L_SH, R_HIP, R_SH, GestureClassifier, facing_from_pose
from find_gestures_centered import SIZE, centred_crop
from find_gestures_crop import owner

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true", help="MediaPipe GPU delegate for pose/hands/face")
ap.add_argument("--frames", type=int, default=150)
a = ap.parse_args()
WARM = 10
MOD = os.path.join(HERE, "models")
DELEGATE = BaseOptions.Delegate.GPU if a.gpu else BaseOptions.Delegate.CPU
bo = lambda name: BaseOptions(model_asset_path=os.path.join(MOD, name), delegate=DELEGATE)


# ------------------------------------------------------------------ system info
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "") \
    if os.path.exists("/proc/cpuinfo") else platform.processor()
model = sh("tr -d '\\0' < /proc/device-tree/model")
if not cpu:
    cpu = sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip()
info = {"host": socket.gethostname(), "board": model, "cpu": cpu, "cores": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1),
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader") or None, "delegate": "GPU" if a.gpu else "CPU",
        "os": sh(". /etc/os-release && echo $PRETTY_NAME"), "python": platform.python_version(),
        "mediapipe": mp.__version__, "opencv": cv2.__version__, "numpy": np.__version__}
print(json.dumps(info, indent=1), flush=True)

# ------------------------------------------------------------------ models (same options as the paper's scripts)
try:
    pose_strip = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
        base_options=bo("pose_landmarker_full.task"), running_mode=mpv.RunningMode.IMAGE, num_poses=3,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
    pose_whole = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
        base_options=bo("pose_landmarker_full.task"), running_mode=mpv.RunningMode.IMAGE, num_poses=6,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
    face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=bo("face_landmarker.task"), running_mode=mpv.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    hl = mpv.HandLandmarker.create_from_options(mpv.HandLandmarkerOptions(
        base_options=bo("hand_landmarker.task"), running_mode=mpv.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
except Exception as e:                                   # GPU delegate not available on this machine
    out = {"system": info, "error": f"{type(e).__name__}: {e}"}
    json.dump(out, open(os.path.join(HERE, f"results_{info['host']}{'_gpu' if a.gpu else ''}.json"), "w"), indent=1)
    print("MODEL LOADING FAILED:", out["error"])
    sys.exit(1)
clf = GestureClassifier()
depth = np.load(os.path.join(HERE, "depth_frame.npy"))   # one aligned depth frame (mm, half resolution)


NOSE, L_EAR, R_EAR = 0, 7, 8
NOSE_TIP, CHEEK_R, CHEEK_L = 1, 234, 454
EYES = [(33, 133, 468), (362, 263, 473)]
GAZE_W, EYE_RANGE, FACE_SIDE = 20.0, 35.0, 3.0


def face_detect(full, lm, w, h):
    """First half of full_cues.gaze_on: head crop + FaceLandmarker (neural network)."""
    nx, ny = lm[NOSE].x * w, lm[NOSE].y * h
    ear = math.hypot((lm[L_EAR].x - lm[R_EAR].x) * w, (lm[L_EAR].y - lm[R_EAR].y) * h)
    shw = math.hypot((lm[L_SH].x - lm[R_SH].x) * w, (lm[L_SH].y - lm[R_SH].y) * h)
    side = int(max(FACE_SIDE * max(ear, 0.5 * shw), 48))
    x0, y0 = int(nx - side / 2), int(ny - side / 2)
    Hh, Ww = full.shape[:2]
    crop = full[max(0, y0):min(Hh, y0 + side), max(0, x0):min(Ww, x0 + side)]
    crop = cv2.copyMakeBorder(crop, max(0, -y0), max(0, y0 + side - Hh), max(0, -x0), max(0, x0 + side - Ww),
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC if side < 256 else cv2.INTER_AREA)
    r = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return (r.face_landmarks[0] if r.face_landmarks else None), side


def gaze_post(f, side):
    """Second half of full_cues.gaze_on: head yaw + iris offset -> gaze angle, eye width (ours)."""
    if f is None:
        return None
    half_face = abs(f[CHEEK_L].x - f[CHEEK_R].x) / 2
    if half_face < 1e-3:
        return None
    hyaw = math.degrees(math.asin(max(-1, min(1, (f[NOSE_TIP].x - (f[CHEEK_L].x + f[CHEEK_R].x) / 2) / half_face))))
    eyes, widths = [], []
    for a_, b_, c_ in EYES:
        hw = abs(f[b_].x - f[a_].x) / 2
        if hw > 1e-4:
            eyes.append((f[c_].x - (f[a_].x + f[b_].x) / 2) / hw)
            widths.append(2 * hw * side)
    if not eyes:
        return None
    return hyaw + EYE_RANGE * float(np.clip(np.mean(eyes), -1, 1)), float(np.mean(widths))


def depth_distance(lm, W, H, dep):
    """full_person_distance.raw_distance (perception.py's depth distance)."""
    pts = [(lm[j].x * W, lm[j].y * H) for j in (L_SH, R_SH, L_HIP, R_HIP)]
    pts.append(((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2))
    dh, dw = dep.shape
    vals = []
    for x, y in pts:
        xd, yd = int((W - x) * dw / W), int(y * dw / W)
        if 3 <= xd < dw - 3 and 3 <= yd < dh - 3:
            patch = dep[yd - 3:yd + 4, xd - 3:xd + 4].astype(np.float32)
            patch = patch[(patch > 200) & (patch < 9000)]
            if patch.size:
                vals.append(float(np.median(patch)) / 1000.0)
    if len(vals) >= 2:
        med = float(np.median(vals))
        return med, 0.05 + 1.5 * float(np.median(np.abs(np.array(vals) - med))) + 0.1 * (5 - len(vals)) / 5
    return None


# ------------------------------------------------------------------ part A: perception
cap = cv2.VideoCapture(os.path.join(HERE, "frames.mp4"))
W, H = int(cap.get(3)), int(cap.get(4))
pw, ph = 960, int(round(H * 960 / W))
T = {k: [] for k in ("decode", "people_strips", "people_whole_frame", "hands_per_person", "gaze_per_person",
                     "facing_distance_per_person")}
npeople = []
cache = []                                   # per frame: [(box, lm, wl, face_lm, side, gesture, conf)]
t_start = time.time()
for f in range(a.frames + WARM):
    t0 = time.perf_counter()
    ok, raw = cap.read()
    if not ok:
        break
    full = cv2.flip(raw, 1)
    t1 = time.perf_counter()
    try:
        people = full_cues.detect_people(pose_strip, full)
    except Exception as e:                                 # e.g. GPU delegate loads but cannot run
        out = {"system": info, "error": f"{type(e).__name__}: {str(e)[-300:]}"}
        json.dump(out, open(os.path.join(HERE, f"results_{info['host']}{'_gpu' if a.gpu else ''}.json"), "w"), indent=1)
        print("INFERENCE FAILED:", out["error"])
        sys.exit(1)
    t2 = time.perf_counter()
    small = cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA)
    pose_whole.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    t3 = time.perf_counter()
    boxes = [c[0] for c in people]
    persons = []
    th = tg = tc = 0.0
    for k, (b, _, lm, wl) in enumerate(people):
        q0 = time.perf_counter()
        crop, sx, sy, sc = centred_crop(full, b)
        res = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        gest = ("none", 0.0)
        for hlm in res.hand_landmarks or []:
            if owner(sx + hlm[0].x * SIZE / sc, sy + hlm[0].y * SIZE / sc, boxes) == k:
                gc = clf([[min(int(q.x * SIZE), SIZE - 1), min(int(q.y * SIZE), SIZE - 1)] for q in hlm])
                gest = max(gest, gc, key=lambda x: x[1])
        q1 = time.perf_counter()
        fl, side = face_detect(full, lm, W, H)
        gaze_post(fl, side)
        q2 = time.perf_counter()
        if f >= WARM:
            persons.append((b, lm, wl, fl, side) + gest)
        facing_from_pose(lm, wl)
        depth_distance(lm, W, H, depth)
        q3 = time.perf_counter()
        th, tg, tc = th + q1 - q0, tg + q2 - q1, tc + q3 - q2
    if f >= WARM and people:
        n = len(people)
        for key, v in (("decode", t1 - t0), ("people_strips", t2 - t1), ("people_whole_frame", t3 - t2),
                       ("hands_per_person", th / n), ("gaze_per_person", tg / n), ("facing_distance_per_person", tc / n)):
            T[key].append(v)
        npeople.append(n)
        cache.append(persons)
    if f % 20 == 0:
        print(f"  perception frame {f}/{a.frames + WARM}  ({time.time() - t_start:.0f} s)", flush=True)
ms = {k: 1000 * statistics.median(v) for k, v in T.items()}
p = float(np.mean(npeople))
per_person = ms["hands_per_person"] + ms["gaze_per_person"] + ms["facing_distance_per_person"]

# ------------------------------------------------------------------ part B: decision step of every policy
Z = np.load(os.path.join(HERE, "decision_inputs.npz"))
X, G, C = Z["X"], [str(g) for g in Z["gesture"]], Z["conf"]       # X: d, sd, f, sf, v, g, sg
groups = np.split(Z["near_d"], Z["near_off"][1:-1])
lr_mean, lr_scale, lr_w, lr_b, lr_tau = Z["lr_mean"], Z["lr_scale"], Z["lr_w"], float(Z["lr_b"]), float(Z["lr_tau"])
tau = {k: float(Z["tau_" + k]) for k in ("distance", "facing", "gaze")}
RULES4 = RULES + [({"face": "HIGH", "gaze": "HIGH"}, 0.95), ({"face": "HIGH", "gaze": "LOW"}, 0.40)]


def fls(x, it2=True, rules=RULES):
    """full_eval.fls (k = 1)."""
    z = 1.0 if it2 else 0.0
    sets = {"dist": mf_distance(x[0], z * x[1]), "face": mf_facing(x[2], z * x[3]),
            "appr": mf_approach(x[4], z * 0.08), "gaze": mf_facing(x[5], z * x[6])}
    y, fl, fu = [], [], []
    for ante, cons in rules:
        y.append(cons)
        fl.append(min(sets[v][lab][0] for v, lab in ante.items()))
        fu.append(min(sets[v][lab][1] for v, lab in ante.items()))
    if not it2:
        s = sum(fu)
        return sum(c * w for c, w in zip(y, fu)) / s if s > 1e-9 else 0.0
    return _km(y, fl, fu)[0]


def lr(x):
    zz = float(((x[[0, 4, 2, 1, 3]] - lr_mean) / lr_scale) @ lr_w + lr_b)   # FEATS: d, v, f, sd, sf
    return 1.0 / (1.0 + math.exp(-zz)) >= lr_tau


def nearest(ds):
    k = min(range(len(ds)), key=ds.__getitem__)
    return [i == k for i in range(len(ds))]


rows = [X[i] for i in range(len(X))]
policies = {
    "gesture_only": lambda i: G[i] in COMMAND_MAP and C[i] >= 0.45,
    "nearest": None,
    "distance_only": lambda i: -rows[i][0] >= tau["distance"],
    "facing_only": lambda i: rows[i][2] >= tau["facing"],
    "gaze_only": lambda i: rows[i][5] >= tau["gaze"],
    "logreg": lambda i: lr(rows[i]),
    "type1": lambda i: fls(rows[i], False) >= 0.6,
    "type1_gaze": lambda i: fls(rows[i], False, RULES4) >= 0.6,
    "it2": lambda i: fls(rows[i], True) >= 0.6,
    "it2_gaze": lambda i: fls(rows[i], True, RULES4) >= 0.6,
}
us = {}
for name, fn in policies.items():
    reps = []
    for _ in range(5):
        if name == "nearest":
            t0 = time.perf_counter()
            for g in groups:
                nearest(g.tolist())
            reps.append((time.perf_counter() - t0) / sum(len(g) for g in groups))
        else:
            t0 = time.perf_counter()
            for i in range(len(rows)):
                fn(i)
            reps.append((time.perf_counter() - t0) / len(rows))
    us[name] = 1e6 * statistics.median(reps)
    print(f"  decision {name:13s} {us[name]:8.2f} us / person-frame", flush=True)

# ------------------------------------------------------------------ part C: our end only (after the networks)
from collections import deque
OFF = 0.15


def iou1(a_, b_):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0]))
    iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    return inter / ((a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - inter)


class IoUTracker:
    """find_people_tiled.Tracker (greedy IoU, max age 15 frames): the dataset's tracker."""

    def __init__(self, max_age=15):
        self.tracks, self.next, self.max_age = {}, 1, max_age

    def update(self, boxes, i):
        pairs = sorted(((iou1(b_, t_["box"]), k, tid) for k, b_ in enumerate(boxes)
                        for tid, t_ in self.tracks.items()), reverse=True)
        ids, used = {}, set()
        for v, k, tid in pairs:
            if v < 0.25 or k in ids or tid in used:
                continue
            ids[k] = tid
            used.add(tid)
        out = []
        for k, b_ in enumerate(boxes):
            tid = ids.get(k)
            if tid is None:
                tid, self.next = self.next, self.next + 1
            self.tracks[tid] = {"box": b_, "seen": i}
            out.append(tid)
        self.tracks = {t_: v for t_, v in self.tracks.items() if i - v["seen"] <= self.max_age}
        return out


class St:
    __slots__ = ("face", "gaze", "dist", "prev", "v", "run")

    def __init__(self):
        self.face, self.gaze, self.dist = deque(maxlen=8), deque(maxlen=8), deque(maxlen=12)
        self.prev, self.v, self.run = None, 0.0, 0


def our_end(kind, dt=1 / 13.0):
    """One pass over the cached network outputs; returns the number of executed commands."""
    tr, state, executed = IoUTracker(), {}, 0
    for i, persons in enumerate(cache):
        t = i * dt
        tids = tr.update([p_[0] for p_ in persons], i)
        acc = []
        for tid, (box, lm, wl, fl, side, gest, conf) in zip(tids, persons):
            s_ = state.get(tid)
            if s_ is None:
                s_ = state[tid] = St()
            cand = gest in COMMAND_MAP and conf >= 0.45
            stop = gest in ("palm", "stop")
            e = 1.0
            if kind != "gesture_only":
                f_raw, fs = facing_from_pose(lm, wl)                       # facing + sigma_f
                s_.face.append(f_raw)
                fm = sum(s_.face) / len(s_.face)
                if len(s_.face) >= 3:
                    fs = max(fs, float(np.std(s_.face)))
                x = [0.0, 0.0, fm, fs, 0.0, 0.0, 0.4]
                if kind != "facing_only":
                    dd = depth_distance(lm, W, H, depth)                    # depth distance + sigma_d
                    d_, sd = dd if dd else (2.5, 0.45)
                    s_.dist.append(min(d_, 10.0))
                    if len(s_.dist) >= 4:
                        sd = max(sd, float(np.std(s_.dist)))
                    dsm = float(np.median(s_.dist))
                    if s_.prev is not None:                                  # approach speed
                        s_.v = 0.8 * s_.v + 0.2 * float(np.clip(-(dsm - s_.prev) / dt, -2, 2))
                    s_.prev = dsm
                    x[0], x[1], x[4] = dsm, sd, s_.v
                if kind in ("type1_gaze", "it2_gaze"):
                    gz = gaze_post(fl, side)                                 # gaze + sigma_g
                    if gz is None:
                        g_raw, gs = 0.0, 0.4
                    else:
                        g_raw = math.exp(-gz[0] ** 2 / (2 * GAZE_W ** 2))
                        gs = float(np.clip(math.hypot(5.0, 70.0 / max(gz[1], 1.0)) / 40.0, 0.02, 0.5))
                    s_.gaze.append(g_raw)
                    gm = sum(s_.gaze) / len(s_.gaze)
                    if len(s_.gaze) >= 3:
                        gs = max(gs, float(np.std(s_.gaze)))
                    x[5], x[6] = gm, gs
                if kind == "facing_only":
                    e = 1.0 if fm >= tau["facing"] else 0.0
                elif kind == "logreg":
                    e = 1.0 if lr(np.array(x)) else 0.0
                else:
                    e = fls(x, kind == "it2_gaze", RULES4)
            ok = cand and (e + OFF * stop >= 0.6 if kind in ("type1_gaze", "it2_gaze") else e >= 0.5)
            s_.run = s_.run + 1 if ok else 0
            if s_.run >= 3:
                acc.append((stop, e, tid))
        if acc:                                                              # arbitration
            stops = [a_ for a_ in acc if a_[0]]
            (stops[0] if stops else max(acc, key=lambda a_: a_[1]))
            executed += 1
    return executed


pf = sum(len(p_) for p_ in cache)
ours = {}
for kind in ("gesture_only", "facing_only", "logreg", "type1_gaze", "it2_gaze"):
    reps = []
    for _ in range(9):
        t0 = time.perf_counter()
        our_end(kind)
        reps.append(time.perf_counter() - t0)
    tot = statistics.median(reps)
    ours[kind] = {"ms_per_frame": 1000 * tot / len(cache), "us_per_person": 1e6 * tot / pf}
    print(f"  our end {kind:13s} {ours[kind]['ms_per_frame']:.3f} ms / frame, "
          f"{ours[kind]['us_per_person']:.0f} us / person", flush=True)

frame_strips = ms["decode"] + ms["people_strips"] + p * per_person
frame_whole = ms["decode"] + ms["people_whole_frame"] + p * per_person
out = {"system": info, "frames": len(npeople), "mean_people": p, "perception_median_ms": ms,
       "per_person_perception_ms": per_person, "frame_ms_strips": frame_strips, "frame_ms_whole_frame": frame_whole,
       "fps_strips": 1000 / frame_strips, "fps_whole_frame": 1000 / frame_whole,
       "decision_us_per_person": us, "our_end": ours, "our_end_person_frames": pf, "decision_rows": len(rows), "decision_frames_nearest": len(groups),
       "gate_share_of_frame_pct_whole": 100 * p * us["it2_gaze"] / 1000 / frame_whole}
dst = os.path.join(HERE, f"results_{info['host']}{'_gpu' if a.gpu else ''}.json")
json.dump(out, open(dst, "w"), indent=1)
print(f"\nperception (median ms): " + ", ".join(f"{k} {v:.1f}" for k, v in ms.items()))
print(f"frame: {frame_strips:.0f} ms strips ({1000 / frame_strips:.1f} fps), {frame_whole:.0f} ms whole frame "
      f"({1000 / frame_whole:.1f} fps), {p:.1f} people")
print(f"IT2 + gaze gate: {us['it2_gaze']:.0f} us per person = {out['gate_share_of_frame_pct_whole']:.2f} % of the frame")
print(f"our end (IT2 + gaze, everything after the networks): {ours['it2_gaze']['ms_per_frame']:.2f} ms per frame "
      f"with {p:.1f} people = {100 * ours['it2_gaze']['ms_per_frame'] / frame_whole:.2f} % of the frame")
print("wrote", dst)
