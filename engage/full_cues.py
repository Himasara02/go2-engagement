#!/usr/bin/env python3
"""Per-person addressee cues that need only the colour video (facing, gaze, monocular distance)
for the full dataset received on 10 Nov 2026. Distance from depth and approach speed are added
later, when the depth data arrives.

    ~/miniconda3/envs/go2/bin/python full_cues.py [--workers 10] [take ...]

For every frame i of a take (raw.mp4 frame i <-> log.jsonl record i), every logged person
(track ID + box, which came from find_people_tiled.py) gets landmarks by re-running the SAME
detection that produced the box: PoseLandmarker (pose_landmarker_full, IMAGE, <=3 poses,
conf 0.3) on 6 overlapping vertical strips of the mirrored full-resolution frame, same box
rules (inner-strip-edge cut, min height 25 %, merge at IoU/overlap >= 0.6). The candidate box
with the highest IoU against the logged box (scaled to full resolution) is that person's
pose; IoU < 0.7 => unmatched. Then, unchanged from the existing code:
  facing f, sigma_f:  engage/perception.py facing_from_pose (0.65 head + 0.35 body yaw);
                      per track mean of the last 8, sigma = max(sigma, temporal sd) once 3 values.
  gaze g, sigma_g:    task9_reperceive.py gaze_on (FaceLandmarker + iris on a 256 px head crop),
                      g = exp(-gamma^2 / (2 * 20^2)); no face => g = 0, sigma_g = 0.4;
                      per track mean of the last 8, sigma = max(sigma, temporal sd) once 3 values.
  mono_d, mono_sd:    perception.py's no-depth fallback (fx 615 px at 960 px, shoulder 0.38 m),
                      per track median of the last 12, sigma = max(0.15 + 0.08 d, sd) once 4 values.
Per-frame raw values are stored too, so smoothing can be redone. Nothing in the dataset is changed.
Writes ../../data/cues_2026-11-10/<take>.jsonl.gz and a summary cues_summary.json.
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

from engage.perception import L_EAR, L_SH, NOSE, R_EAR, R_SH, SHOULDER_WIDTH_M, facing_from_pose
from engage.takeio import proc_size

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TAKES = os.path.join(ROOT, "data", "dataset_2026-11-10", "takes_crop3")
OUT = os.path.join(ROOT, "data", "cues_2026-11-10")
POSE_MODEL = os.path.join(ROOT, "data", "labels_pack_2026-11-10", "code", "models", "pose_landmarker_full.task")
FACE_MODEL = os.path.join(HERE, "models", "face_landmarker.task")
# strip detection: identical to find_people_tiled.py (labels pack)
STRIP_W, STRIP_STEP, EDGE, MIN_H = 0.40, 0.15, 4, 0.25
# gaze: identical to task9_reperceive.py
NOSE_TIP, CHEEK_R, CHEEK_L = 1, 234, 454
EYES = [(33, 133, 468), (362, 263, 473)]
GAZE_W, EYE_RANGE, FACE_SIDE = 20.0, 35.0, 3.0
FX_PROC = 615.0          # px at the 960 px processing width (perception.py)
MATCH_IOU = 0.7


class P:
    """A landmark in full-frame normalised coordinates (what gaze_on / facing need)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, v):
        self.x, self.y, self.z, self.visibility = x, y, z, v


def models():
    pose = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL), running_mode=mpv.RunningMode.IMAGE,
        num_poses=3, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
    face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL), running_mode=mpv.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    return pose, face


def iou2(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0, 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter), inter / min(aa, ab)


def detect_people(pose, img):
    """find_people_tiled.detect_people, but also returning each kept box's landmarks
    (full-frame normalised) and world landmarks."""
    H, W = img.shape[:2]
    sw, step = int(W * STRIP_W), int(W * STRIP_STEP)
    starts = list(range(0, W - sw + 1, step))
    if starts[-1] != W - sw:
        starts.append(W - sw)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cands = []
    for x0 in starts:
        res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb[:, x0:x0 + sw].copy()))
        for k, lm in enumerate(res.pose_landmarks or []):
            xs = [p.x * sw for p in lm]
            ys = [p.y * H for p in lm]
            b = [max(0, min(xs)), max(0, min(ys)), min(sw - 1, max(xs)), min(H - 1, max(ys))]
            if (b[0] <= EDGE and x0 > 0) or (b[2] >= sw - 1 - EDGE and x0 + sw < W):
                continue
            if b[3] - b[1] < MIN_H * H:
                continue
            vis = sum(p.visibility for p in lm) / len(lm)
            full_lm = [P((p.x * sw + x0) / W, p.y, p.z, p.visibility) for p in lm]
            wl = res.pose_world_landmarks[k] if res.pose_world_landmarks else None
            cands.append(([b[0] + x0, b[1], b[2] + x0, b[3]], vis * (b[3] - b[1]), full_lm, wl))
    cands.sort(key=lambda c: -c[1])
    keep = []
    for c in cands:
        if all(max(iou2(c[0], k[0])) < 0.6 for k in keep):
            keep.append(c)
    return keep


def gaze_on(face, full, lm, w, h, s):
    """Verbatim from task9_reperceive.py (importing that module needs the pilot dataset)."""
    nx, ny = lm[NOSE].x * w * s, lm[NOSE].y * h * s
    ear = math.hypot((lm[L_EAR].x - lm[R_EAR].x) * w * s, (lm[L_EAR].y - lm[R_EAR].y) * h * s)
    sh = math.hypot((lm[L_SH].x - lm[R_SH].x) * w * s, (lm[L_SH].y - lm[R_SH].y) * h * s)
    side = int(max(FACE_SIDE * max(ear, 0.5 * sh), 48))
    x0, y0 = int(nx - side / 2), int(ny - side / 2)
    H, W = full.shape[:2]
    crop = full[max(0, y0):min(H, y0 + side), max(0, x0):min(W, x0 + side)]
    crop = cv2.copyMakeBorder(crop, max(0, -y0), max(0, y0 + side - H), max(0, -x0), max(0, x0 + side - W),
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC if side < 256 else cv2.INTER_AREA)
    r = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    if not r.face_landmarks:
        return None
    f = r.face_landmarks[0]
    half_face = abs(f[CHEEK_L].x - f[CHEEK_R].x) / 2
    if half_face < 1e-3:
        return None
    hyaw = math.degrees(math.asin(max(-1, min(1, (f[NOSE_TIP].x - (f[CHEEK_L].x + f[CHEEK_R].x) / 2) / half_face))))
    eyes, widths = [], []
    for a, b, c in EYES:
        hw = abs(f[b].x - f[a].x) / 2
        if hw > 1e-4:
            eyes.append((f[c].x - (f[a].x + f[b].x) / 2) / hw)
            widths.append(2 * hw * side)
    if not eyes:
        return None
    return hyaw + EYE_RANGE * float(np.clip(np.mean(eyes), -1, 1)), float(np.mean(widths))


class Track:
    def __init__(self):
        self.face, self.gaze, self.mono = deque(maxlen=8), deque(maxlen=8), deque(maxlen=12)


def r4(v):
    return None if v is None else round(float(v), 4)


def run(name):
    t0 = time.time()
    take = os.path.join(TAKES, name)
    dst = os.path.join(OUT, name + ".jsonl.gz")
    if os.path.exists(dst):
        return {"take": name, "skipped": True}
    meta = json.load(open(os.path.join(take, "meta.json")))
    recs = [json.loads(line) for line in open(os.path.join(take, "log.jsonl"))]
    pose, face = models()
    cap = cv2.VideoCapture(os.path.join(take, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    pw, ph = proc_size(W, H)
    s = W / pw
    fx_full = FX_PROC * s
    tracks, out = {}, []
    st = {"take": name, "frames": 0, "people": 0, "matched": 0, "faces": 0, "iou": []}
    for r in recs:
        ok, raw = cap.read()
        if not ok:
            break
        st["frames"] += 1
        full = cv2.flip(raw, 1) if meta.get("mirror", True) else raw
        cands = detect_people(pose, full)
        row = {}
        for p in r["people"]:
            st["people"] += 1
            tr = tracks.setdefault(p["tid"], Track())
            lb = [v * s for v in p["bbox"]]
            best = max(cands, key=lambda c: iou2(c[0], lb)[0], default=None)
            bi = iou2(best[0], lb)[0] if best else 0.0
            if best is None or bi < MATCH_IOU or best[3] is None:
                row[str(p["tid"])] = {"matched": False}
                continue
            st["matched"] += 1
            st["iou"].append(bi)
            lm, wl = best[2], best[3]
            # facing (perception.py)
            f_raw, fs_raw = facing_from_pose(lm, wl)
            tr.face.append(f_raw)
            fsm, fs = float(np.mean(tr.face)), fs_raw
            if len(tr.face) >= 3:
                fs = max(fs, float(np.std(tr.face)))
            # gaze (task9_reperceive.py); landmarks are full-frame normalised -> w=W, h=H, s=1
            gz = gaze_on(face, full, lm, W, H, 1.0)
            if gz is None:
                g_raw, gs_raw, gamma, eye_px = 0.0, 0.4, None, None
            else:
                st["faces"] += 1
                gamma, eye_px = gz
                g_raw = math.exp(-gamma ** 2 / (2 * GAZE_W ** 2))
                gs_raw = float(np.clip(math.hypot(5.0, 70.0 / max(eye_px, 1.0)) / 40.0, 0.02, 0.5))
            tr.gaze.append(g_raw)
            gm, gs = float(np.mean(tr.gaze)), gs_raw
            if len(tr.gaze) >= 3:
                gs = max(gs, float(np.std(tr.gaze)))
            # monocular distance (perception.py fallback), shoulder width in full-res px
            sh_px = math.hypot((lm[L_SH].x - lm[R_SH].x) * W, (lm[L_SH].y - lm[R_SH].y) * H)
            d = fx_full * SHOULDER_WIDTH_M / max(sh_px, 1.0)
            tr.mono.append(d)
            md, mds = float(np.median(tr.mono)), 0.15 + 0.08 * d
            if len(tr.mono) >= 4:
                mds = max(mds, float(np.std(tr.mono)))
            row[str(p["tid"])] = {"matched": True, "iou": r4(bi),
                                  "facing": r4(fsm), "facing_sd": r4(fs), "facing_raw": r4(f_raw),
                                  "facing_sd_raw": r4(fs_raw),
                                  "gaze": r4(gm), "gaze_sd": r4(gs), "gaze_raw": r4(g_raw), "gaze_sd_raw": r4(gs_raw),
                                  "face": gz is not None, "gamma": None if gamma is None else round(gamma, 2),
                                  "eye_px": None if eye_px is None else round(eye_px, 2),
                                  "mono_d": r4(md), "mono_sd": r4(mds)}
        out.append({"i": r["i"], "t": r["t"], "people": row})
    cap.release()
    os.makedirs(OUT, exist_ok=True)
    tmp = dst + ".part"
    with gzip.open(tmp, "wt") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    os.replace(tmp, dst)
    ious = st.pop("iou")
    st.update(log_frames=len(recs), iou_median=float(np.median(ious)) if ious else None,
              seconds=round(time.time() - t0, 1))
    print(f"{name}: {st['frames']}/{len(recs)} frames, matched {st['matched']}/{st['people']}, "
          f"faces {st['faces']}, {st['seconds']} s", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="*")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for m in (POSE_MODEL, FACE_MODEL):
        if not os.path.exists(m):
            sys.exit(f"missing model: {m}")
    names = a.takes or sorted(os.listdir(TAKES))
    with Pool(a.workers) as pool:
        stats = pool.map(run, names, chunksize=1)
    summ = os.path.join(OUT, "cues_summary.json")
    old = json.load(open(summ)) if os.path.exists(summ) else {}
    old.update({s["take"]: s for s in stats if not s.get("skipped")})
    json.dump(old, open(summ, "w"), indent=1)
    tot = {k: sum(v.get(k, 0) for v in old.values()) for k in ("frames", "people", "matched", "faces")}
    print("TOTAL", tot)


if __name__ == "__main__":
    main()
