#!/usr/bin/env python3
"""Task 8b + 9: re-run perception on the raw colour video for two new signals,
keeping every logged person, track ID and frame index unchanged.

For every frame of a take (raw.mp4 frame i <-> log record i):
  1. PoseLandmarker (same model and settings as perception.py, IMAGE mode) on the
     processing view; each pose's bbox is computed exactly as perception.py does
     and matched to a logged person by IoU (>= 0.7).
  2. Monocular distance for the matched person, exactly as perception.py's
     fallback: d = fx * 0.38 m / shoulder_px (fx = 615 px at 960 px width), then
     the same per-track smoothing (median of the last 12, sigma = max(0.15 +
     0.08 d, temporal sd) once 4 values exist) and the same approach estimate.
  3. Gaze: FaceLandmarker (with iris) on a square crop around the head (centred on
     the nose, side = 3 x ear-to-ear distance, from the full-resolution mirrored
     frame, resized to 256 px). Gaze angle gamma = head yaw + eye yaw, both
     measured in the image so that gamma = 0 means looking along the ray to the
     camera:
        head yaw = asin(clip((nose_x - cheek_mid_x) / half face width))
        eye yaw  = 35 deg x mean over eyes of (iris_x - eye_mid_x) / half eye width
     gaze score g = exp(-gamma^2 / (2 * 20^2)); sigma_gamma = sqrt(5^2 + (70 / eye_px)^2)
     deg (iris localisation ~1 px; eye_px = mean eye width in full-res pixels);
     sigma_g = clip(sigma_gamma / 40, 0.02, 0.5). No face found: g = 0, sigma_g = 0.4
     (perception.py's convention for a missing pose). Per track: mean of the last 8,
     sigma = max(sigma_g, temporal sd) once 3 values exist (as for facing).
All constants were fixed before looking at any result.
Writes results/reperceive/<take>.jsonl.gz: one line per frame,
{i, people: {tid: {mono_d, mono_sd, mono_approach, gaze, gaze_sd, gamma, eye_px, face}}}.

    python task9_reperceive.py <raw_root> [take ...]
raw_root = folder with <take>/raw.mp4 (the 31 Oct data's engage/takes).
"""
import gzip
import json
import math
import os
import sys
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

from analysis import MAIN_DIR, RESULTS
from engage.perception import L_EAR, L_HIP, L_SH, NOSE, R_EAR, R_HIP, R_SH, SHOULDER_WIDTH_M
from engage.takeio import to_proc_view

HERE = os.path.dirname(os.path.abspath(__file__))
FX = 615.0
CAL_DIR = os.path.join(os.path.dirname(MAIN_DIR), "takes_v2")
OUT = os.path.join(RESULTS, "reperceive")
# face mesh indices: nose tip, cheek contour, eye corners, iris centres
NOSE_TIP, CHEEK_R, CHEEK_L = 1, 234, 454
EYES = [(33, 133, 468), (362, 263, 473)]       # (corner, corner, iris centre)
GAZE_W, EYE_RANGE, FACE_SIDE = 20.0, 35.0, 3.0


def models():
    bo = lambda f: BaseOptions(model_asset_path=os.path.join(HERE, "models", f))
    pose = mpv.PoseLandmarker.create_from_options(mpv.PoseLandmarkerOptions(
        base_options=bo("pose_landmarker_full.task"), running_mode=mpv.RunningMode.IMAGE, num_poses=6,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3))
    face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=bo("face_landmarker.task"), running_mode=mpv.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))
    return pose, face


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gaze_on(face, full, lm, w, h, s):
    """lm: pose landmarks (normalised, processing view); full: full-res mirrored frame;
    s = full-res px per processing px. Returns (gamma_deg, eye_px) or None."""
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
            widths.append(2 * hw * side)          # crop-normalised -> full-res px
    if not eyes:
        return None
    return hyaw + EYE_RANGE * float(np.clip(np.mean(eyes), -1, 1)), float(np.mean(widths))


class Track:
    def __init__(self):
        self.dist, self.gaze = deque(maxlen=12), deque(maxlen=8)
        self.prev_d = self.prev_t = None
        self.approach = 0.0


def run(take_dir, raw_dir, pose, face):
    name = os.path.basename(take_dir)
    recs = [json.loads(l) for l in open(os.path.join(take_dir, "log.jsonl"))]
    meta = json.load(open(os.path.join(take_dir, "meta.json")))
    cap = cv2.VideoCapture(os.path.join(raw_dir, name, "raw.mp4"))
    tracks, out, stats = {}, [], {"frames": 0, "people": 0, "matched": 0, "faces": 0}
    for r in recs:
        ok, raw = cap.read()
        if not ok:
            break
        stats["frames"] += 1
        img, _ = to_proc_view(raw, None, mirror=meta.get("mirror", True))
        full = cv2.flip(raw, 1) if meta.get("mirror", True) else raw
        h, w = img.shape[:2]
        s = full.shape[1] / w
        pr = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        poses = pr.pose_landmarks or []
        boxes = []
        for lm in poses:
            xs, ys = [p.x * w for p in lm], [p.y * h for p in lm]
            boxes.append((int(max(0, min(xs))), int(max(0, min(ys))), int(min(w - 1, max(xs))), int(min(h - 1, max(ys)))))
        row = {}
        for p in r["people"]:
            stats["people"] += 1
            tr = tracks.setdefault(p["tid"], Track())
            best = max(range(len(poses)), key=lambda k: iou(boxes[k], p["bbox"]), default=None)
            if best is None or iou(boxes[best], p["bbox"]) < 0.7:
                row[str(p["tid"])] = {"matched": False}
                continue
            stats["matched"] += 1
            lm = poses[best]
            sh_px = math.hypot((lm[L_SH].x - lm[R_SH].x) * w, (lm[L_SH].y - lm[R_SH].y) * h)
            d = FX * SHOULDER_WIDTH_M / max(sh_px, 1.0)
            ds = 0.15 + 0.08 * d
            tr.dist.append(d)
            if len(tr.dist) >= 4:
                ds = max(ds, float(np.std(tr.dist)))
            dsm = float(np.median(tr.dist))
            t = r["t"]
            if tr.prev_d is not None and t > tr.prev_t:
                v = -(dsm - tr.prev_d) / (t - tr.prev_t)
                tr.approach = 0.8 * tr.approach + 0.2 * float(np.clip(v, -2, 2))
            tr.prev_d, tr.prev_t = dsm, t
            gz = gaze_on(face, full, lm, w, h, s)
            if gz is None:
                g, gs, gamma, eye_px = 0.0, 0.4, None, None
            else:
                stats["faces"] += 1
                gamma, eye_px = gz
                g = math.exp(-gamma ** 2 / (2 * GAZE_W ** 2))
                gs = float(np.clip(math.hypot(5.0, 70.0 / max(eye_px, 1.0)) / 40.0, 0.02, 0.5))
            tr.gaze.append(g)
            gm = float(np.mean(tr.gaze))
            if len(tr.gaze) >= 3:
                gs = max(gs, float(np.std(tr.gaze)))
            row[str(p["tid"])] = {"matched": True, "mono_d": round(dsm, 4), "mono_sd": round(ds, 4),
                                  "mono_approach": round(tr.approach, 4), "gaze": round(gm, 4),
                                  "gaze_sd": round(gs, 4), "face": gz is not None,
                                  "gamma": None if gamma is None else round(gamma, 2),
                                  "eye_px": None if eye_px is None else round(eye_px, 2)}
        out.append({"i": r["i"], "people": row})
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, name + ".jsonl.gz"), "wt") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    stats["log_frames"] = len(recs)
    print(f"{name}: {stats}", flush=True)
    return stats


def main():
    raw_root = sys.argv[1]
    names = sys.argv[2:] or sorted(os.listdir(MAIN_DIR)) + ["D1_CAL", "D1_CAL_r2", "D1_CAL_r3"]
    pose, face = models()
    allst = {}
    for n in names:
        src = os.path.join(MAIN_DIR, n) if os.path.isdir(os.path.join(MAIN_DIR, n)) else os.path.join(CAL_DIR, n)
        if not os.path.exists(os.path.join(raw_root, n, "raw.mp4")):
            print(f"{n}: raw.mp4 missing, skipped", flush=True)
            continue
        allst[n] = run(src, raw_root, pose, face)
    json.dump(allst, open(os.path.join(OUT, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
