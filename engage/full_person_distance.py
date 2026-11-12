#!/usr/bin/env python3
"""Depth distance d, sigma_d and approach speed v of EVERY tracked person in EVERY frame, from the full
depth recordings (data/depth_2026-11-10/<take>/<chunk>.npz, received 12 Nov 2026 on a portable drive).

    ~/miniconda3/envs/go2/bin/python full_person_distance.py [--workers 8] [take ...]

Same measurement as full_clip_distance.py (perception.py's depth distance, unchanged except for
resolution), but for everyone and over the whole recording instead of only the gesturer inside the clip:
  - landmarks: the same strip pose detection as full_cues.py, matched to each logged box (IoU >= 0.7);
  - raw d: median of 7x7-pixel patches (perception.py's 9x9 patch in the 960 px processing view, scaled to
    the 640 px depth image) at both shoulders, both hips and the shoulder midpoint, valid 200-9000 mm;
    sigma_d = 0.05 + 1.5 MAD + 0.1 (5 - n) / 5 for n >= 2 valid points, else monocular d with sigma 0.45;
  - per track (whole recording): median of the last 12 values, sigma_d = max(sigma_d, sd) once 4 values,
    approach = EMA (0.8/0.2) of -dd/dt on the smoothed d (the live system's history, not reset per clip).
Depth chunks: arrays `depth` (N, H/2, W/2) uint16 mm aligned to the unmirrored colour frame, `idx` (N,) =
video frame index (= log.jsonl line). Writes ../../data/person_distance_2026-11-12/<take>.json.gz:
{frame: {tid: [d, sd, v, n_pts, depth_ok]}}.
"""
import argparse
import glob
import gzip
import json
import math
import os
import time
from collections import deque
from multiprocessing import Pool

import cv2
import numpy as np

from engage.perception import L_HIP, L_SH, R_HIP, R_SH, SHOULDER_WIDTH_M
from engage.takeio import proc_size
from full_cues import FX_PROC, MATCH_IOU, ROOT, detect_people, iou2, models

TAKES = os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3")
DEPTH = os.path.join(ROOT, "data", "depth_2026-11-10")
OUT = os.path.join(ROOT, "data", "person_distance_2026-11-12")
HALF = 3   # 7x7 patch


class DepthReader:
    def __init__(self, take):
        self.where, self.cache = {}, (None, None)
        for f in sorted(glob.glob(os.path.join(DEPTH, take, "*.npz"))):
            for k, i in enumerate(np.load(f)["idx"]):
                self.where[int(i)] = (f, k)

    def __call__(self, i):
        if i not in self.where:
            return None
        f, k = self.where[i]
        if self.cache[0] != f:
            self.cache = (f, np.load(f)["depth"])
        return self.cache[1][k]


def raw_distance(lm, W, H, dep, mirror, s):
    pts = [(lm[j].x * W, lm[j].y * H) for j in (L_SH, R_SH, L_HIP, R_HIP)]
    pts.append(((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2))
    vals = []
    if dep is not None:
        dh, dw = dep.shape
        q = dw / W                                           # raw -> depth pixels (0.5)
        for x, y in pts:
            xr = (W - x) if mirror else x                    # landmarks are mirrored, depth is not
            xd, yd = int(xr * q), int(y * q)
            if HALF <= xd < dw - HALF and HALF <= yd < dh - HALF:
                patch = dep[yd - HALF:yd + HALF + 1, xd - HALF:xd + HALF + 1].astype(np.float32)
                patch = patch[(patch > 200) & (patch < 9000)]
                if patch.size:
                    vals.append(float(np.median(patch)) / 1000.0)
    if len(vals) >= 2:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(np.array(vals) - med)))
        return med, 0.05 + 1.5 * mad + 0.1 * (5 - len(vals)) / 5, len(vals), True
    sh_px = math.hypot((lm[L_SH].x - lm[R_SH].x) * W, (lm[L_SH].y - lm[R_SH].y) * H)
    return FX_PROC * s * SHOULDER_WIDTH_M / max(sh_px, 1.0), 0.45, len(vals), False


def run(name):
    t0 = time.time()
    dst = os.path.join(OUT, name + ".json.gz")
    if os.path.exists(dst):
        return {"take": name, "skipped": True}
    meta = json.load(open(os.path.join(TAKES, name, "meta.json")))
    recs = [json.loads(line) for line in open(os.path.join(TAKES, name, "log.jsonl"))]
    depth = DepthReader(name)
    pose, _ = models()
    cap = cv2.VideoCapture(os.path.join(TAKES, name, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    s = W / proc_size(W, H)[0]
    mirror = meta.get("mirror", True)
    st = {"take": name, "frames": 0, "depth_frames": 0, "person_frames": 0, "matched": 0, "depth_ok": 0}
    hist, prev, vel, out = {}, {}, {}, {}
    for i, rec in enumerate(recs):
        ok, raw = cap.read()
        if not ok:
            break
        st["frames"] += 1
        if not rec["people"]:
            continue
        dep = depth(i)
        st["depth_frames"] += dep is not None
        full = cv2.flip(raw, 1) if mirror else raw
        cands = detect_people(pose, full)
        row = {}
        for p in rec["people"]:
            st["person_frames"] += 1
            lb = [v * s for v in p["bbox"]]
            best = max(cands, key=lambda c: iou2(c[0], lb)[0], default=None)
            if best is None or iou2(best[0], lb)[0] < MATCH_IOU:
                continue
            st["matched"] += 1
            d, sd, n, dep_ok = raw_distance(best[2], W, H, dep, mirror, s)
            st["depth_ok"] += dep_ok
            tid, t = p["tid"], rec["t"]
            h = hist.setdefault(tid, deque(maxlen=12))
            h.append(min(d, 10.0))
            if len(h) >= 4:
                sd = max(sd, float(np.std(h)))
            dsm = float(np.median(h))
            v = vel.get(tid, 0.0)
            if tid in prev and t > prev[tid][1]:
                v = 0.8 * v + 0.2 * float(np.clip(-(dsm - prev[tid][0]) / (t - prev[tid][1]), -2, 2))
            prev[tid], vel[tid] = (dsm, t), v
            row[str(tid)] = [round(dsm, 4), round(sd, 4), round(v, 4), n, dep_ok]
        out[str(i)] = row
    cap.release()
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(dst + ".part", "wt") as f:
        json.dump(out, f)
    os.replace(dst + ".part", dst)
    st["log_frames"] = len(recs)
    st["seconds"] = round(time.time() - t0, 1)
    print(f"{name}: frames {st['frames']}/{len(recs)}, depth frames {st['depth_frames']}, person-frames "
          f"{st['person_frames']}, matched {st['matched']}, depth ok {st['depth_ok']}, {st['seconds']} s", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = a.takes or sorted(os.listdir(TAKES))
    # longest first, so the pool finishes together
    names.sort(key=lambda n: -os.path.getsize(os.path.join(TAKES, n, "log.jsonl")))
    with Pool(a.workers) as pool:
        stats = pool.map(run, names, chunksize=1)
    summ = os.path.join(OUT, "summary.json")
    old = json.load(open(summ)) if os.path.exists(summ) else {}
    old.update({x["take"]: x for x in stats if not x.get("skipped")})
    json.dump(old, open(summ, "w"), indent=1)
    print("TOTAL", {k: sum(v.get(k, 0) for v in old.values())
                    for k in ("frames", "depth_frames", "person_frames", "matched", "depth_ok")})


if __name__ == "__main__":
    main()
