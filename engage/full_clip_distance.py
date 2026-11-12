#!/usr/bin/env python3
"""Depth distance d, sigma_d and approach speed v of the gesturing person during every labelled clip,
from the processed depth clips (data/dataset_2026-11-11/depth_clips, only the gesturer, only during the
clip; received 11 Nov 2026 while the full 9.6 GB depth is still uploading).

    ~/miniconda3/envs/go2/bin/python full_clip_distance.py [--workers 12] [take ...]

Per clip frame: the gesturer's pose landmarks come from the same strip detection as full_cues.py
(matched to the logged box, IoU >= 0.7). Then perception.py's depth distance, unchanged except for
resolution: median of 7x7-pixel patches (the 9x9 patch of the 960 px processing view, scaled to the
640 px depth image) at both shoulders, both hips and the shoulder midpoint, valid 200-9000 mm;
sigma_d = 0.05 + 1.5 MAD + 0.1 (5 - n) / 5 for n >= 2 valid points, else depth missing (fallback:
monocular d, sigma_d = 0.45, as perception.py). Per clip: median of the last 12 values, sigma_d =
max(sigma_d, sd) once 4 values; approach = EMA (0.8/0.2) of -dd/dt on the smoothed d, starting at 0
at the clip's first frame (no depth exists before the clip in these files).
Writes ../../data/clip_distance_2026-11-11/<take>.json: {clip_id: {frame: {d, sd, v, n_pts, depth}}}.
"""
import argparse
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
CLIPS = os.path.join(ROOT, "data", "dataset_2026-11-11", "depth_clips")
INDEX = os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10", "depth_clips.csv")
OUT = os.path.join(ROOT, "data", "clip_distance_2026-11-11")
HALF = 3   # 7x7 patch


def run(name):
    import csv
    t0 = time.time()
    dst = os.path.join(OUT, name + ".json")
    if os.path.exists(dst):
        return {"take": name, "skipped": True}
    rows = [r for r in csv.DictReader(open(INDEX)) if r["take"] == name]
    meta = json.load(open(os.path.join(TAKES, name, "meta.json")))
    recs = [json.loads(line) for line in open(os.path.join(TAKES, name, "log.jsonl"))]
    clips = {int(r["clip"]): (int(r["track_id"]), np.load(os.path.join(CLIPS, name, f"clip_{int(r['clip']):04d}.npz")))
             for r in rows}
    want = {}
    for c, (tid, z) in clips.items():
        for k, i in enumerate(z["frames"]):
            want.setdefault(int(i), []).append((c, tid, k))
    pose, _ = models()
    cap = cv2.VideoCapture(os.path.join(TAKES, name, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    s = W / proc_size(W, H)[0]
    mirror = meta.get("mirror", True)
    raw_pts = {}                     # (clip, frame) -> (d, sd, n) or None
    st = {"take": name, "clip_frames": 0, "matched": 0, "depth_ok": 0}
    last = max(want) if want else -1
    for i in range(last + 1):
        ok = cap.grab()
        if not ok:
            break
        if i not in want:
            continue
        _, raw = cap.retrieve()
        full = cv2.flip(raw, 1) if mirror else raw
        cands = detect_people(pose, full)
        people = {p["tid"]: p for p in recs[i]["people"]}
        for c, tid, k in want[i]:
            st["clip_frames"] += 1
            p = people.get(tid)
            if p is None:
                raw_pts[(c, i)] = None
                continue
            lb = [v * s for v in p["bbox"]]
            best = max(cands, key=lambda q: iou2(q[0], lb)[0], default=None)
            if best is None or iou2(best[0], lb)[0] < MATCH_IOU:
                raw_pts[(c, i)] = None
                continue
            st["matched"] += 1
            lm = best[2]
            z = clips[c][1]
            dep, box, has = z["depth"][k], z["box"], z["has"][k]
            dh, dw = dep.shape
            q = 0.5                                          # raw -> depth pixels
            pts = [(lm[j].x * W, lm[j].y * H) for j in (L_SH, R_SH, L_HIP, R_HIP)]
            pts.append(((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2))
            vals = []
            if has:
                for x, y in pts:
                    xr = (W - x) if mirror else x            # landmarks are mirrored, depth is not
                    xd, yd = int(xr * q) - box[0], int(y * q) - box[1]
                    if HALF <= xd < dw - HALF and HALF <= yd < dh - HALF:
                        patch = dep[yd - HALF:yd + HALF + 1, xd - HALF:xd + HALF + 1].astype(np.float32)
                        patch = patch[(patch > 200) & (patch < 9000)]
                        if patch.size:
                            vals.append(float(np.median(patch)) / 1000.0)
            if len(vals) >= 2:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(np.array(vals) - med)))
                raw_pts[(c, i)] = (med, 0.05 + 1.5 * mad + 0.1 * (5 - len(vals)) / 5, len(vals), True)
                st["depth_ok"] += 1
            else:
                sh_px = math.hypot((lm[L_SH].x - lm[R_SH].x) * W, (lm[L_SH].y - lm[R_SH].y) * H)
                raw_pts[(c, i)] = (FX_PROC * s * SHOULDER_WIDTH_M / max(sh_px, 1.0), 0.45, len(vals), False)
    cap.release()
    out = {}
    for c, (tid, z) in clips.items():
        hist, prev, v, res = deque(maxlen=12), None, 0.0, {}
        for i, t in ((int(i), recs[int(i)]["t"]) for i in z["frames"]):
            x = raw_pts.get((c, i))
            if x is None:
                continue
            d, sd, n, dep_ok = x
            hist.append(min(d, 10.0))
            if len(hist) >= 4:
                sd = max(sd, float(np.std(hist)))
            dsm = float(np.median(hist))
            if prev is not None and t > prev[1]:
                v = 0.8 * v + 0.2 * float(np.clip(-(dsm - prev[0]) / (t - prev[1]), -2, 2))
            prev = (dsm, t)
            res[str(i)] = {"d": round(dsm, 4), "sd": round(sd, 4), "v": round(v, 4), "n_pts": n, "depth": dep_ok}
        out[str(c)] = res
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(dst + ".part", "w"))
    os.replace(dst + ".part", dst)
    st["seconds"] = round(time.time() - t0, 1)
    print(f"{name}: clip frames {st['clip_frames']}, matched {st['matched']}, depth ok {st['depth_ok']}, "
          f"{st['seconds']} s", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="*")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    names = a.takes or sorted(os.listdir(TAKES))
    with Pool(a.workers) as pool:
        stats = pool.map(run, names, chunksize=1)
    summ = os.path.join(OUT, "summary.json")
    old = json.load(open(summ)) if os.path.exists(summ) else {}
    old.update({x["take"]: x for x in stats if not x.get("skipped")})
    json.dump(old, open(summ, "w"), indent=1)
    print("TOTAL", {k: sum(v.get(k, 0) for v in old.values()) for k in ("clip_frames", "matched", "depth_ok")})


if __name__ == "__main__":
    main()
