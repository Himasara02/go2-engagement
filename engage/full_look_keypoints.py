#!/usr/bin/env python3
"""17 COCO body keypoints of the gesturing person in every labelled episode frame, for the LOOK baseline
(Belkada et al., BMVC 2021: a pretrained "is this person looking at the camera" classifier on 2D keypoints).

    ~/miniconda3/envs/go2/bin/python full_look_keypoints.py [--workers 8] [take ...]

Same person detection and box matching as full_cues.py (strip pose detection on the mirrored processing view,
IoU >= 0.7 with the logged box). The 33 MediaPipe landmarks are converted to the 17 COCO keypoints LOOK expects,
in pixels of the UNMIRRORED camera frame (x -> W - x, left/right swapped back), confidence = landmark visibility.
Only frames inside r/n episodes of the gesturer are processed (the evaluation reads nothing else).
Env: LK_TAKES, LK_LABELS, LK_OUT (walking run). Writes <out>/<take>.json.gz: {frame: {tid: [x17, y17, c17]}}.
"""
import argparse
import csv
import gzip
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2

from engage.takeio import proc_size
from full_cues import MATCH_IOU, ROOT, detect_people, iou2, models

TAKES = os.environ.get("LK_TAKES", os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3"))
LABELS = os.environ.get("LK_LABELS", os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10",
                                                  "labels_final.csv"))
OUT = os.environ.get("LK_OUT", os.path.join(ROOT, "data", "look_kps_2026-11-13"))
# COCO order: nose, l/r eye, l/r ear, l/r shoulder, l/r elbow, l/r wrist, l/r hip, l/r knee, l/r ankle
MP_LEFT = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# landmarks found on the mirrored image: after flipping x back, MediaPipe's "left" is the person's right
MP_UNMIRRORED = [0, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27]


def needed(take):
    want = defaultdict(set)
    for r in csv.DictReader(open(LABELS)):
        if r["take"] == take and r["label"] in ("r", "n"):
            for i in range(int(r["first_frame"]), int(r["last_frame"]) + 1):
                want[i].add(int(r["track_id"]))
    return want


def run(name):
    t0 = time.time()
    dst = os.path.join(OUT, name + ".json.gz")
    if os.path.exists(dst):
        return {"take": name, "skipped": True}
    want = needed(name)
    meta = json.load(open(os.path.join(TAKES, name, "meta.json")))
    recs = [json.loads(line) for line in open(os.path.join(TAKES, name, "log.jsonl"))]
    pose, _ = models()
    cap = cv2.VideoCapture(os.path.join(TAKES, name, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    s = W / proc_size(W, H)[0]
    mirror = meta.get("mirror", True)
    idx = MP_UNMIRRORED if mirror else MP_LEFT
    st = {"take": name, "person_frames": 0, "matched": 0}
    out = {}
    last = max(want) if want else -1
    for i, rec in enumerate(recs):
        if i > last:
            break
        ok, raw = cap.read()
        if not ok:
            break
        if i not in want:
            continue
        full = cv2.flip(raw, 1) if mirror else raw
        cands = detect_people(pose, full)
        row = {}
        for p in rec["people"]:
            if p["tid"] not in want[i]:
                continue
            st["person_frames"] += 1
            lb = [v * s for v in p["bbox"]]
            best = max(cands, key=lambda c: iou2(c[0], lb)[0], default=None)
            if best is None or iou2(best[0], lb)[0] < MATCH_IOU:
                continue
            st["matched"] += 1
            lm = best[2]
            xs = [(1 - lm[j].x) * W if mirror else lm[j].x * W for j in idx]
            ys = [lm[j].y * H for j in idx]
            cs = [lm[j].visibility for j in idx]
            row[str(p["tid"])] = [[round(v, 1) for v in xs], [round(v, 1) for v in ys], [round(v, 3) for v in cs]]
        out[str(i)] = row
    cap.release()
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(dst + ".part", "wt") as f:
        json.dump({"W": W, "H": H, "frames": out}, f)
    os.replace(dst + ".part", dst)
    st["seconds"] = round(time.time() - t0, 1)
    print(f"{name}: person-frames {st['person_frames']}, matched {st['matched']}, {st['seconds']} s", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    names = a.takes or sorted({r["take"] for r in csv.DictReader(open(LABELS))
                               if os.path.isdir(os.path.join(TAKES, r["take"]))})
    names.sort(key=lambda n: -os.path.getsize(os.path.join(TAKES, n, "log.jsonl")))
    with Pool(a.workers) as pool:
        stats = pool.map(run, names, chunksize=1)
    summ = os.path.join(OUT, "summary.json")
    old = json.load(open(summ)) if os.path.exists(summ) else {}
    old.update({x["take"]: x for x in stats if not x.get("skipped")})
    json.dump(old, open(summ, "w"), indent=1)
    print("TOTAL", {k: sum(v.get(k, 0) for v in old.values()) for k in ("person_frames", "matched")})


if __name__ == "__main__":
    main()
