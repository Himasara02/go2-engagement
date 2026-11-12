#!/usr/bin/env python3
"""Render teaser candidates (faces blurred) or the final teaser figure for the paper.

    ~/miniconda3/envs/go2/bin/python teaser_render.py --preview N OUT.jpg     # contact sheet of top N
    ~/miniconda3/envs/lgd/bin/python ... (final figure uses matplotlib + LaTeX; see --final)
    ~/miniconda3/envs/go2/bin/python teaser_render.py --final TAKE FRAME OUT_DIR

Faces: every person found by the strip pose detection gets the head region (pose landmarks 0-10,
enlarged) heavily blurred; every logged person box also gets its top part blurred, so a face the pose
model missed is still covered. Boxes: blue = accepted by the IT2 + gaze gate (robot-directed, correct),
pink = rejected (bystander, correct; gesture-only would have obeyed it).
"""
import json
import os
import sys

import cv2
import numpy as np

from engage.takeio import proc_size
from full_cues import detect_people, models

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TAKES = os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3")
CAND = os.path.join(HERE, "results", "full", "teaser_candidates.json")
BLUE, PINK = (255, 145, 35), (99, 0, 247)        # BGR of #2391FF, #F70063


def frame_and_people(take, f):
    meta = json.load(open(os.path.join(TAKES, take, "meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(TAKES, take, "log.jsonl"))]
    cap = cv2.VideoCapture(os.path.join(TAKES, take, "raw.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, raw = cap.read()
    assert ok, (take, f)
    H, W = raw.shape[:2]
    s = W / proc_size(W, H)[0]
    mir = meta.get("mirror", True)
    boxes = {}
    for p in recs[f]["people"]:
        x0, y0, x1, y1 = (v * s for v in p["bbox"])
        if mir:
            x0, x1 = W - x1, W - x0
        boxes[p["tid"]] = [x0, y0, x1, y1]
    return raw, boxes, mir


def blur_faces(img, boxes, mir, pose):
    H, W = img.shape[:2]
    full = cv2.flip(img, 1) if mir else img
    regions = []
    for _, _, lm, _ in detect_people(pose, full):
        xs = [lm[i].x * W for i in range(11)]
        ys = [lm[i].y * H for i in range(11)]
        cx, cy = np.mean(xs), np.mean(ys)
        r = max(max(xs) - min(xs), max(ys) - min(ys), 20) * 1.1
        if mir:
            cx = W - cx
        regions.append((cx - r, cy - 1.2 * r, cx + r, cy + 1.0 * r))
    for x0, y0, x1, y1 in boxes.values():
        w, h = x1 - x0, y1 - y0
        regions.append((x0 + 0.1 * w, y0 - 0.02 * h, x1 - 0.1 * w, y0 + 0.17 * h))
    out = img.copy()
    for x0, y0, x1, y1 in regions:
        x0, y0, x1, y1 = (int(max(0, v)) for v in (x0, y0, min(x1, W), min(y1, H)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_AREA)
        roi = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST), (0, 0), 6)
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
        out[y0:y1, x0:x1] = (mask * roi + (1 - mask) * out[y0:y1, x0:x1]).astype(np.uint8)
    return out


def render(c, pose, draw=True):
    img, boxes, mir = frame_and_people(c["take"], c["frame"])
    img = blur_faces(img, boxes, mir, pose)
    if draw:
        for group, col in (("accept", BLUE), ("reject", PINK)):
            for a in c[group]:
                b = boxes.get(a["tid"])
                if b:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 4)
                    cv2.putText(img, f"{a['gesture']} E={a['E_low']:.2f}", (int(b[0]), int(b[1]) + 24), 0, 0.8, col, 2)
    return img, boxes


def main():
    pose, _ = models()
    if sys.argv[1] == "--preview":
        n, out = int(sys.argv[2]), sys.argv[3]
        cands = json.load(open(CAND))[:n]
        tiles = []
        for k, c in enumerate(cands):
            img, _ = render(c, pose)
            img = cv2.resize(img, (640, 360))
            cv2.putText(img, f"#{k} {c['take']} f{c['frame']}", (8, 26), 0, 0.8, (255, 255, 255), 2)
            tiles.append(img)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        cv2.imwrite(out, np.vstack(rows))
    else:
        take, f, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        c = next(c for c in json.load(open(CAND)) if c["take"] == take and c["frame"] == f)
        img, boxes = render(c, pose, draw=False)
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, "teaser_blurred.png"), img)
        json.dump({"take": take, "frame": f, "accept": c["accept"], "reject": c["reject"],
                   "boxes": {str(k): v for k, v in boxes.items()}, "size": img.shape[:2]},
                  open(os.path.join(out, "teaser_meta.json"), "w"), indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
