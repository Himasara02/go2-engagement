#!/usr/bin/env python3
"""Short face-blurred video clips for the supplementary video (Clip A: D4_S3_r5, the teaser recording).

    ~/miniconda3/envs/go2/bin/python video_clip.py --sheet TAKE F0 F1 STEP OUT.jpg   # blurred contact sheet
    ~/miniconda3/envs/go2/bin/python video_clip.py --render TAKE F0 F1 OUT.mp4       # blurred, levelled clip

Privacy, same as the teaser (teaser_render.py / teaser_figure.py), made stronger for video:
  * head region of every person found by the strip pose detection, plus the top of every logged box;
  * union of those regions over +-3 frames, so a single missed detection cannot show a face;
  * shirt logo of the right-hand person (tid 1 in D4_S3_r5) blurred relative to their box.
Then: camera roll levelled by +6.3 deg (as in the teaser) and a 16:9 crop without black corners
that leaves out the fan at the right edge. Frame times come from log.jsonl (real speed).
Raw frames never leave this script unblurred.
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from teaser_render import TAKES, frame_and_people
from full_cues import detect_people, models

ROT = 6.3
OUT_W, OUT_H = 1280, 720
LOGO = {"D4_S3_r5": [(1, 0.652, 0.172, 0.20, 0.07)],    # tid, centre (rel. to box), half size (rel. to box)
        "D4_S3_r3": [(2, 0.761, 0.179, 0.22, 0.07)]}
FAN = {"D4_S3_r3": 1100}                                 # raw x of the fan's left edge if not the default 1150
HOLD = 3


def head_regions(img, boxes, mir, pose):
    H, W = img.shape[:2]
    full = cv2.flip(img, 1) if mir else img
    reg = []
    for _, _, lm, _ in detect_people(pose, full):
        xs = [lm[i].x * W for i in range(11)]
        ys = [lm[i].y * H for i in range(11)]
        cx, cy = np.mean(xs), np.mean(ys)
        r = max(max(xs) - min(xs), max(ys) - min(ys), 20) * 1.25
        if mir:
            cx = W - cx
        reg.append((cx - r, cy - 1.3 * r, cx + r, cy + 1.1 * r))
    for x0, y0, x1, y1 in boxes.values():
        w, h = x1 - x0, y1 - y0
        reg.append((x0 + 0.05 * w, y0 - 0.04 * h, x1 - 0.05 * w, y0 + 0.19 * h))
    return reg


def logo_regions(take, boxes):
    reg = []
    for tid, rx, ry, hw, hh in LOGO.get(take, []):
        b = boxes.get(tid)
        if b:
            w, h = b[2] - b[0], b[3] - b[1]
            cx, cy = b[0] + rx * w, b[1] + ry * h
            reg.append((cx - hw * w, cy - hh * h, cx + hw * w, cy + hh * h))
    return reg


def blur(img, regions):
    H, W = img.shape[:2]
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


def level_and_crop(img, fan_x=1150):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ROT, 1.0)
    rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    fan = (M @ np.array([fan_x, H / 2, 1.0]))[0]          # fan's left edge after rotation
    best = None
    for half_w in np.arange(W / 2, 200, -2):
        half_h = half_w * 9 / 16
        for cx in np.arange(fan - half_w, half_w - 1, -4):   # as far right as allowed, then left
            x0, x1 = int(cx - half_w), int(cx + half_w)
            y0, y1 = int(H / 2 - half_h), int(H / 2 + half_h)
            if x0 >= 0 and y0 >= 0 and x1 <= min(W, fan) and y1 <= H and valid[y0:y1, x0:x1].all():
                best = (x0, y0, x1, y1)
                break
        if best:
            break
    x0, y0, x1, y1 = best
    return cv2.resize(rot[y0:y1, x0:x1], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA
                      if x1 - x0 > OUT_W else cv2.INTER_CUBIC), best


def frames(take, f0, f1, pose):
    reg = {}
    for f in range(max(0, f0 - HOLD), f1 + HOLD + 1):
        try:
            img, boxes, mir = frame_and_people(take, f)
        except (AssertionError, IndexError):
            continue
        reg[f] = head_regions(img, boxes, mir, pose) + logo_regions(take, boxes)
    for f in range(f0, f1 + 1):
        img, boxes, mir = frame_and_people(take, f)
        rs = [r for g in range(f - HOLD, f + HOLD + 1) for r in reg.get(g, [])]
        out, crop = level_and_crop(blur(img, rs), FAN.get(take, 1150))
        yield f, out, crop, boxes


def times(take):
    return [json.loads(l)["t"] for l in open(os.path.join(TAKES, take, "log.jsonl"))]


def main():
    mode, take, f0, f1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pose, _ = models()
    if mode == "--sheet":
        step, out = int(sys.argv[5]), sys.argv[6]
        t = times(take)
        tiles = []
        for f, img, crop, _ in frames(take, f0, f1, pose):
            if (f - f0) % step == 0:
                img = cv2.resize(img, (480, 270))
                cv2.putText(img, f"f{f} t={t[f]:.1f}", (6, 22), 0, 0.6, (255, 255, 255), 2)
                tiles.append(img)
        print("crop", crop)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    else:
        out = sys.argv[5]
        t = times(take)
        with tempfile.TemporaryDirectory() as tmp:
            lst = []
            for f, img, crop, _ in frames(take, f0, f1, pose):
                p = os.path.join(tmp, f"{f:05d}.png")
                cv2.imwrite(p, img)
                dur = (t[f + 1] - t[f]) if f < f1 else (t[f] - t[f - 1])
                lst.append(f"file '{p}'\nduration {dur:.4f}\n")
            lst.append(f"file '{p}'\n")
            with open(os.path.join(tmp, "list.txt"), "w") as fh:
                fh.writelines(lst)
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                            os.path.join(tmp, "list.txt"), "-vf", "fps=30,format=yuv420p", "-c:v", "libx264",
                            "-crf", "18", "-preset", "slow", "-movflags", "+faststart", "-an", out], check=True)
        print("crop", crop, "wrote", out, f"{t[f1] - t[f0]:.1f} s")


if __name__ == "__main__":
    main()
