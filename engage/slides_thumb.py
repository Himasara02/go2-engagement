#!/usr/bin/env python3
"""Levelled 'frame' thumbnail for the overview figure in the slides (presentation/figures/pipeline_src/frame.png).

    ~/miniconda3/envs/go2/bin/python slides_thumb.py

Same frame, face blur and shirt-logo blur as pipeline_thumbs.py (D4_S3_r5, frame 437), then the camera roll is
removed (+6.3 deg, as for the teaser and the video clips) and a 16:9 crop around the people without black corners
is taken. The paper's thumbnail (cvpr2027_paper/figures/pipeline_src/frame.png) is unchanged.
"""
import os

import cv2
import numpy as np

from full_cues import models
from teaser_render import blur_faces, frame_and_people

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "presentation", "figures", "pipeline_src", "frame.png")
TAKE, FRAME, CMD, ROT = "D4_S3_r5", 437, 3, 6.3
BLUE, GREY = (255, 145, 35), (150, 150, 150)       # BGR

pose, _ = models()
raw, boxes, mir = frame_and_people(TAKE, FRAME)
H, W = raw.shape[:2]
blur = blur_faces(raw, boxes, mir, pose)
yy, xx = np.mgrid[0:H, 0:W]
for cx, cy, rx, ry in ((853, 168, 26, 20),):          # same feathered shirt-logo blur as the paper thumbnail
    m = np.clip(1.6 - np.hypot((xx - cx) / rx, (yy - cy) / ry), 0, 1)[..., None]
    bl = cv2.GaussianBlur(blur.astype(np.float32), (0, 0), 7)
    blur = (m * bl + (1 - m) * blur).astype(np.uint8)

M = cv2.getRotationMatrix2D((W / 2, H / 2), ROT, 1.0)
rot = cv2.warpAffine(blur, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
valid = cv2.warpAffine(np.full((H, W), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
rb = {}
for tid, b in boxes.items():                           # rotate box centres, keep box size (as the teaser)
    cx, cy = M @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
    w, h = b[2] - b[0], b[3] - b[1]
    rb[tid] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
for tid, b in rb.items():
    col, th = (BLUE, 5) if tid == CMD else (GREY, 3)
    cv2.rectangle(rot, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)

# 16:9 crop centred on the people (people span + 40 px, as the paper thumbnail), no black corners
ccx = (min(b[0] for b in rb.values()) + max(b[2] for b in rb.values())) / 2
ccy = (min(b[1] for b in rb.values()) + max(b[3] for b in rb.values())) / 2
span = max(b[2] for b in rb.values()) - min(b[0] for b in rb.values())
for half_w in np.arange(span / 2 + 40, 150, -2):
    half_h = half_w * 9 / 16
    for dy in np.arange(0, H, 4):                      # shift down if the top would leave the image
        x0, x1 = int(ccx - half_w), int(ccx + half_w)
        y0, y1 = int(ccy - half_h + dy), int(ccy + half_h + dy)
        if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H and valid[y0:y1, x0:x1].all():
            break
    else:
        continue
    break
print("crop", x0, y0, x1, y1)
cv2.imwrite(OUT, cv2.resize(rot[y0:y1, x0:x1], (640, 360), interpolation=cv2.INTER_AREA))
print("wrote", OUT)

# hand and iris close-ups: the paper's thumbnails (keypoints already drawn) rotated by the same roll about their
# centre, then the largest centred square without black corners, resized back to the original size
PAPER = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "pipeline_src")
for name in ("hand", "head"):
    img = cv2.imread(os.path.join(PAPER, name + ".png"))
    S = img.shape[0]
    R = cv2.warpAffine(img, cv2.getRotationMatrix2D((S / 2, S / 2), ROT, 1.0), (S, S), flags=cv2.INTER_CUBIC)
    t = np.radians(ROT)
    c = S / (np.cos(t) + np.sin(t))                  # side of the largest axis-aligned square inside
    a = int(np.ceil((S - c) / 2)) + 1
    cv2.imwrite(os.path.join(os.path.dirname(OUT), name + ".png"),
                cv2.resize(R[a:S - a, a:S - a], (S, S), interpolation=cv2.INTER_CUBIC))
    print("wrote", name, "crop", a, S - a)
