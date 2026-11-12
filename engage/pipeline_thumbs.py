#!/usr/bin/env python3
"""Small real images for the overview figure (Fig. 3), all from the teaser frame (D4_S3_r5, frame 437), so no new
participant appears. Faces blurred as in the teaser (teaser_render.blur_faces + the same feathered shirt-logo blur).

    ~/miniconda3/envs/go2/bin/python pipeline_thumbs.py

Writes cvpr2027_paper/figures/pipeline_src/{frame,hand,head}.png:
  frame : the camera frame with every detected person's box (grey) and the commander's (blue)
  hand  : the commander's square upper-body crop with the 21 hand keypoints (HandLandmarker on the unblurred crop,
          drawn on the blurred one)
  head  : the commander's head crop with eye corners and iris centres (FaceLandmarker + iris on the unblurred crop,
          drawn on the blurred one)
"""
import json
import os

import cv2
import mediapipe as mp
import numpy as np

from find_gestures_centered import centred_crop
from find_gestures_crop import hands_model
from full_cues import EYES, FACE_SIDE, L_EAR, L_SH, NOSE, R_EAR, R_SH, detect_people, iou2, models
from teaser_render import blur_faces, frame_and_people

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures")
OUT = os.path.join(FIG, "pipeline_src")
TAKE, FRAME, CMD = "D4_S3_r5", 437, 3
BLUE, GREY, PINK = (255, 145, 35), (150, 150, 150), (99, 0, 247)      # BGR
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

os.makedirs(OUT, exist_ok=True)
pose, face = models()
raw, boxes, mir = frame_and_people(TAKE, FRAME)
H, W = raw.shape[:2]
blur = blur_faces(raw, boxes, mir, pose)
yy, xx = np.mgrid[0:H, 0:W]
for cx, cy, rx, ry in ((853, 168, 26, 20),):          # exactly teaser_figure.py's feathered shirt-logo blur
    m = np.clip(1.6 - np.hypot((xx - cx) / rx, (yy - cy) / ry), 0, 1)[..., None]
    bl = cv2.GaussianBlur(blur.astype(np.float32), (0, 0), 7)
    blur = (m * bl + (1 - m) * blur).astype(np.uint8)

# frame with boxes
fr = blur.copy()
for tid, b in boxes.items():
    col, th = (BLUE, 5) if tid == CMD else (GREY, 3)
    cv2.rectangle(fr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)
x0 = int(min(b[0] for b in boxes.values()) - 40)
x1 = int(max(b[2] for b in boxes.values()) + 40)
cw = x1 - x0
ch = int(cw * 9 / 16)
y0 = max(0, int(min(b[1] for b in boxes.values()) - 30))
y0 = min(y0, H - ch)
cv2.imwrite(os.path.join(OUT, "frame.png"), cv2.resize(fr[y0:y0 + ch, x0:x1], (640, 360), interpolation=cv2.INTER_AREA))

# commander: hand crop with keypoints
b = boxes[CMD]
crop_raw, sx, sy, sc = centred_crop(raw, b)
crop_blr, _, _, _ = centred_crop(blur, b)
res = hands_model().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop_raw, cv2.COLOR_BGR2RGB)))
S = crop_raw.shape[0]
hand = crop_blr.copy()
for hl in res.hand_landmarks or []:
    pts = [(int(p.x * S), int(p.y * S)) for p in hl]
    for a, c in HAND_EDGES:
        cv2.line(hand, pts[a], pts[c], BLUE, 4, cv2.LINE_AA)
    for p in pts:
        cv2.circle(hand, p, 7, PINK, -1, cv2.LINE_AA)
print("hands found:", len(res.hand_landmarks or []))
cv2.imwrite(os.path.join(OUT, "hand.png"), cv2.resize(hand, (320, 320), interpolation=cv2.INTER_AREA))

# commander: head crop with eyes / irises (pose landmarks from the same strip detection, unmirrored frame)
cands = detect_people(pose, raw)
lb = b
best = max(cands, key=lambda q: iou2(q[0], lb)[0])
lm = best[2]
nx, ny = lm[NOSE].x * W, lm[NOSE].y * H
ear = np.hypot((lm[L_EAR].x - lm[R_EAR].x) * W, (lm[L_EAR].y - lm[R_EAR].y) * H)
shw = np.hypot((lm[L_SH].x - lm[R_SH].x) * W, (lm[L_SH].y - lm[R_SH].y) * H)
side = int(max(FACE_SIDE * max(ear, 0.5 * shw), 48))
hx, hy = int(nx - side / 2), int(ny - side / 2)
hr = cv2.resize(raw[hy:hy + side, hx:hx + side], (256, 256), interpolation=cv2.INTER_CUBIC)
hb = cv2.resize(blur[hy:hy + side, hx:hx + side], (256, 256), interpolation=cv2.INTER_CUBIC)
r = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)))
print("face found:", bool(r.face_landmarks))
if r.face_landmarks:
    f = r.face_landmarks[0]
    for a, c, iris in EYES:
        pa, pc, pi = [(int(f[k].x * 256), int(f[k].y * 256)) for k in (a, c, iris)]
        cv2.line(hb, pa, pc, BLUE, 2, cv2.LINE_AA)
        cv2.circle(hb, pi, 5, PINK, -1, cv2.LINE_AA)
cv2.imwrite(os.path.join(OUT, "head.png"), hb)
json.dump({"take": TAKE, "frame": FRAME, "commander_tid": CMD, "frame_crop": [x0, y0, x1, y0 + ch], "head_crop": [hx, hy, side]},
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("wrote", OUT)
