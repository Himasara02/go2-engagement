#!/usr/bin/env python3
"""Teaser figure (page 1) from the face-blurred frame made by teaser_render.py --final.

    ~/miniconda3/envs/lgd/bin/python teaser_figure.py

Reads cvpr2027_paper/figures/teaser_src/{teaser_blurred.png, teaser_meta.json}; writes
cvpr2027_paper/figures/teaser.pdf at column width. Extra blur: a shirt logo (double-blind).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "teaser_src")
OUT = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "teaser.pdf")
BLUE, PINK, INK = "#2391FF", "#F70063", "#1A1A1A"
CMD = {"palm": "STOP", "stop": "STOP", "fist": "FORWARD", "stop_inverted": "FORWARD", "call": "BACK",
       "mute": "BACK", "one": "LEFT", "two_up": "RIGHT"}
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
                     "font.family": "serif", "savefig.bbox": "tight", "savefig.pad_inches": 0.0})

img = mpimg.imread(os.path.join(SRC, "teaser_blurred.png"))[..., :3].copy()
meta = json.load(open(os.path.join(SRC, "teaser_meta.json")))
# extra privacy blur (feathered): the shirt logo of the right-hand person (double-blind)
yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
for cx, cy, rx, ry in ((853, 168, 26, 20),):
    m = np.clip(1.6 - np.hypot((xx - cx) / rx, (yy - cy) / ry), 0, 1)[..., None]
    bl = np.stack([gaussian_filter(img[..., c], 7) for c in range(3)], -1)
    img = m * bl + (1 - m) * img
# level the camera roll: vertical room edges (Hough lines, fitted against x to separate perspective)
# come out vertical at a rotation of +6.3 deg (residual tilt crosses zero between 6.0 and 6.5 deg)
import cv2
ROT = 6.3
H0, W0 = img.shape[:2]
M = cv2.getRotationMatrix2D((W0 / 2, H0 / 2), ROT, 1.0)
img = cv2.warpAffine(img.astype(np.float32), M, (W0, H0), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)).clip(0, 1)
valid = cv2.warpAffine(np.ones((H0, W0), np.float32), M, (W0, H0), flags=cv2.INTER_NEAREST) > 0.5
for k, (x0, y0, x1, y1) in list(meta["boxes"].items()):        # rotate box centres, keep box size
    cx, cy = M @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, 1.0])
    w, h = x1 - x0, y1 - y0
    meta["boxes"][k] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
# largest crop (aspect 900:575, centred on the people) without empty corners
bx = np.array(list(meta["boxes"].values()))
ccx, ccy = (bx[:, 0].min() + bx[:, 2].max()) / 2, (bx[:, 1].min() + bx[:, 3].max()) / 2 + 20
for half_w in np.arange(470, 200, -2):
    half_h = half_w * 575 / 900
    CX0, CX1, CY0, CY1 = (int(round(v)) for v in (ccx - half_w, ccx + half_w, ccy - half_h, ccy + half_h))
    if CX0 >= 0 and CY0 >= 0 and CX1 <= W0 and CY1 <= H0 and valid[CY0:CY1, CX0:CX1].all():
        break
crop = img[CY0:CY1, CX0:CX1]
print("crop", CX0, CX1, CY0, CY1)
H, W = crop.shape[:2]
fig = plt.figure(figsize=(3.25, 3.25 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(crop, extent=(0, W, H, 0))
ax.set_axis_off()


def box(tid, col, title, sub):
    x0, y0, x1, y1 = meta["boxes"][str(tid)]
    x0, x1, y0, y1 = x0 - CX0, x1 - CX0, y0 - CY0, y1 - CY0
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=1.4))
    yb = y1 - 6
    ax.add_patch(FancyBboxPatch((x0 + 4, yb - 52), min(x1 - x0 - 8, 190), 48, boxstyle="round,pad=2",
                                fc=col, ec="none", alpha=0.92))
    ax.text(x0 + 10, yb - 38, title, color="white", fontsize=6.3, va="center", ha="left")
    ax.text(x0 + 10, yb - 16, sub, color="white", fontsize=5.6, va="center", ha="left")


for a in meta["accept"]:
    box(a["tid"], BLUE, rf"\textbf{{{CMD[a['gesture']]}}}: for the robot",
        rf"$\underline{{E}}={a['E_low']:.2f}$ $\rightarrow$ executed")
for a in meta["reject"]:
    box(a["tid"], PINK, rf"\textbf{{{CMD[a['gesture']]}}}: to a friend",
        rf"$\underline{{E}}={a['E_low']:.2f}$ $\rightarrow$ ignored")
fig.savefig(OUT)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=220)
print("wrote", OUT)
