#!/usr/bin/env python3
"""Levelled version of the supplementary examples figure, for the slides (presentation/figures/examples_level.pdf).

    ~/miniconda3/envs/lgd/bin/python slides_examples.py

Same face-blurred frames and boxes as supp_figure.py (cvpr2027_paper/figures/supp_src/, already privacy-fixed),
but each frame is rotated to remove the camera roll and centre-cropped (same aspect, no black corners).
Roll per recording day = median over sampled frames of that day (vertical room edges, Hough fit against x):
day 3 = 3.0 deg, day 4 = 6.0 deg; day 4 uses 6.3 deg, the value fitted for the teaser and the video clips.
The paper and supplementary figures are unchanged.
"""
import json
import math
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_src")
OUT = os.path.join(HERE, "..", "..", "presentation", "figures", "examples_level.pdf")
ROLL = {"D3": 3.0, "D4": 6.3}
BLUE, PINK = "#2391FF", "#F70063"
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
                     "font.family": "serif", "font.size": 8})


def level(img, boxes, deg):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    t = math.radians(abs(deg))
    s = min(W / (W * math.cos(t) + H * math.sin(t)), H / (W * math.sin(t) + H * math.cos(t)))
    cw, ch = W * s, H * s
    x0, y0 = (W - cw) / 2, (H - ch) / 2
    crop = rot[int(math.ceil(y0)):int(y0 + ch), int(math.ceil(x0)):int(x0 + cw)]
    out = {}
    for k, (a, b, c, d) in boxes.items():
        cx, cy = M @ np.array([(a + c) / 2, (b + d) / 2, 1.0])
        w, h = c - a, d - b
        out[k] = [max(0, cx - w / 2 - x0), max(0, cy - h / 2 - y0), min(cw, cx + w / 2 - x0), min(ch, cy + h / 2 - y0)]
    return crop, out


ex = json.load(open(os.path.join(SRC, "examples.json")))
fig, axs = plt.subplots(2, 3, figsize=(6.875, 2.95), gridspec_kw={"wspace": 0.03, "hspace": 0.14})
for k, (ax, m) in enumerate(zip(axs.flat, ex)):
    img = mpimg.imread(os.path.join(SRC, f"ex{k}.png"))[..., :3].astype(np.float32)
    img, boxes = level(img, m["boxes"], ROLL[m["take"].split("_")[0]])
    img = img.clip(0, 1)
    H, W = img.shape[:2]
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_axis_off()
    for p in m["people"]:
        b = boxes.get(str(p["tid"]))
        if not b:
            continue
        col = BLUE if p["executed"] else PINK
        ok = p["executed"] == (p["label"] == "r")
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.1))
        wrong = r" $\boldsymbol{\times}$" if not ok else ""
        ax.text(b[0] + 3, b[1] + 4, rf"${p['label']}$ {p['E_low']:.2f}{wrong}",
                color="white", fontsize=6.2, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none", alpha=0.95))
    tag = "abcdef"[k]
    ax.set_title(rf"\textbf{{({tag})}} {m['kind'].replace(' (S6)', '')}, {m['take'].split('_')[1]}", fontsize=8,
                 loc="left", pad=2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
