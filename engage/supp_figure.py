#!/usr/bin/env python3
"""Supplementary qualitative figure: 2 x 3 grid (correct decisions, then failures), faces blurred.

    ~/miniconda3/envs/lgd/bin/python supp_figure.py
Reads cvpr2027_paper/figures/supp_src/{ex*.png, examples.json}; writes figures/supp_examples.pdf.
Box colour = decision of the IT2 + gaze gate (blue executed, pink ignored); label = correct / wrong.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_src")
OUT = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_examples.pdf")
BLUE, PINK = "#2391FF", "#F70063"
CMD = {"palm": "STOP", "stop": "STOP", "fist": "FWD", "stop_inverted": "FWD", "call": "BACK", "mute": "BACK",
       "one": "LEFT", "two_up": "RIGHT"}
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
                     "font.family": "serif", "font.size": 8})
ex = json.load(open(os.path.join(SRC, "examples.json")))
fig, axs = plt.subplots(2, 3, figsize=(6.875, 2.95), gridspec_kw={"wspace": 0.03, "hspace": 0.14})
for k, (ax, m) in enumerate(zip(axs.flat, ex)):
    img = mpimg.imread(os.path.join(SRC, f"ex{k}.png"))[..., :3]
    H, W = img.shape[:2]
    x0, y0, x1, y1 = 0, 0, W, H
    ax.imshow(img, extent=(0, W, H, 0))
    ax.set_axis_off()
    for p in m["people"]:
        b = m["boxes"].get(str(p["tid"]))
        if not b:
            continue
        col = BLUE if p["executed"] else PINK
        ok = p["executed"] == (p["label"] == "r")
        ax.add_patch(Rectangle((b[0] - x0, b[1] - y0), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=1.1))
        wrong = r" $\boldsymbol{\times}$" if not ok else ""
        ax.text(b[0] - x0 + 3, b[1] - y0 + 4, rf"${p['label']}$ {p['E_low']:.2f}{wrong}",
                color="white", fontsize=6.2, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none", alpha=0.95))
    tag = "abcdef"[k]
    ax.set_title(rf"\textbf{{({tag})}} {m['kind'].replace(' (S6)', '')}, {m['take'].split('_')[1]}", fontsize=8, loc="left", pad=2)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
