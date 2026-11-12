#!/usr/bin/env python3
"""Manual privacy fixes for the supplementary frames, found by zooming into every torso and background:
one background face the detectors missed (ex3) and shirt logos / brand marks (double-blind).
Spots are given relative to a logged person box (tile coordinates of a 220 px wide torso crop that starts
at 10 % of the box height), so they follow the box. Run after supp_examples.py and supp_faces.py.
    ~/miniconda3/envs/go2/bin/python supp_privacy_fix.py
"""
import json
import os

import cv2

from supp_faces import blur_ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_src")
LOGOS = {1: [("2", 55, 80)], 2: [("40", 198, 75), ("28", 158, 63)], 3: [("27", 150, 60)], 4: [("3", 135, 50)],
         5: [("20", 175, 50)]}
FACES = {3: [(496, 197, 30)]}            # full-frame x, y, radius
ABS = {2: [(647, 183, 18)], 3: [(618, 199, 20)], 4: [(753, 220, 20)]}   # logos measured on the zoom check
ex = json.load(open(os.path.join(SRC, "examples.json")))
for k, m in enumerate(ex):
    f = os.path.join(SRC, f"ex{k}.png")
    img = cv2.imread(f)
    for tid, tx, ty in LOGOS.get(k, []):
        x0, y0, x1, y1 = m["boxes"][tid]
        sc = 220 / (int(x1) - int(x0))
        cx, cy = int(x0) + tx / sc, int(y0 + 0.1 * (y1 - y0)) + (ty - 20) / sc
        r = max(14, 26 / sc)
        blur_ellipse(img, cx, cy, r, r)
        print(f"ex{k} logo of person {tid} at ({cx:.0f}, {cy:.0f}) r={r:.0f}")
    for cx, cy, r in FACES.get(k, []):
        blur_ellipse(img, cx, cy, r, 1.2 * r)
        print(f"ex{k} face at ({cx}, {cy})")
    for cx, cy, r in ABS.get(k, []):
        blur_ellipse(img, cx, cy, r, r)
        print(f"ex{k} logo at ({cx}, {cy})")
    cv2.imwrite(f, img)
