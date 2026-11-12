#!/usr/bin/env python3
"""Pick qualitative examples for the supplementary material (IT2 + gaze gate) and render them with faces
blurred (teaser_render.blur_faces).

  correct  : a robot-directed gesture executed and a bystander gesture ignored at the same moment
             (from results/full/teaser_candidates.json), one each from S2, S4 and S6 if available
  failures : (a) bystander episode executed (false trigger) in S6, (b) the most confident false trigger
             elsewhere, (c) a robot-directed episode the gate ignored (missed command)

    ~/miniconda3/envs/go2/bin/python supp_examples.py
Writes cvpr2027_paper/figures/supp_src/ex<k>.png + examples.json.
"""
import csv
import json
import os

import cv2
import numpy as np

from full_eval import LABELS, OFF, OUT, P, RULES4, Data
from teaser_render import blur_faces, frame_and_people
from full_cues import models

HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_src")
d = Data()
E = d.engage(True, rules=RULES4)
u = E + OFF * d.stop
acc = d.gate_value(E) >= P.theta_e_motion
rows = {(r["take"], r["clip"]): r for r in csv.DictReader(open(LABELS))}


def frames_of(e):
    r = rows[(e["take"], str(e["clip"]))]
    f0 = int(r["first_frame"])
    return [(f0 + k, i) for k, i in enumerate(e["idx"]) if i is not None], int(r["track_id"]), r["gesture"]


def people_at(take, f):
    """Every labelled episode of this take active at frame f: (tid, label, gesture, E_low, accepted)."""
    out = []
    for j, e in enumerate(d.eps):
        if e["take"] != take:
            continue
        fr, tid, g = frames_of(e)
        hit = [i for ff, i in fr if ff == f]
        if hit:
            out.append({"tid": tid, "label": e["label"], "gesture": g, "E_low": float(E[hit[0]]),
                        "executed": bool(acc[j])})
    return out


picked = []
tc = json.load(open(os.path.join(OUT, "teaser_candidates.json")))
for sc in ("S2", "S4", "S6"):
    c = next((c for c in tc if c["take"].split("_")[1] == sc and c["take"] != "D4_S3_r5"), None)
    if c:
        picked.append({"kind": "correct", "take": c["take"], "frame": c["frame"]})
# failures
ft = [(j, e) for j, e in enumerate(d.eps) if e["label"] == "n" and acc[j]]
miss = [(j, e) for j, e in enumerate(d.eps) if e["label"] == "r" and not acc[j]]


def peak(e):
    fr, _, _ = frames_of(e)
    return max(fr, key=lambda t: u[t[1]] if d.cand[t[1]] else -1)


s6 = [x for x in ft if x[1]["scenario"] == "S6"]
for group, kind in ((s6, "false trigger (S6)"), ([x for x in ft if x[1]["scenario"] != "S6"], "false trigger")):
    if group:
        j, e = max(group, key=lambda x: u[peak(x[1])[1]])
        picked.append({"kind": kind, "take": e["take"], "frame": peak(e)[0]})
if miss:
    # a missed command whose face was not found (sigma_g high) - the typical reason
    j, e = max(miss, key=lambda x: np.mean([d.x["gaze_sigma"][i] for i in x[1]["idx"] if i is not None]))
    fr, _, _ = frames_of(e)
    picked.append({"kind": "missed command", "take": e["take"], "frame": fr[len(fr) // 2][0]})

pose, _ = models()
os.makedirs(DST, exist_ok=True)
meta = []
for k, p in enumerate(picked):
    img, boxes, mir = frame_and_people(p["take"], p["frame"])
    img = blur_faces(img, boxes, mir, pose)
    cv2.imwrite(os.path.join(DST, f"ex{k}.png"), img)
    meta.append({**p, "people": people_at(p["take"], p["frame"]), "boxes": {str(t): b for t, b in boxes.items()}})
json.dump(meta, open(os.path.join(DST, "examples.json"), "w"), indent=1)
for m in meta:
    print(m["kind"], m["take"], m["frame"], [(q["label"], q["gesture"], round(q["E_low"], 2), q["executed"]) for q in m["people"]])
