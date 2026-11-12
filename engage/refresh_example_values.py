#!/usr/bin/env python3
"""Refresh the engagement values shown in the teaser and in the supplementary examples after the evaluation
inputs change, WITHOUT re-rendering the images (their face/logo blurring was checked by hand).

    ~/miniconda3/envs/go2/bin/python refresh_example_values.py

Same frames, same people; recomputes E_low (IT2 + gaze, frame value) and the episode decision with full_eval.Data
(current DIST_SRC). Rewrites the values in cvpr2027_paper/figures/teaser_src/teaser_meta.json and
figures/supp_src/examples.json and prints every change, so a changed decision is noticed.
"""
import csv
import json
import os

from full_eval import LABELS, OFF, P, RULES4, Data

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures")
d = Data()
E = d.engage(True, rules=RULES4)
acc = d.gate_value(E) >= P.theta_e_motion
rows = {(r["take"], r["clip"]): r for r in csv.DictReader(open(LABELS))}


def lookup(take, f, tid):
    """(E_low at frame f, episode executed) for the labelled episode of person tid active at frame f."""
    for j, e in enumerate(d.eps):
        if e["take"] != take:
            continue
        r = rows[(take, str(e["clip"]))]
        if int(r["track_id"]) != tid:
            continue
        k = f - int(r["first_frame"])
        if 0 <= k < len(e["idx"]) and e["idx"][k] is not None:
            return float(E[e["idx"][k]]), bool(acc[j])
    return None, None


tp = os.path.join(FIG, "teaser_src", "teaser_meta.json")
t = json.load(open(tp))
for grp in ("accept", "reject"):
    for a in t[grp]:
        e, ex = lookup(t["take"], t["frame"], a["tid"])
        print(f"teaser {grp} tid {a['tid']}: E_low {a['E_low']:.3f} -> {e:.3f}, executed {ex}")
        assert ex == (grp == "accept"), "teaser decision changed"
        a["E_low"] = e
json.dump(t, open(tp, "w"), indent=1)

sp = os.path.join(FIG, "supp_src", "examples.json")
ex = json.load(open(sp))
for m in ex:
    for p in m["people"]:
        e, x = lookup(m["take"], m["frame"], p["tid"])
        flag = "" if x == p["executed"] else "   <-- DECISION CHANGED"
        print(f"supp {m['take']} f{m['frame']} tid {p['tid']} ({p['label']}): E_low {p['E_low']:.3f} -> {e:.3f}, "
              f"executed {p['executed']} -> {x}{flag}")
        p["E_low"], p["executed"] = e, x
json.dump(ex, open(sp, "w"), indent=1)
