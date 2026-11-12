#!/usr/bin/env python3
"""Find frames for the paper's teaser figure: at the same moment, a robot-directed gesture that the
IT2 + gaze gate accepts and a bystander gesture that it rejects although gesture-only would obey it.
Both decisions must be correct (labels r / n). Writes results/full/teaser_candidates.json.

    ~/miniconda3/envs/go2/bin/python teaser_candidates.py
"""
import json
import os

import numpy as np

from full_eval import OFF, P, RULES4, Data, OUT

d = Data()
E = d.engage(True, rules=RULES4)
u = E + OFF * d.stop
acc_it2g = d.gate_value(E) >= P.theta_e_motion
acc_go = d.best_window(np.zeros(d.n)) >= 0
labels = {}
by_frame = {}
for j, e in enumerate(d.eps):
    first = int(next(k for k, i in enumerate(e["idx"]) if True) if e["idx"] else 0)
    for k, i in enumerate(e["idx"]):
        if i is None or not d.cand[i]:
            continue
        by_frame.setdefault((e["take"], e.get("first", 0) + k), [])
cands = []
for j, e in enumerate(d.eps):
    e["j"] = j
for take in d.takes:
    eps = [e for e in d.eps if e["take"] == take]
    # frame number of each idx position = first_frame + position (Data walks first..last)
    import csv
    rows = {(r["take"], r["clip"]): r for r in csv.DictReader(open(os.path.join(
        os.path.dirname(OUT), "..", "..", "..", "data", "depthpack_2026-11-11", "labels_pack_2026-11-10",
        "labels_final.csv"))) if r["take"] == take}
    frames = {}
    for e in eps:
        r = rows[(take, str(e["clip"]))]
        f0 = int(r["first_frame"])
        for k, i in enumerate(e["idx"]):
            if i is not None and d.cand[i]:
                frames.setdefault(f0 + k, []).append((e, i, int(r["track_id"]), r["gesture"]))
    for f, lst in frames.items():
        good_r = [(e, i, t, g) for e, i, t, g in lst if e["label"] == "r" and acc_it2g[e["j"]] and u[i] >= P.theta_e_motion]
        good_n = [(e, i, t, g) for e, i, t, g in lst if e["label"] == "n" and acc_go[e["j"]] and not acc_it2g[e["j"]]]
        if good_r and good_n:
            gap = max(u[i] for _, i, _, _ in good_r) - min(u[i] for _, i, _, _ in good_n)
            cands.append({"take": take, "frame": f, "gap": float(gap),
                          "accept": [{"tid": t, "gesture": g, "E_low": float(E[i])} for _, i, t, g in good_r],
                          "reject": [{"tid": t, "gesture": g, "E_low": float(E[i])} for _, i, t, g in good_n]})
cands.sort(key=lambda c: (-(len(c["accept"]) + len(c["reject"])), -c["gap"]))
# keep at most 3 per take, spread over time
out, per = [], {}
for c in cands:
    if per.get(c["take"], 0) >= 3 or any(o["take"] == c["take"] and abs(o["frame"] - c["frame"]) < 60 for o in out):
        continue
    per[c["take"]] = per.get(c["take"], 0) + 1
    out.append(c)
json.dump(out, open(os.path.join(OUT, "teaser_candidates.json"), "w"), indent=1)
print(len(cands), "frames qualify;", len(out), "kept")
for c in out[:15]:
    print(c["take"], c["frame"], "accept", [(a["gesture"], round(a["E_low"], 2)) for a in c["accept"]],
          "reject", [(a["gesture"], round(a["E_low"], 2)) for a in c["reject"]])
