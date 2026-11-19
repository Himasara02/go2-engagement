#!/usr/bin/env python3
"""Numbers that explain the main results (standing recordings, paper setting: annotator A, track distance).

    ~/miniconda3/envs/go2/bin/python full_result_explanations.py   -> results/full/explanations.{json,md}

Uses exactly the evaluation of full_eval.py (same episodes, cues, IT2 + gaze gate, frozen thresholds, 3-frame sustain).
  1. Perception ceiling: why gesture only (the upper bound of every policy) obeys 70.6 % of commands / 67.6 % of
     bystander gestures: episodes without 3 consecutive confident command frames, split by cause.
  2. How well the cues separate commanders and bystanders: per-episode median facing, gaze, distance (r vs n).
  3. Residual false triggers of IT2 + gaze: cues of the bystander episodes it accepts, by scenario.
  4. Missed commands of IT2 + gaze (accepted by gesture only, rejected by the gate): cues and uncertainty.
"""
import json
import os
from collections import Counter

import numpy as np

from full_eval import OUT, P, RULES4, Data

d = Data()
E = d.engage(True, rules=RULES4)
go = d.best_window(np.zeros(d.n)) >= 0
ours = d.gate_value(E) >= P.theta_e_motion
lab = np.array([e["label"] for e in d.eps])
scen = np.array([e["scenario"] for e in d.eps])
x = d.x


def ep_median(e, key):
    v = [x[key][i] for i in e["idx"] if i is not None]
    return float(np.median(v)) if v else float("nan")


def cause(e):
    """Why an episode has no 3 consecutive confident command frames."""
    idx = e["idx"]
    run = best = 0
    for i in idx:                                # longest run of frames with person + cues + depth
        run = run + 1 if i is not None else 0
        best = max(best, run)
    if best < 3:
        return "person, cues or depth missing"
    return "classifier: command class with confidence >= 0.45 not held for 3 frames"


def stats(mask, keys=("facing", "gaze", "distance", "facing_sigma", "distance_sigma")):
    eps = [e for e, m in zip(d.eps, mask) if m]
    out = {"n": len(eps)}
    for k in keys:
        v = np.array([ep_median(e, k) for e in eps])
        v = v[~np.isnan(v)]
        out[k] = {"median": round(float(np.median(v)), 3), "q25": round(float(np.percentile(v, 25)), 3),
                  "q75": round(float(np.percentile(v, 75)), 3)} if len(v) else None
    return out


res = {}
for L in ("r", "n"):
    m = lab == L
    miss = [cause(e) for e, mm, g in zip(d.eps, m, go) if mm and not g]
    res[f"ceiling_{L}"] = {"episodes": int(m.sum()), "gesture_only_accepts": int((go & m).sum()),
                           "rate": round(float((go & m).sum() / m.sum()), 4),
                           "not_sustained_by_cause": dict(Counter(miss))}
r, n = lab == "r", lab == "n"
res["separation"] = {"r": stats(r), "n": stats(n)}
res["it2_gaze_share_of_gesture_only"] = {
    "commands_kept_of_those_gesture_only_obeys": round(float((ours & r).sum() / (go & r).sum()), 4),
    "bystander_gestures_passed_of_those_gesture_only_obeys": round(float((ours & n).sum() / (go & n).sum()), 4)}
ft = ours & n
res["false_triggers"] = {"cues": stats(ft), "by_scenario": {s: {"accepted": int((ft & (scen == s)).sum()),
                         "bystander_episodes": int((n & (scen == s)).sum())} for s in sorted(set(scen))},
                         "facing_at_least_0.76": round(float(np.mean([ep_median(e, "facing") >= 0.76
                                                                      for e, m in zip(d.eps, ft) if m])), 3)}
missed = go & r & ~ours
res["missed_commands"] = {"cues": stats(missed), "kept_for_comparison": stats(ours & r)}

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "explanations.json"), "w"), indent=1)
with open(os.path.join(OUT, "explanations.md"), "w") as f:
    f.write("# Numbers that explain the main results (standing, IT2 + gaze, paper setting)\n\n```json\n")
    f.write(json.dumps(res, indent=1) + "\n```\n")
print(json.dumps(res, indent=1))
