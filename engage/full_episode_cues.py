#!/usr/bin/env python3
"""Episode-mean cues (depth distance, facing, gaze and their sigma) per labelled episode, for the histogram
figure. Uses full_eval.Data (same frames as the evaluation). Writes results/full/episode_cues.json.

    ~/miniconda3/envs/go2/bin/python full_episode_cues.py
"""
import json
import os

import numpy as np

from full_eval import OUT, Data

d = Data()
rows = []
for e in d.eps:
    idx = [i for i in e["idx"] if i is not None]
    if not idx:
        continue
    rows.append({"take": e["take"], "scenario": e["scenario"], "label": e["label"],
                 **{k: float(np.mean(d.x[k][idx])) for k in ("distance", "distance_sigma", "facing", "facing_sigma",
                                                             "gaze", "gaze_sigma")}})
json.dump(rows, open(os.path.join(OUT, "episode_cues.json"), "w"))
for lab in "rn":
    v = [r for r in rows if r["label"] == lab]
    print(lab, len(v), {k: round(float(np.median([r[k] for r in v])), 3) for k in ("distance", "facing", "gaze")})
