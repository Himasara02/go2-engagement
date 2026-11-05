#!/usr/bin/env python3
"""Task 8: distance sources.

A. Depth distance on the calibration takes D1_CAL, D1_CAL_r2, D1_CAL_r3 (one
   person at marks 1/2/3/4 m, each mark: three 5 s orientation phases). Frames
   are assigned to the mark k with |d - k| < 0.5 m (the plateaus are well inside
   this window); only the longest continuous block per mark is kept. Per mark: detection rate, bias (median d - k), spread (sd),
   median reported sigma_d, and coverage = share of frames with |d - k| <= sigma_d
   (and <= 2 sigma_d). Facing per orientation phase: each mark's frames split
   into three equal time thirds.
B. Monocular shoulder-width distance needs the pose landmarks, which are not
   logged; it needs a re-run of pose on raw.mp4 (31 Oct data). Until then:
   sigma-model check on the 14 scenario takes - IT2 with the logged depth
   distance but sigma_d from the monocular model (0.15 + 0.08 d, perception.py)
   and from the depth-hole fallback (0.45), vs the depth sigma_d.
Writes results/distance.{json,md}.
"""
import json
import os

import numpy as np

from analysis import NAMES, rate, save
from task6_analysis import Data

CAL = os.path.join(os.path.dirname(__file__), "..", "handover", "handover_2026-11-03_update", "data",
                   "engage", "takes_v2")
CAL_TAKES = ["D1_CAL", "D1_CAL_r2", "D1_CAL_r3"]
MARKS = [1, 2, 3, 4]


def longest_block(pts, gap=6.0):
    """Keep the longest run of frames without a gap > `gap` s (drops stray frames
    from walking past a mark)."""
    if not pts:
        return pts
    pts = sorted(pts, key=lambda a: a[0])
    blocks, cur = [], [pts[0]]
    for a in pts[1:]:
        if a[0] - cur[-1][0] > gap:
            blocks.append(cur)
            cur = []
        cur.append(a)
    blocks.append(cur)
    return max(blocks, key=len)


def cal():
    out = {}
    for t in CAL_TAKES:
        recs = [json.loads(l) for l in open(os.path.join(CAL, t, "log.jsonl"))]
        out[t] = {}
        for k in MARKS:
            pts = [(r["t"], p) for r in recs for p in r["people"] if abs(p["distance"] - k) < 0.5]
            pts = longest_block(pts)
            if not pts:
                out[t][k] = None
                continue
            ts = np.array([a for a, _ in pts])
            d = np.array([p["distance"] for _, p in pts])
            sd = np.array([p["distance_sigma"] for _, p in pts])
            f = np.array([p["facing"] for _, p in pts])
            t0, t1 = ts.min(), ts.max()
            frames = [r for r in recs if t0 <= r["t"] <= t1]
            third = [(ts >= t0 + i * (t1 - t0) / 3) & (ts <= t0 + (i + 1) * (t1 - t0) / 3) for i in range(3)]
            err = d - k
            out[t][k] = {"n": int(len(d)), "span_s": float(t1 - t0),
                         "detect_rate": len({a for a in ts}) / max(len(frames), 1),
                         "bias": float(np.median(err)), "sd": float(d.std()), "mae": float(np.mean(np.abs(err))),
                         "sigma_d_med": float(np.median(sd)),
                         "cover_1s": float(np.mean(np.abs(err) <= sd)),
                         "cover_2s": float(np.mean(np.abs(err) <= 2 * sd)),
                         "cover_1s_debiased": float(np.mean(np.abs(d - np.median(d)) <= sd)),
                         "facing_phase": [float(np.median(f[m])) if m.any() else None for m in third]}
    return out


def sigma_models():
    d = Data()
    dist = d.x["distance"]
    models = {"depth σ_d (logged)": d.x["distance_sigma"],
              "monocular σ_d = 0.15 + 0.08·d": 0.15 + 0.08 * dist,
              "depth-hole σ_d = 0.45": np.full_like(dist, 0.45)}
    res = {}
    for name, sd in models.items():
        x = dict(d.x)
        x["distance_sigma"] = sd
        n, r = (lambda o: (o[0], o[1]))(rates(d, d.gate(d.engage(x=x))))
        res[name] = {"false_trigger": n, "recall": r, "sigma_d_median": float(np.median(sd))}
    res["Type-1 (no FOU)"] = dict(zip(("false_trigger", "recall"), rates(d, d.gate(d.engage(it2=False)))))
    return res


def rates(d, ok):
    s = d.score(ok)
    n = [a for *_, lab, a in s if lab == "n"]
    r = [a for *_, lab, a in s if lab == "r"]
    return rate(sum(n), len(n)), rate(sum(r), len(r))


def main():
    c = cal()
    sm = sigma_models()
    pc = lambda r: f"{100 * r['p']:.0f}% ({r['k']}/{r['n']})"
    L = ["# Distance sources (task 8)", "",
         "## A. Depth distance on the calibration takes (1 person at 1/2/3/4 m marks)", "",
         "| Take | Mark | Frames | Person detected | Bias (median d − mark) | SD | Median σ_d | |err| ≤ σ_d | |err| ≤ 2σ_d | "
         "Facing by phase (thirds) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t, v in c.items():
        for k in MARKS:
            a = v[k]
            if a is None:
                L.append(f"| {t} | {k} m | 0 | – | – | – | – | – | – | – |")
                continue
            ph = " / ".join("–" if x is None else f"{x:.2f}" for x in a["facing_phase"])
            L.append(f"| {t} | {k} m | {a['n']} | {100 * a['detect_rate']:.0f}% | {a['bias']:+.2f} m | {a['sd']:.2f} m | "
                     f"{a['sigma_d_med']:.2f} m | {100 * a['cover_1s']:.0f}% | {100 * a['cover_2s']:.0f}% | {ph} |")
    pooled = {k: [v[k] for v in c.values() if v[k]] for k in MARKS}
    L += ["", "Pooled over the three takes (median of take values):", "",
          "| Mark | Bias | SD | Median σ_d | |err| ≤ σ_d | |err − bias| ≤ σ_d |", "|---|---|---|---|---|---|"]
    for k, vs in pooled.items():
        med = lambda key: float(np.median([a[key] for a in vs]))
        L.append(f"| {k} m | {med('bias'):+.2f} m | {med('sd'):.2f} m | {med('sigma_d_med'):.2f} m | "
                 f"{100 * med('cover_1s'):.0f}% | {100 * med('cover_1s_debiased'):.0f}% |")
    L += ["", "Notes: the mark is the nominal floor position, so the bias also includes the offset between the "
          "camera and the mark line. D1_CAL (first take) lost the person for long stretches; D1_CAL_r2/r3 are clean. "
          "Phase order within a mark is taken from time (front, then the two non-frontal phases); it should be "
          "checked against the video.", "",
          "## B. IT2 with different σ_d models (14 scenario takes, logged depth distance)", "",
          "Monocular *distance values* need pose landmarks, which are not in the logs (re-run on raw.mp4 from the "
          "31 Oct data). This table changes only σ_d.", "",
          "| σ_d model | Median σ_d | Bystander false triggers | Recall |", "|---|---|---|---|"]
    for name, v in sm.items():
        L.append(f"| {name} | {v.get('sigma_d_median', 0):.2f} m | {pc(v['false_trigger'])} | {pc(v['recall'])} |")
    save("distance", {"calibration": c, "sigma_models": sm}, "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
