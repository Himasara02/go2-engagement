#!/usr/bin/env python3
"""Aggregate labelled takes -> poster table, figures, results.json.

Per labelled episode and per policy, the episode counts as ACCEPTED if the
policy's sustained decision included that person at any frame of it.
  bystander false-trigger rate = accepted 'n' episodes / all 'n' episodes
  command recall               = accepted 'r' episodes / all 'r' episodes
95% CIs: Wilson interval.
Usage: python evaluate.py takes/*   (writes results/)
"""
import glob
import json
import math
import os
import sys

from engage.methods import METHODS, accept_per_person, engagement
from label_episodes import episodes

NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person",
         "type1": "+ Type-1 engagement", "it2": "+ Interval Type-2"}


def recompute(recs):
    """Re-run all four policies from the logged features with the CURRENT
    parameters, so every take is scored with identical settings."""
    cnt = {m: {} for m in METHODS}
    for r in recs:
        for p in r["people"]:
            p.update(engagement(p))
        a = accept_per_person(r["people"])
        r["sus"] = {}
        for m in METHODS:
            c = cnt[m]
            for t in list(c):
                if t not in a[m]:
                    del c[t]
            for t in a[m]:
                c[t] = c.get(t, 0) + 1
            r["sus"][m] = [t for t, v in c.items() if v >= 3]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def main():
    takes = [t for t in (sys.argv[1:] or glob.glob("takes/*"))
             if os.path.exists(os.path.join(t, "labels.json"))]
    os.makedirs("results", exist_ok=True)
    cnt = {}          # (scenario, method, label) -> [accepted, total]
    e_vals = {"r": [], "n": []}
    minutes = {}
    per_take = []
    for t in takes:
        meta = json.load(open(os.path.join(t, "meta.json")))
        sc = meta["scenario"]
        minutes[sc] = minutes.get(sc, 0) + meta["duration_s"] / 60
        labels = json.load(open(os.path.join(t, "labels.json")))
        recs, eps = episodes(t)
        recompute(recs)
        nr = nn = 0
        for e in eps:
            lab = labels.get(str(e["id"]))
            if lab not in ("r", "n"):
                continue
            nr += lab == "r"
            nn += lab == "n"
            frs = recs[e["start"]:e["end"] + 1]
            for m in METHODS:
                hit = any(e["tid"] in r["sus"][m] for r in frs)
                c = cnt.setdefault((sc, m, lab), [0, 0])
                c[0] += hit
                c[1] += 1
            lows = [p["e2_low"] for r in frs for p in r["people"] if p["tid"] == e["tid"]]
            if lows:
                e_vals[lab].append(sum(lows) / len(lows))
        per_take.append({"take": meta["take"], "scenario": sc, "robot_eps": nr, "other_eps": nn,
                         "duration_s": meta["duration_s"], "fps": meta.get("mean_fps")})

    res = {"takes": per_take, "minutes": minutes, "table": {}}
    for sc in ("stationary", "walking"):
        for m in METHODS:
            a, n = cnt.get((sc, m, "n"), [0, 0])
            ar, nr = cnt.get((sc, m, "r"), [0, 0])
            res["table"][f"{sc}/{m}"] = {
                "false_trigger": wilson(a, n), "false_n": [a, n],
                "recall": wilson(ar, nr), "recall_n": [ar, nr]}
    res["engagement_e2low"] = e_vals
    json.dump(res, open("results/results.json", "w"), indent=1)

    lines = ["| Policy | Stationary FT | Stationary recall | Walking FT | Walking recall |",
             "|---|---|---|---|---|"]
    for m in METHODS:
        row = [NAMES[m]]
        for sc in ("stationary", "walking"):
            r = res["table"][f"{sc}/{m}"]
            ft, fn = r["false_trigger"], r["false_n"]
            rc, rn = r["recall"], r["recall_n"]
            row.append(f"{100*ft[0]:.0f}% ({fn[0]}/{fn[1]})" if fn[1] else "–")
            row.append(f"{100*rc[0]:.0f}% ({rn[0]}/{rn[1]})" if rn[1] else "–")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    open("results/table.md", "w", encoding="utf-8").write(md + "\n")
    print(md)
    print("takes:", len(per_take), " minutes:", {k: round(v, 1) for k, v in minutes.items()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, sc in zip(axs, ("stationary", "walking")):
            xs = range(len(METHODS))
            ft = [100 * res["table"][f"{sc}/{m}"]["false_trigger"][0] for m in METHODS]
            rc = [100 * res["table"][f"{sc}/{m}"]["recall"][0] for m in METHODS]
            ax.bar([x - 0.2 for x in xs], ft, 0.4, label="bystander false triggers", color="#c0392b")
            ax.bar([x + 0.2 for x in xs], rc, 0.4, label="commander recall", color="#2e86c1")
            ax.set_xticks(list(xs), ["Gesture\nonly", "Nearest", "Type-1", "IT2"])
            ax.set_ylim(0, 105)
            ax.set_title(sc.capitalize())
            ax.set_ylabel("%")
            ax.spines[["top", "right"]].set_visible(False)
        axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig("results/bars.png", dpi=200)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.hist(e_vals["r"], bins=15, range=(0, 1), alpha=.7, label="meant for robot", color="#2e86c1")
        ax.hist(e_vals["n"], bins=15, range=(0, 1), alpha=.7, label="not for robot", color="#c0392b")
        ax.axvline(0.6, ls="--", c="k", lw=1)
        ax.set_xlabel("IT2 engagement lower bound $E_{low}$")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig("results/engagement_hist.png", dpi=200)
    except ImportError:
        print("matplotlib missing - skipped figures")


if __name__ == "__main__":
    main()
