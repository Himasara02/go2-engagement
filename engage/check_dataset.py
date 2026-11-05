#!/usr/bin/env python3
"""Check every take before handing the dataset over.

    python check_dataset.py            # checks takes/*, writes dataset_report.md

For each take it checks that all files are there, that annotator A and B
labelled every episode, and counts the labels. Fix anything marked PROBLEM
(usually: run label_episodes.py again to finish labelling).
"""
import glob
import json
import os
import sys
from collections import Counter

from label_episodes import episodes

REQUIRED = ["raw.mp4", "annot.mp4", "log.jsonl", "meta.json", "frames.csv"]


def check(t):
    probs, row = [], {"take": os.path.basename(t)}
    for f in REQUIRED:
        if not os.path.exists(os.path.join(t, f)):
            probs.append(f"missing {f}")
    if probs:
        return row, probs
    meta = json.load(open(os.path.join(t, "meta.json")))
    row.update(scenario=meta.get("scenario"), location=meta.get("location"),
               commander=meta.get("commander"), people=len(meta.get("people", [])),
               dur=meta.get("duration_s"), fps=meta.get("mean_fps"), depth=meta.get("has_depth"))
    if meta.get("scenario") in ("calibration", "test"):
        row["note"] = "not labelled (" + meta["scenario"] + ")"
        return row, probs
    if not meta.get("commander"):
        probs.append("no --commander recorded")
    if not meta.get("has_depth"):
        probs.append("no depth saved")
    if meta.get("raw_format") == 2 and ((meta.get("max_people") or 4) < 6 or meta.get("perception_mode") != "image"):
        probs.append("old detection (misses people) - run: python reprocess.py, then label again")
    _, eps = episodes(t)
    row["episodes"] = len(eps)
    for ann, fname in (("A", "labels.json"), ("B", "labels_B.json")):
        p = os.path.join(t, fname)
        lab = json.load(open(p)) if os.path.exists(p) else {}
        c = Counter(lab.get(str(e["id"])) for e in eps)
        miss = lab.get("_missed", "-")
        row[ann] = f"r{c['r']} n{c['n']} x{c['x']} missed {miss}"
        if not c[None] and eps and "_missed" not in lab:
            probs.append(f"annotator {ann}: missed-commands question not answered (run label_episodes.py again)")
        if c[None]:
            probs.append(f"annotator {ann}: {c[None]} of {len(eps)} episodes not labelled")
        row[f"_{ann}"] = c
    if len(eps) == 0:
        probs.append("no gesture episodes found (were hands visible?)")
    return row, probs


def main():
    takes = sorted(d for d in glob.glob("takes/*") if os.path.isdir(d))
    if not takes:
        sys.exit("no takes/ found - run this inside the engage folder")
    rows, all_probs = [], {}
    tot = Counter()
    for t in takes:
        row, probs = check(t)
        rows.append(row)
        if probs:
            all_probs[row["take"]] = probs
        if "_A" in row:
            tot["r"] += row["_A"]["r"]
            tot["n"] += row["_A"]["n"]
            tot["eps"] += row.get("episodes", 0)
            tot[f"takes_{row['scenario']}"] += 1
            tot["minutes"] += (row.get("dur") or 0) / 60
    lines = ["# Dataset report", "",
             f"Takes: {len(takes)}  |  stationary {tot['takes_stationary']}  walking {tot['takes_walking']}"
             f"  |  {tot['minutes']:.0f} min  |  episodes {tot['eps']}  |  annotator A: "
             f"for robot {tot['r']}, not for robot {tot['n']}", "",
             "| take | scenario | location | commander | people | s | fps | depth | episodes | A (r/n/x) | B (r/n/x) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in
                                         ("take", "scenario", "location", "commander", "people", "dur", "fps",
                                          "depth", "episodes", "A", "B")) + " |")
    lines += ["", "## Problems", ""]
    if all_probs:
        for k, v in all_probs.items():
            lines.append(f"- **{k}**: " + "; ".join(v))
    else:
        lines.append("None. Ready to hand over.")
    open("dataset_report.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    goal = "OK" if tot["r"] + tot["n"] >= 300 else "below the 300 target"
    print(f"\nlabelled episodes (A, r+n): {tot['r'] + tot['n']}  -> {goal}")
    print("wrote dataset_report.md")
    sys.exit(1 if all_probs else 0)


if __name__ == "__main__":
    main()
