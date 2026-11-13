#!/usr/bin/env python3
"""Tracking quality on the full dataset: track IDs per person in view (fragmentation).

    python3 full_tracking_stats.py

People in view per recording = 95th percentile of the per-frame person count (people standing still are
nearly always detected). Tracks = track IDs present for >= 10 frames. Extra tracks per person =
tracks / people - 1; extra tracks per minute ~ ID switches per minute (each new ID for a person who was
already tracked). Reads data/dataset_2026-11-11/takes_crop3/*/log.jsonl. Writes results/full/tracking.{json,md}.
"""
import glob
import json
import math
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TAKES = os.environ.get("TAKES", os.path.join(HERE, "..", "..", "data", "dataset_2026-11-11", "takes_crop3"))
NAME = os.environ.get("TRACK_NAME", "tracking")   # tracking_walking for the walking recordings
rows = []
for d in sorted(glob.glob(os.path.join(TAKES, "*"))):
    recs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    meta = json.load(open(os.path.join(d, "meta.json")))
    counts = sorted(len(r["people"]) for r in recs)
    people = max(1, counts[min(len(counts) - 1, int(math.ceil(0.95 * len(counts))) - 1)])
    life = Counter(p["tid"] for r in recs for p in r["people"])
    tracks = sum(1 for v in life.values() if v >= 10)
    rows.append({"take": os.path.basename(d), "scenario": os.path.basename(d).split("_")[1], "people": people,
                 "tracks": tracks, "minutes": meta["duration_s"] / 60,
                 "extra_per_min": max(0, tracks - people) / (meta["duration_s"] / 60)})
by = {}
for s in sorted({r["scenario"] for r in rows}):
    rs = [r for r in rows if r["scenario"] == s]
    by[s] = {"recordings": len(rs), "people": sum(r["people"] for r in rs), "tracks": sum(r["tracks"] for r in rs),
             "tracks_per_person": sum(r["tracks"] for r in rs) / sum(r["people"] for r in rs),
             "extra_per_min": sum(max(0, r["tracks"] - r["people"]) for r in rs) / sum(r["minutes"] for r in rs)}
tot = {"tracks_per_person": sum(r["tracks"] for r in rows) / sum(r["people"] for r in rows),
       "extra_per_min": sum(max(0, r["tracks"] - r["people"]) for r in rows) / sum(r["minutes"] for r in rows),
       "standing_only": None}
st = [r for r in rows if r["scenario"] != "S6"] or rows
tot["standing_only"] = {"tracks_per_person": sum(r["tracks"] for r in st) / sum(r["people"] for r in st),
                        "extra_per_min": sum(max(0, r["tracks"] - r["people"]) for r in st) / sum(r["minutes"] for r in st)}
out = os.path.join(HERE, "results", "full")
json.dump({"per_take": rows, "by_scenario": by, "total": tot}, open(os.path.join(out, NAME + ".json"), "w"), indent=1)
md = ["# Tracking: track IDs per person in view (tracks >= 10 frames)", "",
      "| Scenario | Rec. | People | Tracks | Tracks/person | Extra IDs/min |", "|---|---|---|---|---|---|"]
md += [f"| {s} | {v['recordings']} | {v['people']} | {v['tracks']} | {v['tracks_per_person']:.2f} | {v['extra_per_min']:.1f} |"
       for s, v in by.items()]
md += ["", f"All: {tot['tracks_per_person']:.2f} tracks per person, {tot['extra_per_min']:.1f} extra IDs per minute. "
       f"Without S6 (people walk across): {tot['standing_only']['tracks_per_person']:.2f}, "
       f"{tot['standing_only']['extra_per_min']:.1f} per minute."]
open(os.path.join(out, NAME + ".md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
