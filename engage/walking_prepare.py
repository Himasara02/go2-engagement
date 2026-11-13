#!/usr/bin/env python3
"""Prepare the Day-5 walking recordings (handover_D5_walking_2026-11-12) for the same pipeline as the standing ones.

    python3 walking_prepare.py [--check-standing]

1. takes_crop3/<take>/raw.mp4 -> symlink to takes/<take>/raw.mp4 (the handover's restore_crop_videos.py copies them;
   a link saves 3.5 GB and gives the same bytes).
2. labels_walking.csv in the format of labels_final.csv (take, clip, track_id, gesture, first_frame, last_frame,
   start_s, end_s, frames, label, label_source, commander), one row per episode of label_episodes.episodes() (the
   same function as for the standing labels pack). Label file per the handover README: W1-W3 labels_cmd.json
   (commander chosen once), W4-W6 labels.json (episode by episode). W3_r1 (bad take) has no crop3 folder.
--check-standing rebuilds the standing labels_final.csv rows with the same code and reports mismatches.
"""
import csv
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WALK = os.path.join(ROOT, "data", "walking_2026-11-12", "handover_D5_walking_2026-11-12", "engage")
STAND = os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3")
LABELS_A = os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10", "labels_final.csv")
OUT = os.path.join(ROOT, "data", "walking_2026-11-12", "labels_walking.csv")
FIELDS = ["take", "clip", "track_id", "gesture", "first_frame", "last_frame", "start_s", "end_s", "frames", "label",
          "label_source", "commander"]

sys.path.insert(0, WALK)
spec = importlib.util.spec_from_file_location("label_episodes_walk", os.path.join(WALK, "label_episodes.py"))
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)


def rows_for(take_dir, label_file):
    name = os.path.basename(take_dir)
    recs, eps = le.episodes(take_dir)
    lab = json.load(open(os.path.join(take_dir, label_file)))
    meta = json.load(open(os.path.join(take_dir, "meta.json")))
    t = {r["i"]: r["t"] for r in recs}
    out = []
    for e in eps:
        out.append({"take": name, "clip": e["id"], "track_id": e["tid"], "gesture": e["gesture"],
                    "first_frame": e["start"], "last_frame": e["end"], "start_s": round(t[e["start"]], 3),
                    "end_s": round(t[e["end"]], 3), "frames": len(e["frames"]), "label": lab.get(str(e["id"]), ""),
                    "label_source": label_file, "commander": meta.get("commander", "")})
    return out


if "--check-standing" in sys.argv:
    ref = list(csv.DictReader(open(LABELS_A)))
    bad = n = 0
    for take in sorted({r["take"] for r in ref}):
        mine = {str(r["clip"]): r for r in rows_for(os.path.join(STAND, take), next(r["label_source"] for r in ref if r["take"] == take))}
        for r in (x for x in ref if x["take"] == take):
            n += 1
            m = mine.get(r["clip"])
            if not m or any(str(m[k]) != r[k] for k in ("track_id", "gesture", "first_frame", "last_frame", "label")):
                bad += 1
    print(f"standing check: {n} rows, {bad} mismatches")
    sys.exit(0)

links = 0
for t in sorted(os.listdir(os.path.join(WALK, "takes_crop3"))):
    dst, src = os.path.join(WALK, "takes_crop3", t, "raw.mp4"), os.path.join(WALK, "takes", t, "raw.mp4")
    if not os.path.exists(dst):
        os.symlink(src, dst)
        links += 1
rows = []
for t in sorted(os.listdir(os.path.join(WALK, "takes_crop3"))):
    lf = "labels_cmd.json" if t.split("_")[1] in ("W1", "W2", "W3") else "labels.json"
    rows += rows_for(os.path.join(WALK, "takes_crop3", t), lf)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    w.writerows(rows)
from collections import Counter
print(f"linked {links} videos; wrote {OUT}: {len(rows)} episodes", dict(Counter(r["label"] for r in rows)))
print({t: dict(Counter(r["label"] for r in rows if r["take"] == t)) for t in sorted({r["take"] for r in rows})})
