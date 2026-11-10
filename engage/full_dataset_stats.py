#!/usr/bin/env python3
"""Statistics of the full labelled dataset (received 10 Nov 2026) for the CVPR paper.

    python3 full_dataset_stats.py

Reads (never modifies):
  ../../data/dataset_2026-11-10/takes_crop3/<take>/{meta.json, log.jsonl, frames.csv}
  ../../data/labels_pack_2026-11-10/labels_final.csv   (one row per clip, final label r / n / x)
Writes:
  results/full/dataset.json, results/full/dataset.md
  ../../cvpr2027_paper/tables/dataset.tex             (generated, do not edit by hand)
Standard library only.
"""
import csv
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TAKES = os.path.join(ROOT, "data", "dataset_2026-11-10", "takes_crop3")
LABELS = os.path.join(ROOT, "data", "labels_pack_2026-11-10", "labels_final.csv")
OUT = os.path.join(HERE, "results", "full")
TABLE = os.path.join(ROOT, "cvpr2027_paper", "tables", "dataset.tex")
SCEN = {"S1": "single commander", "S2": "bystanders gesture", "S3": "simultaneous gestures",
        "S4": "turns to a friend", "S5": "two commanders", "S6": "bystanders walk past"}


def take_info(d):
    name = os.path.basename(d)
    m = json.load(open(os.path.join(d, "meta.json")))
    ppl = [len(json.loads(line)["people"]) for line in open(os.path.join(d, "log.jsonl"))]
    ts = [float(r["t"]) for r in csv.DictReader(open(os.path.join(d, "frames.csv")))]
    return {"take": name, "day": name.split("_")[0], "scenario": name.split("_")[1],
            "duration_s": m["duration_s"], "frames": len(ppl),
            "fps": (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None,
            "people_mean": st.mean(ppl) if ppl else 0.0, "people_max": max(ppl) if ppl else 0,
            "participant_codes": m.get("people", []), "location": m.get("location"),
            "has_depth_recorded": m.get("has_depth")}


def main():
    takes = {t["take"]: t for t in map(take_info, sorted(glob.glob(os.path.join(TAKES, "*"))))}
    clips = list(csv.DictReader(open(LABELS)))
    assert {c["take"] for c in clips} <= set(takes), "labels for takes that are not in the dataset"
    lab = defaultdict(Counter)
    src = {}
    gest = defaultdict(Counter)
    for c in clips:
        lab[c["take"]][c["label"]] += 1
        src[c["take"]] = c["label_source"]
        gest[c["label"]][c["gesture"]] += 1

    per_s = {}
    for s in SCEN:
        ts = [t for t in takes.values() if t["scenario"] == s]
        frames = sum(t["frames"] for t in ts)
        per_s[s] = {"recordings": len(ts), "minutes": sum(t["duration_s"] for t in ts) / 60,
                    "people_mean": sum(t["people_mean"] * t["frames"] for t in ts) / frames if frames else 0,
                    **{k: sum(lab[t["take"]][k] for t in ts) for k in "rnx"},
                    "clip_by_clip": sum(src.get(t["take"]) == "labels.json" for t in ts)}
    allf = sum(t["frames"] for t in takes.values())
    fps = [t["fps"] for t in takes.values() if t["fps"]]
    tot = {"recordings": len(takes), "minutes": sum(t["duration_s"] for t in takes.values()) / 60,
           "people_mean": sum(t["people_mean"] * t["frames"] for t in takes.values()) / allf,
           "people_max": max(t["people_max"] for t in takes.values()),
           **{k: sum(v[k] for v in per_s.values()) for k in "rnx"}, "clips": len(clips),
           "days": sorted({t["day"] for t in takes.values()}),
           "locations": sorted({t["location"] for t in takes.values()}),
           "participant_codes": sorted({p for t in takes.values() for p in t["participant_codes"]}),
           "label_source": dict(Counter(src.values())),
           "fps_median": st.median(fps), "fps_min": min(fps), "fps_max": max(fps),
           "depth_recorded_all": all(t["has_depth_recorded"] for t in takes.values())}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"source": {"takes": TAKES, "labels": LABELS}, "total": tot, "per_scenario": per_s,
               "per_take": list(takes.values()), "per_take_labels": {k: dict(v) for k, v in lab.items()},
               "gesture_by_label": {k: dict(v.most_common()) for k, v in gest.items()}},
              open(os.path.join(OUT, "dataset.json"), "w"), indent=1)

    md = ["# Full dataset (received 10 Nov 2026): statistics", "",
          f"{tot['recordings']} standing recordings, {tot['minutes']:.1f} min, days {', '.join(tot['days'])}, "
          f"location(s) {', '.join(tot['locations'])}, {len(tot['participant_codes'])} participant codes "
          "(placeholders, see the label pack's GUIDE).", "",
          f"Clips: {tot['clips']} = {tot['r']} r / {tot['n']} n / {tot['x']} x. Label source: "
          f"{tot['label_source']}. People in view per frame: mean {tot['people_mean']:.1f}, max {tot['people_max']}. "
          f"Logged frame rate: median {tot['fps_median']:.1f} fps (range {tot['fps_min']:.1f}-{tot['fps_max']:.1f}).", "",
          "| Scenario | Rec. | Min | People/frame | r | n | x | clip-by-clip takes |", "|---|---|---|---|---|---|---|---|"]
    for s, v in per_s.items():
        md.append(f"| {s} {SCEN[s]} | {v['recordings']} | {v['minutes']:.1f} | {v['people_mean']:.1f} | {v['r']} | "
                  f"{v['n']} | {v['x']} | {v['clip_by_clip']} |")
    md.append(f"| Total | {tot['recordings']} | {tot['minutes']:.1f} | {tot['people_mean']:.1f} | {tot['r']} | "
              f"{tot['n']} | {tot['x']} | {tot['label_source'].get('labels.json', 0)} |")
    md += ["", "Gesture classes of the clips, by label:", ""]
    for k, v in sorted(gest.items()):
        md.append(f"- {k}: " + ", ".join(f"{g} {n}" for g, n in v.most_common()))
    open(os.path.join(OUT, "dataset.md"), "w").write("\n".join(md) + "\n")

    rows = "\n".join(f"{s} & {SCEN[s]:<22} & {v['recordings']} & {v['minutes']:.1f} & {v['people_mean']:.1f} & "
                     f"{v['r']} & {v['n']} & {v['x']}\\\\" for s, v in per_s.items())
    walk = "\n".join(f"{w} & {d:<22} & \\RES{{--}} & \\RES{{--}} & \\RES{{--}} & \\RES{{--}} & \\RES{{--}} & \\RES{{--}}\\\\"
                     for w, d in (("W1", "bystanders still"), ("W2", "bystanders gesture"), ("W3", "crossing")))
    tex = rf"""% generated by go2-engagement/engage/full_dataset_stats.py from the 10 Nov 2026 dataset - do not edit by hand
\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{2.6pt}}
\caption{{\dataset statistics per scenario: recordings, minutes, mean number
of people in view per frame, and gesture episodes addressed to the robot
($r$), addressed elsewhere ($n$) or excluded as detection errors ($x$; only
annotated in the {tot['label_source'].get('labels.json', 0)} recordings labelled episode by episode).
\TODO{{walking rows}}}}
\label{{tab:dataset}}
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
& Scenario & Rec. & Min & People & $r$ & $n$ & $x$\\
\midrule
\multicolumn{{8}}{{@{{}}l}}{{\emph{{Robot standing}}}}\\
{rows}
\midrule
\multicolumn{{8}}{{@{{}}l}}{{\emph{{Robot walking}}}}\\
{walk}
\midrule
& Total (standing)          & {tot['recordings']} & {tot['minutes']:.1f} & {tot['people_mean']:.1f} & {tot['r']:,} & {tot['n']:,} & {tot['x']}\\
\bottomrule
\end{{tabular}}
\end{{table}}
""".replace(",", "{,}")
    open(TABLE, "w").write(tex)
    print("\n".join(md[:9]))
    print("wrote", OUT, "and", TABLE)


if __name__ == "__main__":
    main()
