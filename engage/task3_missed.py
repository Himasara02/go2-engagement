#!/usr/bin/env python3
"""Task 3: recall with and without the robot-directed commands the detector
never turned into a clip (labels.json `_missed`), plus perception recall.

  recall (clips)        = accepted r / r clips
  recall (incl. missed) = accepted r / (r clips + missed)
  perception recall     = r clips / (r clips + missed)
x clips (bad detections) are excluded from both rates and counted.
Writes results/missed.{json,md}.
"""
from analysis import MAIN_DIR, NAMES, episode_records, fmt, rate, save, scenario_takes
from engage.methods import METHODS


def main():
    eps, missed = episode_records(scenario_takes())
    r = [e for e in eps if e["label"] == "r"]
    n = [e for e in eps if e["label"] == "n"]
    x = [e for e in eps if e["label"] == "x"]
    unl = [e for e in eps if e["label"] not in ("r", "n", "x")]
    m_tot = sum(missed.values())

    res = {"data": MAIN_DIR, "clips": {"r": len(r), "n": len(n), "x": len(x), "unlabelled": len(unl)},
           "missed": missed, "missed_total": m_tot,
           "perception_recall": rate(len(r), len(r) + m_tot), "policies": {}}
    rows = []
    for m in METHODS:
        ar = sum(e["acc"][m] for e in r)
        an = sum(e["acc"][m] for e in n)
        d = {"false_trigger": rate(an, len(n)), "recall_clips": rate(ar, len(r)),
             "recall_incl_missed": rate(ar, len(r) + m_tot)}
        res["policies"][m] = d
        rows.append(f"| {NAMES[m]} | {fmt(d['false_trigger'])} | {fmt(d['recall_clips'])} | "
                    f"{fmt(d['recall_incl_missed'])} |")

    per_take = ["| Take | r clips | missed | perception recall |", "|---|---|---|---|"]
    for t in sorted(missed):
        k = sum(e["label"] == "r" for e in eps if e["take"] == t)
        per_take.append(f"| {t} | {k} | {missed[t]} | {fmt(rate(k, k + missed[t]))} |")

    md = "\n".join([
        "# Missed commands (task 3)", "",
        f"Clips: {len(r)} r, {len(n)} n, {len(x)} x (excluded), {len(unl)} unlabelled. "
        f"Robot-directed commands never detected as a clip: {m_tot}.", "",
        f"Perception recall (r clips / (r clips + missed)): {fmt(res['perception_recall'], ci=True)}", "",
        "| Policy | Bystander false triggers | Recall (clips) | Recall (incl. missed) |",
        "|---|---|---|---|", *rows, "",
        "## Per take", "", *per_take])
    save("missed", res, md)
    print(md)


if __name__ == "__main__":
    main()
