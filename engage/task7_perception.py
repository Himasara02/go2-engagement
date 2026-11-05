#!/usr/bin/env python3
"""Task 7: hand/gesture perception, full frame (takes_v2) vs per-person centred
crops (takes_crop2), on the same people, track IDs and frames.

The full-frame result per person is taken from the crop2 logs' own
`gesture_full_frame` / `gesture_conf_full_frame` fields, which are identical to
the takes_v2 logs (checked on every take whose takes_v2 log is present), so all
14 takes are compared, D2_S1_r1 included.
Per take: command-gesture person-frames (command class, conf >= 0.4, the
episode rule), episodes (label_episodes.episodes) and the commander's command
frames (commander track IDs from takes_crop/labels_cmd.json).
Also: gesture classes of labelled crop2 episodes (classifier confusion note).
Writes results/perception.{json,md}.
"""
import collections
import json
import os
import shutil
import tempfile

from analysis import MAIN_DIR, save, scenario_takes
from engage.methods import COMMAND_MAP
from label_episodes import CONF, episodes

ROOT = os.path.dirname(MAIN_DIR)


def full_frame_view(take, tmp):
    """Copy of a crop2 log with the full-frame gesture as `gesture` (for episodes())."""
    d = os.path.join(tmp, os.path.basename(take))
    os.makedirs(d)
    with open(os.path.join(take, "log.jsonl")) as f, open(os.path.join(d, "log.jsonl"), "w") as g:
        for line in f:
            r = json.loads(line)
            for p in r["people"]:
                p["gesture"], p["gesture_conf"] = p.get("gesture_full_frame"), p.get("gesture_conf_full_frame", 0.0)
            g.write(json.dumps(r) + "\n")
    return d


def stats(take_dir, commanders):
    recs, eps = episodes(take_dir)
    cmd = [(p["tid"], p["gesture"]) for r in recs for p in r["people"]
           if p.get("gesture") in COMMAND_MAP and p["gesture_conf"] >= CONF]
    return {"cmd_frames": len(cmd), "episodes": len(eps),
            "commander_frames": sum(t in commanders for t, _ in cmd),
            "commander_bursts": sum(e["tid"] in commanders for e in eps)}


def check_v2(name, view):
    """True/False if takes_v2 log present and identical in gesture fields, None if absent."""
    v2 = os.path.join(ROOT, "takes_v2", name, "log.jsonl")
    if not os.path.exists(v2):
        return None
    a = [json.loads(l) for l in open(v2)]
    b = [json.loads(l) for l in open(os.path.join(view, "log.jsonl"))]
    key = lambda r: sorted((p["tid"], p.get("gesture"), round(p["gesture_conf"], 3)) for p in r["people"])
    return len(a) == len(b) and all(key(x) == key(y) for x, y in zip(a, b))


def main():
    tmp = tempfile.mkdtemp()
    rows, res = [], {"per_take": {}, "v2_identical": {}}
    tot = collections.Counter()
    try:
        for t in scenario_takes():
            name = os.path.basename(t)
            cf = os.path.join(ROOT, "takes_crop", name, "labels_cmd.json")
            commanders = set(json.load(open(cf)).get("_commander_tids", [])) if os.path.exists(cf) else set()
            view = full_frame_view(t, tmp)
            res["v2_identical"][name] = check_v2(name, view)
            ff, cr = stats(view, commanders), stats(t, commanders)
            res["per_take"][name] = {"full_frame": ff, "crop": cr, "commander_tids": sorted(commanders)}
            for k in ff:
                tot["ff_" + k] += ff[k]
                tot["cr_" + k] += cr[k]
            rows.append(f"| {name} | {ff['cmd_frames']} → {cr['cmd_frames']} | {ff['episodes']} → {cr['episodes']} | "
                        + (f"{ff['commander_frames']} → {cr['commander_frames']}" if commanders else "– (none chosen)")
                        + " |")
    finally:
        shutil.rmtree(tmp)

    # classifier note: classes of labelled crop2 episodes
    cls = {"r": collections.Counter(), "n": collections.Counter(), "x": collections.Counter()}
    allg = collections.Counter()
    for t in scenario_takes():
        labels = json.load(open(os.path.join(t, "labels.json")))
        recs, eps = episodes(t)
        for e in eps:
            lab = labels.get(str(e["id"]))
            if lab in cls:
                cls[lab][e["gesture"]] += 1
        for r in recs:
            for p in r["people"]:
                if p.get("gesture") and p["gesture_conf"] >= CONF:
                    allg[p["gesture"]] += 1
    res["totals"] = dict(tot)
    res["episode_classes"] = {k: dict(v) for k, v in cls.items()}
    res["all_classes_conf_ge_0.4"] = dict(allg)

    same = [k for k, v in res["v2_identical"].items() if v]
    absent = [k for k, v in res["v2_identical"].items() if v is None]
    cmdset = ", ".join(sorted(COMMAND_MAP))
    md = "\n".join([
        "# Perception: full-frame hands vs per-person centred crops (task 7)", "",
        f"Same people, track IDs and frames in both; only the hand stage differs. Full-frame values "
        f"identical to the takes_v2 logs for {len(same)} takes (takes_v2 log not available: {', '.join(absent) or 'none'}).", "",
        f"Command gesture = class in {{{cmdset}}} with confidence ≥ {CONF}.", "",
        "| Take | Command person-frames (full → crop) | Episodes (full → crop) | Commander command frames (full → crop) |",
        "|---|---|---|---|", *rows,
        f"| **Total** | **{tot['ff_cmd_frames']} → {tot['cr_cmd_frames']}** | **{tot['ff_episodes']} → {tot['cr_episodes']}** | "
        f"**{tot['ff_commander_frames']} → {tot['cr_commander_frames']}** |", "",
        "## Gesture classes of labelled crop2 episodes", "",
        "| Label | Classes |", "|---|---|",
        *[f"| {k} | " + ", ".join(f"{g} {n}" for g, n in v.most_common()) + " |" for k, v in cls.items()], "",
        "All person-frames with conf ≥ 0.4 (crop2), by class: " +
        ", ".join(f"{g} {n}" for g, n in allg.most_common()) + ".", "",
        "Note: there is no per-frame ground truth of the intended gesture class, so this is a class "
        "distribution, not a confusion matrix. Known issue (collector's handover): `call` is often read as "
        "`mute` (both map to backward), `one`/`two_up` are rare, relaxed hands mostly read as `dislike` "
        "(not a command). This affects which command would run, not whether a gesture is for the robot."])
    save("perception", res, md)
    print(md)


if __name__ == "__main__":
    main()
