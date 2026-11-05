"""Shared helpers for the paper analysis (tasks 3-8).

Loads the labelled takes, recomputes every policy from the logged features
with the frozen parameters (same code path as evaluate.py) and returns one
record per labelled episode. Nothing here writes to the dataset.
"""
import glob
import json
import math
import os

from engage.methods import METHODS
from evaluate import recompute, wilson
from label_episodes import episodes

HERE = os.path.dirname(os.path.abspath(__file__))
# rebuilt dataset if present, else the identical takes_crop2 files shipped in the 3 Nov update
_CANDIDATES = [os.path.join(HERE, "..", "..", "dataset_2026-11-03", "engage", "takes_crop2"),
               os.path.join(HERE, "..", "handover", "handover_2026-11-03_update", "data",
                            "engage", "takes_crop2")]
MAIN_DIR = next((os.path.normpath(d) for d in _CANDIDATES if os.path.isdir(d)), None)
RESULTS = os.path.join(HERE, "results")

NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person",
         "type1": "Type-1 fuzzy", "it2": "IT2 fuzzy"}


def scenario_takes(root=MAIN_DIR):
    """The labelled scenario takes D*_S* (no calibration or test takes)."""
    return sorted(t for t in glob.glob(os.path.join(root, "D*_S*"))
                  if os.path.exists(os.path.join(t, "labels.json")))


def load_take(t, label_file="labels.json"):
    meta = json.load(open(os.path.join(t, "meta.json")))
    labels = json.load(open(os.path.join(t, label_file)))
    recs, eps = episodes(t)
    recompute(recs)
    return meta, labels, recs, eps


def load_all(takes, label_file="labels.json"):
    return [load_take(t, label_file) for t in takes]


def sustained(frames_acc, n=3):
    """Per-frame accepted tid sets -> per-frame sustained sets (>= n frames in a row),
    the same rule as run_live.py / evaluate.py."""
    cnt, out = {}, []
    for acc in frames_acc:
        cnt = {t: cnt.get(t, 0) + 1 for t in acc}
        out.append({t for t, c in cnt.items() if c >= n})
    return out


def episode_records(takes, label_file="labels.json", extra=None, loaded=None):
    """One dict per episode: take, scenario, label (r/n/x/None), tid, gesture,
    start/end, acc[method] (bool), mean features.
    extra(recs, take) -> {method: per-frame list of sustained tid sets} adds policies.
    loaded: output of load_all(), to avoid reloading."""
    out, missed = [], {}
    for meta, labels, recs, eps in (loaded or load_all(takes, label_file)):
        name = meta["take"]
        missed[name] = int(labels.get("_missed", 0) or 0)
        sus = {m: [set(r["sus"][m]) for r in recs] for m in METHODS}
        if extra:
            sus.update(extra(recs, name))
        for e in eps:
            frs = range(e["start"], e["end"] + 1)
            feats = [p for i in frs for p in recs[i]["people"] if p["tid"] == e["tid"]]
            mean = (lambda k: sum(p[k] for p in feats) / len(feats)) if feats else (lambda k: math.nan)
            out.append({
                "take": name, "scenario": name.split("_")[1], "label": labels.get(str(e["id"])),
                "tid": e["tid"], "gesture": e["gesture"], "start": e["start"], "end": e["end"],
                "acc": {m: any(e["tid"] in s[i] for i in frs) for m, s in sus.items()},
                **{k: mean(k) for k in ("distance", "distance_sigma", "facing", "facing_sigma",
                                        "approach", "e_t1", "e2_low")},
            })
    return out, missed


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi}


def fmt(r, ci=False):
    if not r["n"]:
        return "–"
    s = f"{100 * r['p']:.0f}% ({r['k']}/{r['n']})"
    return s + f" [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}]" if ci else s


def save(name, obj, md=None):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(RESULTS, name + ".json"), "w"), indent=1)
    if md is not None:
        open(os.path.join(RESULTS, name + ".md"), "w", encoding="utf-8").write(md + "\n")
