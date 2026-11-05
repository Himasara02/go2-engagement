#!/usr/bin/env python3
"""Task 8b + 9: evaluate monocular distance and gaze (from task9_reperceive.py).

Gaze as a 4th fuzzy input (fuzzy.py itself is unchanged; the system is rebuilt
here from its own sets, rules and Karnik-Mendel code):
  - gaze sets LOW / HIGH = the same membership functions as facing (mf_facing),
    FOU half-width = sigma_gaze (k = 1), as for the other inputs;
  - two rules added to the nine: (face HIGH and gaze HIGH) -> 0.95,
    (face HIGH and gaze LOW) -> 0.40 (body toward the robot but eyes elsewhere).
  Fixed before looking at any result; no tuning.
Policies added: gaze_only (threshold on gaze, leave-one-take-out Youden J as in
task 4), type1_gaze, it2_gaze, and type1/it2 with monocular distance, sigma_d and
approach instead of depth (type1_mono, it2_mono).
Also: monocular vs depth distance on the calibration takes, and gaze per
orientation phase there.
Writes results/gaze_mono.{json,md}.
"""
import gzip
import json
import os

import numpy as np

from analysis import NAMES, RESULTS, fmt, rate, save, sustained
from engage.fuzzy import RULES, _km, mf_approach, mf_distance, mf_facing
from task4_baselines import ORDER, THETA_C, compute, frame_sets, labelled_eps, youden
from task5_stats import holm, mcnemar
from task6_analysis import P, STOP_OFF
from task8_distance import CAL, CAL_TAKES, MARKS, longest_block
from engage.methods import COMMAND_MAP
from engage.fuzzy import STOP_LABELS

REP = os.path.join(RESULTS, "reperceive")
RULES4 = RULES + [({"face": "HIGH", "gaze": "HIGH"}, 0.95), ({"face": "HIGH", "gaze": "LOW"}, 0.40)]
NAMES = {**NAMES, "facing_only": "Facing only", "distance_only": "Distance only", "logreg": "Logistic regression",
         "gaze_only": "Gaze only", "type1_gaze": "Type-1 + gaze", "it2_gaze": "IT2 + gaze",
         "type1_mono": "Type-1, monocular distance", "it2_mono": "IT2, monocular distance"}


def fls(x, it2=True, rules=RULES4):
    z = 1.0 if it2 else 0.0
    sets = {"dist": mf_distance(x["distance"], z * x["distance_sigma"]),
            "face": mf_facing(x["facing"], z * x["facing_sigma"]),
            "appr": mf_approach(x["approach"], z * 0.08),
            "gaze": mf_facing(x.get("gaze", 0.0), z * x.get("gaze_sigma", 0.4))}
    y, fl, fu = [], [], []
    for ante, cons in rules:
        y.append(cons)
        fl.append(min(sets[v][lab][0] for v, lab in ante.items()))
        fu.append(min(sets[v][lab][1] for v, lab in ante.items()))
    if not it2:
        s = sum(fu)
        return sum(c * w for c, w in zip(y, fu)) / s if s > 1e-9 else 0.0
    return _km(y, fl, fu)[0]


def load_rep(name):
    with gzip.open(os.path.join(REP, name + ".jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]


def attach(loaded):
    """Add gaze / mono fields to every logged person. Unmatched: gaze 0 / sigma 0.4,
    mono = logged depth values (counted)."""
    cnt = {"people": 0, "matched": 0, "face": 0}
    for meta, labels, recs, eps in loaded:
        rep = load_rep(meta["take"])
        for r, q in zip(recs, rep):
            for p in r["people"]:
                cnt["people"] += 1
                v = q["people"].get(str(p["tid"]), {})
                if v.get("matched"):
                    cnt["matched"] += 1
                    cnt["face"] += v["face"]
                    p.update(gaze=v["gaze"], gaze_sigma=v["gaze_sd"], mono_d=v["mono_d"],
                             mono_sd=v["mono_sd"], mono_appr=v["mono_approach"])
                else:
                    p.update(gaze=0.0, gaze_sigma=0.4, mono_d=p["distance"], mono_sd=p["distance_sigma"],
                             mono_appr=p["approach"])
    return cnt


def gate(p, e):
    if not (p.get("gesture") in COMMAND_MAP and p["gesture_conf"] >= THETA_C):
        return False
    th = P.theta_e_motion - (STOP_OFF if p["gesture"] in STOP_LABELS else 0.0)
    return e >= th


def policy_sets(recs, efun):
    return sustained([{p["tid"] for p in r["people"] if gate(p, efun(p))} for r in recs])


def mono_x(p):
    return {"distance": p["mono_d"], "distance_sigma": p["mono_sd"], "facing": p["facing"],
            "facing_sigma": p["facing_sigma"], "approach": p["mono_appr"]}


def calibration():
    """Per mark (clean depth block): depth vs monocular bias / sd; gaze per phase."""
    out = {}
    for t in CAL_TAKES:
        recs = [json.loads(l) for l in open(os.path.join(CAL, t, "log.jsonl"))]
        rep = {q["i"]: q for q in load_rep(t)}
        out[t] = {}
        for k in MARKS:
            pts = longest_block([(r["t"], (r["i"], p)) for r in recs for p in r["people"]
                                 if abs(p["distance"] - k) < 0.5])
            dd, mm, gz, face, ts = [], [], [], [], []
            for tt, (i, p) in pts:
                v = rep.get(i, {"people": {}})["people"].get(str(p["tid"]), {})
                if v.get("matched"):
                    dd.append(p["distance"])
                    mm.append(v["mono_d"])
                    gz.append(v["gaze"])
                    face.append(v["face"])
                    ts.append(tt)
            if not dd:
                out[t][k] = None
                continue
            dd, mm, gz, ts = map(np.array, (dd, mm, gz, ts))
            third = [(ts >= ts.min() + j * np.ptp(ts) / 3) & (ts <= ts.min() + (j + 1) * np.ptp(ts) / 3)
                     for j in range(3)]
            out[t][k] = {"n": int(len(dd)), "depth_bias": float(np.median(dd - k)), "depth_sd": float(dd.std()),
                         "mono_bias": float(np.median(mm - k)), "mono_sd": float(mm.std()),
                         "gaze_phase": [float(np.median(gz[m])) if m.any() else None for m in third],
                         "face_rate_phase": [float(np.mean(np.array(face)[m])) if m.any() else None for m in third]}
    return out


def main():
    eps0, missed, fitted, names = compute()           # task 4 policies, held-out baselines
    from analysis import load_all, scenario_takes, episode_records
    loaded = load_all(scenario_takes())
    cnt = attach(loaded)
    by = {m["take"]: (m, lab, recs, eps) for m, lab, recs, eps in loaded}
    # gaze-only: leave-one-take-out threshold, same as the task 4 single-cue baselines
    cands = {n: [[(p["tid"], p.get("gesture") in COMMAND_MAP and p["gesture_conf"] >= THETA_C, p)
                  for p in r["people"]] for r in by[n][2]] for n in names}
    leps = {n: labelled_eps(by[n][1], by[n][3]) for n in names}
    score = lambda p: p["gaze"]
    gaze_tau, gaze_sets = {}, {}
    for test in names:
        tau = youden([(cands[n], leps[n]) for n in names if n != test], score, np.round(np.arange(0, 1.001, 0.02), 2))
        gaze_tau[test] = float(tau)
        gaze_sets[test] = frame_sets(cands[test], score, tau)
    # task 4 held-out baselines again, aligned to the same loaded records
    old = {(e["take"], e["start"], e["tid"]): e["acc"] for e in eps0}

    def extra(recs, take):
        return {"gaze_only": gaze_sets[take],
                "type1_gaze": policy_sets(recs, lambda p: fls(p, it2=False)),
                "it2_gaze": policy_sets(recs, lambda p: fls(p)),
                "type1_mono": policy_sets(recs, lambda p: fls(mono_x(p), it2=False, rules=RULES)),
                "it2_mono": policy_sets(recs, lambda p: fls(mono_x(p), rules=RULES))}
    eps, _ = episode_records(None, loaded=loaded, extra=extra)
    for e in eps:
        e["acc"].update({k: v for k, v in old[(e["take"], e["start"], e["tid"])].items() if k not in e["acc"]})
    order = ORDER + ["gaze_only", "type1_gaze", "it2_gaze", "type1_mono", "it2_mono"]
    n = [e for e in eps if e["label"] == "n"]
    r = [e for e in eps if e["label"] == "r"]
    m_tot = sum(missed.values())
    res = {"match": cnt, "gaze_tau": gaze_tau, "policies": {}, "mcnemar": {}, "by_scenario": {}}
    L = ["# Gaze and monocular distance (tasks 8b, 9)", "",
         f"Re-run matched {cnt['matched']}/{cnt['people']} logged person-frames "
         f"({100 * cnt['matched'] / cnt['people']:.1f}%); a face (with iris) was found in {cnt['face']} of the matched "
         f"({100 * cnt['face'] / max(cnt['matched'], 1):.1f}%).", "",
         "| Policy | Bystander false triggers | Recall (clips) | Recall (incl. missed) |", "|---|---|---|---|"]
    for m in order:
        ft = rate(sum(e["acc"][m] for e in n), len(n))
        rc = rate(sum(e["acc"][m] for e in r), len(r))
        res["policies"][m] = {"false_trigger": ft, "recall_clips": rc, "recall_incl_missed": rate(rc["k"], len(r) + m_tot)}
        L.append(f"| {NAMES[m]} | {fmt(ft, True)} | {fmt(rc, True)} | {fmt(rate(rc['k'], len(r) + m_tot))} |")
    pairs = [("it2_gaze", "it2"), ("type1_gaze", "type1"), ("it2_gaze", "facing_only"), ("it2_gaze", "gaze_only"),
             ("it2_mono", "it2"), ("type1_mono", "type1")]
    L += ["", "Exact McNemar on paired episodes (Holm over the 6 pairs, per label):", "",
          "| A vs B | Bystander: only A / only B | p (Holm) | Robot: only A / only B | p (Holm) |", "|---|---|---|---|---|"]
    for lab, g in (("n", n), ("r", r)):
        t = {f"{a}|{b}": mcnemar(g, a, b) for a, b in pairs}
        for k, p in zip(t, holm([v["p"] for v in t.values()])):
            t[k]["p_holm"] = p
        res["mcnemar"][lab] = t
    for a, b in pairs:
        x, y = res["mcnemar"]["n"][f"{a}|{b}"], res["mcnemar"]["r"][f"{a}|{b}"]
        L.append(f"| {NAMES[a]} vs {NAMES[b]} | {x['only_' + a]} / {x['only_' + b]} | {x['p_holm']:.3g} | "
                 f"{y['only_' + a]} / {y['only_' + b]} | {y['p_holm']:.3g} |")
    L += ["", "By scenario (bystander false triggers / recall):", "",
          "| Scenario | IT2 | IT2 + gaze | Facing only | Gaze only |", "|---|---|---|---|---|"]
    pc = lambda q: f"{100 * q['p']:.0f}% ({q['k']}/{q['n']})" if q["n"] else "–"
    for sc in ("S1", "S2", "S3", "S4", "S5", "S6"):
        row, res["by_scenario"][sc] = [], {}
        for m in ("it2", "it2_gaze", "facing_only", "gaze_only"):
            a = rate(sum(e["acc"][m] for e in n if e["scenario"] == sc), sum(e["scenario"] == sc for e in n))
            b = rate(sum(e["acc"][m] for e in r if e["scenario"] == sc), sum(e["scenario"] == sc for e in r))
            res["by_scenario"][sc][m] = [a, b]
            row.append(f"{pc(a)} / {pc(b)}")
        L.append(f"| {sc} | " + " | ".join(row) + " |")
    L += ["", f"Gaze-only threshold per fold: median {np.median(list(gaze_tau.values())):.2f} "
          f"(range {min(gaze_tau.values()):.2f}–{max(gaze_tau.values()):.2f}).", ""]

    cal = calibration()
    res["calibration"] = cal
    L += ["## Calibration takes: depth vs monocular distance, gaze by phase", "",
          "| Take | Mark | Frames | Depth bias | Depth SD | Mono bias | Mono SD | Gaze by phase | Face found by phase |",
          "|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda v: " / ".join("–" if a is None else f"{a:.2f}" for a in v)
    for t, v in cal.items():
        for k in MARKS:
            a = v[k]
            if a:
                L.append(f"| {t} | {k} m | {a['n']} | {a['depth_bias']:+.2f} | {a['depth_sd']:.2f} | {a['mono_bias']:+.2f} | "
                         f"{a['mono_sd']:.2f} | {f2(a['gaze_phase'])} | {f2(a['face_rate_phase'])} |")
    L += ["", "Pooled (median over takes):", "", "| Mark | Depth bias | Depth SD | Mono bias | Mono SD |", "|---|---|---|---|---|"]
    for k in MARKS:
        vs = [v[k] for v in cal.values() if v[k]]
        if vs:
            md = lambda key: float(np.median([a[key] for a in vs]))
            L.append(f"| {k} m | {md('depth_bias'):+.2f} | {md('depth_sd'):.2f} | {md('mono_bias'):+.2f} | {md('mono_sd'):.2f} |")
    save("gaze_mono", res, "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
