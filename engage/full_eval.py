#!/usr/bin/env python3
"""Full-dataset evaluation (60 standing takes received 10-11 Nov 2026), with the pilot's definitions.

    ~/miniconda3/envs/go2/bin/python full_eval.py

Inputs (read only):
  data/dataset_2026-11-11/takes_crop3/<take>/log.jsonl     gesture + confidence per person-frame
  data/depthpack_2026-11-11/.../labels_final.csv          one row per clip: take, clip, tid, frames, label
  data/cues_2026-11-10/<take>.jsonl.gz                     facing, gaze, monocular distance (+ sigma)
  data/clip_distance_2026-11-11/<take>.json                depth distance, sigma_d, approach (gesturer only)
Only the gesturing person's frames inside each clip are evaluated. Distance source (env DIST_SRC):
  clip  = data/clip_distance_2026-11-11 (processed depth, gesturer only, approach reset at the clip start)
  track = data/person_distance_2026-11-12 (full depth, every person, whole-track history as the live system)
"Nearest person" always uses data/person_distance_2026-11-12 (full depth received 12 Nov 2026): a frame
counts if the gesturer's distance is the smallest of everyone tracked in that frame (methods.py).
Output folder: env EVAL_OUT (default engage/results/full).

Definitions copied from the pilot scripts (no tuning, nothing changed in fuzzy.py / methods.py):
  candidate frame   = command gesture (methods.COMMAND_MAP) with confidence >= theta_c 0.45
  fuzzy gates       = fuzzy.py sets / rules / Karnik-Mendel (9 rules; +2 gaze rules from task9_eval.py);
                      accept if E (type-1) or E_low (IT2) >= theta_E, 0.60 motion / 0.45 STOP
  threshold / LR    = task4_baselines.py: score >= tau, tau by Youden J (recall - FT) on the training takes,
                      leave-one-recording-out; LR on distance, approach, facing, sigma_d, sigma_f
  episode accepted  = the policy accepted that person on >= 3 consecutive frames inside the clip
  statistics        = Wilson 95 % CI; exact McNemar on paired episodes (Holm); bootstrap over recordings
  analyses          = task6_analysis.py: theta_E sweep, sigma split at the median, FOU scale k,
                      ablations, noise injection (20 seeds, IT2 sigma' = sqrt(sigma^2 + s^2))
Writes engage/results/full/eval.json and eval.md.
"""
import csv
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engage.fuzzy import RULES, STOP_LABELS, DecisionParams, _km, mf_approach, mf_distance, mf_facing
from engage.methods import COMMAND_MAP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TAKES = os.environ.get("TAKES", os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3"))
LABELS = os.environ.get("LABELS", os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10",
                                                "labels_final.csv"))   # annotator A (paper); env LABELS for C / agreed
CUES = os.environ.get("CUES", os.path.join(ROOT, "data", "cues_2026-11-10"))
DIST = os.path.join(ROOT, "data", "clip_distance_2026-11-11")
PDIST = os.environ.get("PDIST", os.path.join(ROOT, "data", "person_distance_2026-11-12"))
DIST_SRC = os.environ.get("DIST_SRC", "track")   # paper (from 12 Nov): track
OUT = os.environ.get("EVAL_OUT", os.path.join(HERE, "results", "full"))
P = DecisionParams()
OFF = P.theta_e_motion - P.theta_e_stop          # 0.15
RULES4 = RULES + [({"face": "HIGH", "gaze": "HIGH"}, 0.95), ({"face": "HIGH", "gaze": "LOW"}, 0.40)]
FEATS = ["distance", "approach", "facing", "distance_sigma", "facing_sigma"]
SEEDS, BOOT = 20, 10000
NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person", "distance_only": "Distance only", "facing_only": "Facing only",
         "gaze_only": "Gaze only", "logreg": "Logistic regression", "type1": "Type-1 fuzzy",
         "type1_gaze": "Type-1 + gaze", "type1_mono": "Type-1, mono distance", "it2": "IT2 fuzzy",
         "it2_gaze": "IT2 + gaze", "it2_mono": "IT2, mono distance"}


# ------------------------------------------------------------------ engagement (as task6 / task9)
def fls(x, it2=True, k=1.0, rules=RULES, drop=()):
    z = k if it2 else 0.0
    sets = {"dist": mf_distance(x[0], z * x[1]), "face": mf_facing(x[2], z * x[3]),
            "appr": mf_approach(x[4], z * 0.08), "gaze": mf_facing(x[5], z * x[6])}
    y, fl, fu = [], [], []
    for ante, cons in rules:
        a = {v: lab for v, lab in ante.items() if v not in drop}
        if not a:
            continue
        y.append(cons)
        fl.append(min(sets[v][lab][0] for v, lab in a.items()))
        fu.append(min(sets[v][lab][1] for v, lab in a.items()))
    if not it2:
        s = sum(fu)
        return sum(c * w for c, w in zip(y, fu)) / s if s > 1e-9 else 0.0
    return _km(y, fl, fu)[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return {"k": 0, "n": 0, "p": float("nan"), "lo": float("nan"), "hi": float("nan")}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "p": p, "lo": max(0, c - h), "hi": min(1, c + h)}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, labels=None, takes=None, cues=None, pdist=None):
        LABELS_, TAKES_, CUES_, PDIST_ = labels or LABELS, takes or TAKES, cues or CUES, pdist or PDIST
        rows = [r for r in csv.DictReader(open(LABELS_)) if r["label"] in ("r", "n")]
        self.eps, pf = [], []
        self.missing = defaultdict(int)
        by_take = defaultdict(list)
        for r in rows:
            by_take[r["take"]].append(r)
        need = (lambda t: os.path.join(PDIST_, t + ".json.gz")) if DIST_SRC == "track" else (lambda t: os.path.join(DIST, t + ".json"))
        self.skipped = [t for t in sorted(by_take) if not os.path.exists(need(t))]
        for take in sorted(by_take):
            if take in self.skipped:
                continue
            recs = [json.loads(l) for l in open(os.path.join(TAKES_, take, "log.jsonl"))]
            cues = [json.loads(l) for l in gzip.open(os.path.join(CUES_, take + ".jsonl.gz"), "rt")]
            dist = json.load(open(os.path.join(DIST, take + ".json"))) if DIST_SRC != "track" else {}
            pdf = os.path.join(PDIST_, take + ".json.gz")
            pd = json.load(gzip.open(pdf, "rt")) if os.path.exists(pdf) else {}
            for r in by_take[take]:
                tid, c = int(r["track_id"]), r["clip"]
                idx, mono_prev, mono_v = [], None, 0.0
                for i in range(int(r["first_frame"]), int(r["last_frame"]) + 1):
                    p = next((q for q in recs[i]["people"] if q["tid"] == tid), None)
                    cu = cues[i]["people"].get(str(tid), {}) if i < len(cues) else {}
                    allp = pd.get(str(i), {})
                    me = allp.get(str(tid))
                    if DIST_SRC == "track":
                        dd = None if me is None else {"d": me[0], "sd": me[1], "v": me[2], "depth": me[4]}
                    else:
                        dd = dist.get(c, {}).get(str(i))
                    if p is None or not cu.get("matched") or dd is None:
                        self.missing["no person / cues / depth"] += 1
                        idx.append(None)
                        continue
                    t = recs[i]["t"]
                    md = min(cu["mono_d"], 10.0)
                    if mono_prev is not None and t > mono_prev[1]:
                        mono_v = 0.8 * mono_v + 0.2 * float(np.clip(-(md - mono_prev[0]) / (t - mono_prev[1]), -2, 2))
                    mono_prev = (md, t)
                    g = p.get("gesture")
                    pf.append({"cand": g in COMMAND_MAP and (p.get("gesture_conf") or 0) >= P.theta_conf,
                               "stop": g in STOP_LABELS, "distance": dd["d"], "distance_sigma": dd["sd"],
                               "approach": dd["v"], "depth_ok": dd["depth"], "facing": cu["facing"],
                               "facing_sigma": cu["facing_sd"], "gaze": cu["gaze"], "gaze_sigma": cu["gaze_sd"],
                               "nearest": me is not None and me[0] <= min(q[0] for q in allp.values()),
                               "face": cu["face"], "mono_d": md, "mono_sd": cu["mono_sd"], "mono_v": mono_v})
                    idx.append(len(pf) - 1)
                self.eps.append({"take": take, "scenario": take.split("_")[1], "day": take.split("_")[0],
                                 "clip": int(c), "label": r["label"], "source": r["label_source"], "idx": idx})
        self.pf = pf
        self.n = len(pf)
        self.cand = np.array([p["cand"] for p in pf])
        self.stop = np.array([p["stop"] for p in pf])
        self.nearest = np.array([p["nearest"] for p in pf])
        self.x = {k: np.array([p[k] for p in pf], float) for k in
                  ("distance", "distance_sigma", "approach", "facing", "facing_sigma", "gaze", "gaze_sigma",
                   "mono_d", "mono_sd", "mono_v")}
        self.takes = sorted({e["take"] for e in self.eps})
        self.lab = np.array([e["label"] == "r" for e in self.eps])
        self.take_of = np.array([self.takes.index(e["take"]) for e in self.eps])

    def X(self, dist="distance", sd="distance_sigma", v="approach", x=None):
        x = x or self.x
        return np.stack([x[dist], x[sd], x["facing"], x["facing_sigma"], x[v], x["gaze"], x["gaze_sigma"]], 1)

    def engage(self, it2=True, k=1.0, rules=RULES, drop=(), X=None):
        X = self.X() if X is None else X
        return np.array([fls(row, it2, k, rules, drop) for row in X])

    def best_window(self, u):
        """Per episode: max over 3 consecutive candidate frames of min(u) (-inf if none).
        Accepted at threshold th  <=>  value >= th  (the sustain rule)."""
        out = np.full(len(self.eps), -np.inf)
        for j, e in enumerate(self.eps):
            idx = e["idx"]
            for a in range(len(idx) - 2):
                w = idx[a:a + 3]
                if None in w or not all(self.cand[i] for i in w):
                    continue
                out[j] = max(out[j], min(u[i] for i in w))
        return out

    def gate_value(self, e):
        """Fuzzy gates: STOP needs 0.15 less, so test e + 0.15*stop against theta_motion."""
        return self.best_window(e + OFF * self.stop)


def rates(acc, sel=None, lab=None, eps_lab=None):
    sel = np.ones(len(acc), bool) if sel is None else sel
    L = eps_lab
    return wilson(int((acc & sel & ~L).sum()), int((sel & ~L).sum())), wilson(int((acc & sel & L).sum()), int((sel & L).sum()))


def youden(vals, lab, train, grid):
    best = (-2, None)
    for tau in grid:
        a = vals >= tau
        r = a[train & lab].mean() if (train & lab).any() else 0
        n = a[train & ~lab].mean() if (train & ~lab).any() else 0
        if r - n > best[0]:
            best = (r - n, float(tau))
    return best[1]


def main():
    d = Data()
    L = d.lab
    print(f"episodes {len(d.eps)} (r {L.sum()}, n {(~L).sum()}), person-frames {d.n}, missing {dict(d.missing)}")
    X0 = d.X()
    Xm = d.X("mono_d", "mono_sd", "mono_v")
    E = {"type1": d.engage(False), "it2": d.engage(True),
         "type1_gaze": d.engage(False, rules=RULES4), "it2_gaze": d.engage(True, rules=RULES4),
         "type1_mono": d.engage(False, X=Xm), "it2_mono": d.engage(True, X=Xm)}
    V = {m: d.gate_value(e) for m, e in E.items()}
    acc = {"gesture_only": d.best_window(np.zeros(d.n)) >= 0,
           "nearest": d.best_window(d.nearest.astype(float)) >= 1}
    acc.update({m: v >= P.theta_e_motion for m, v in V.items()})

    # leave-one-recording-out baselines (task4)
    single = {"facing_only": (d.x["facing"], np.round(np.arange(0, 1.001, 0.02), 2)),
              "gaze_only": (d.x["gaze"], np.round(np.arange(0, 1.001, 0.02), 2)),
              "distance_only": (-d.x["distance"], -np.round(np.arange(0.5, 6.01, 0.1), 2))}
    SV = {m: d.best_window(s) for m, (s, _) in single.items()}
    fitted = {m: {} for m in list(single) + ["logreg"]}
    for m in single:
        acc[m] = np.zeros(len(d.eps), bool)
    acc["logreg"] = np.zeros(len(d.eps), bool)
    F = np.stack([d.x[k] for k in FEATS], 1)
    ep_rows = [[i for i in e["idx"] if i is not None] for e in d.eps]
    coefs = []
    if os.environ.get("FIT_ON") == "standing":
        # walking evaluation: thresholds and regression fitted once on all 60 standing recordings (no walking data)
        ds = Data(labels=os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10", "labels_final.csv"),
                  takes=os.path.join(ROOT, "data", "dataset_2026-11-11", "takes_crop3"),
                  cues=os.path.join(ROOT, "data", "cues_2026-11-10"),
                  pdist=os.path.join(ROOT, "data", "person_distance_2026-11-12"))
        Ls, alls = ds.lab, np.ones(len(ds.lab), bool)
        sv_s = {"facing_only": ds.best_window(ds.x["facing"]), "gaze_only": ds.best_window(ds.x["gaze"]),
                "distance_only": ds.best_window(-ds.x["distance"])}
        for m, (_, grid) in single.items():
            tau = youden(sv_s[m], Ls, alls, grid)
            fitted[m]["standing"] = tau
            acc[m] = SV[m] >= tau
        Fs = np.stack([ds.x[k] for k in FEATS], 1)
        rows_s = [i for e in ds.eps for i in e["idx"] if i is not None]
        y_s = [l for e, l in zip(ds.eps, Ls) for i in e["idx"] if i is not None]
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        clf.fit(Fs[rows_s], np.array(y_s))
        tau = youden(ds.best_window(clf.predict_proba(Fs)[:, 1]), Ls, alls, np.round(np.arange(0.05, 0.96, 0.025), 3))
        acc["logreg"] = d.best_window(clf.predict_proba(F)[:, 1]) >= tau
        coefs.append(clf[-1].coef_[0])
        fitted["logreg"]["standing"] = {"tau": tau, "coef": dict(zip(FEATS, map(float, clf[-1].coef_[0])))}
    for ti, take in enumerate(d.takes if os.environ.get("FIT_ON") != "standing" else []):
        test, train = d.take_of == ti, d.take_of != ti
        for m, (_, grid) in single.items():
            tau = youden(SV[m], L, train, grid)
            fitted[m][take] = tau
            acc[m][test] = SV[m][test] >= tau
        rows = [i for j in np.where(train)[0] for i in ep_rows[j]]
        y = [L[j] for j in np.where(train)[0] for _ in ep_rows[j]]
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        clf.fit(F[rows], np.array(y))
        prob = clf.predict_proba(F)[:, 1]
        lv = d.best_window(prob)
        tau = youden(lv, L, train, np.round(np.arange(0.05, 0.96, 0.025), 3))
        acc["logreg"][test] = lv[test] >= tau
        coefs.append(clf[-1].coef_[0])
        fitted["logreg"][take] = {"tau": tau, "coef": dict(zip(FEATS, map(float, clf[-1].coef_[0])))}

    order = ["gesture_only", "nearest", "distance_only", "facing_only", "gaze_only", "logreg", "type1", "type1_gaze",
             "type1_mono", "it2", "it2_gaze", "it2_mono"]
    res = {"dist_src": DIST_SRC, "episodes": {"r": int(L.sum()), "n": int((~L).sum())}, "person_frames": d.n,
           "missing_frames": dict(d.missing), "policies": {}, "fitted": fitted,
           "logreg_coef_median": dict(zip(FEATS, map(float, np.median(coefs, 0))))}
    for m in order:
        ft, rc = rates(acc[m], eps_lab=L)
        res["policies"][m] = {"false_trigger": ft, "recall": rc}

    # single-cue thresholds without training data (fixed guesses, applied to every recording)
    res["untuned"] = {m: {f"{t:.2f}": dict(zip(("false_trigger", "recall"), rates(SV[m] >= t, eps_lab=L)))
                          for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)} for m in ("facing_only", "gaze_only")}
    res["fitted_tau_median"] = {m: float(np.median(list(fitted[m].values()))) for m in ("facing_only", "gaze_only")}
    res["nearest_frames"] = {lab: float(d.nearest[[i for e, l in zip(d.eps, L) if l == (lab == "r")
                                                     for i in e["idx"] if i is not None and d.cand[i]]].mean())
                             for lab in ("r", "n")}

    # by scenario / label source / day
    groups = {"scenario": sorted({e["scenario"] for e in d.eps}), "source": ["labels.json", "labels_cmd.json"],
              "day": sorted({e["day"] for e in d.eps})}
    res["by"] = {}
    for g, vals in groups.items():
        res["by"][g] = {}
        for v in vals:
            sel = np.array([e[g] == v for e in d.eps])
            res["by"][g][v] = {m: dict(zip(("false_trigger", "recall"), rates(acc[m], sel, eps_lab=L)))
                               for m in ("gesture_only", "nearest", "facing_only", "gaze_only", "type1", "it2", "it2_gaze")}

    # McNemar + bootstrap: IT2 (and IT2 + gaze) vs every other policy
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(d.takes), (BOOT, len(d.takes)))
    cnt_t = np.bincount(d.take_of, minlength=len(d.takes))

    def per_take(a, lab_mask):
        return np.bincount(d.take_of[lab_mask], weights=a[lab_mask].astype(float), minlength=len(d.takes))
    nn_t, nr_t = per_take(np.ones(len(L), bool), ~L), per_take(np.ones(len(L), bool), L)
    res["stats"] = {}
    for ref in ("it2", "it2_gaze"):
        rows, ps = {}, []
        for m in order:
            if m == ref:
                continue
            row = {}
            for lab, mask in (("n", ~L), ("r", L)):
                b = int((acc[ref] & ~acc[m] & mask).sum())
                c = int((~acc[ref] & acc[m] & mask).sum())
                row[lab] = {"only_" + ref: b, "only_other": c, "p": mcnemar(b, c)}
                ps.append(row[lab]["p"])
            dn = per_take(acc[ref], ~L) - per_take(acc[m], ~L)
            dr = per_take(acc[ref], L) - per_take(acc[m], L)
            bft = dn[boots].sum(1) / np.maximum(nn_t[boots].sum(1), 1)
            brc = dr[boots].sum(1) / np.maximum(nr_t[boots].sum(1), 1)
            row["boot_ft_pts"] = [float(100 * np.percentile(bft, 2.5)), float(100 * np.percentile(bft, 97.5))]
            row["boot_recall_pts"] = [float(100 * np.percentile(brc, 2.5)), float(100 * np.percentile(brc, 97.5))]
            rows[m] = row
        hp = holm(ps)
        k = 0
        for m in rows:
            for lab in ("n", "r"):
                rows[m][lab]["p_holm"] = hp[k]
                k += 1
        res["stats"][ref] = rows

    # theta_E sweep + recall at FT budgets
    grid = np.round(np.arange(0, 1.0001, 0.02), 2)
    sweep = {}
    for m, v in [(m, V[m]) for m in ("type1", "it2", "type1_gaze", "it2_gaze")] + [("facing_only", SV["facing_only"]),
                                                                                   ("gaze_only", SV["gaze_only"])]:
        sweep[m] = [[float(t), rates(v >= t, eps_lab=L)[0]["p"], rates(v >= t, eps_lab=L)[1]["p"]] for t in grid]
    res["sweep"] = {"grid": grid.tolist(), "curves": sweep,
                    "recall_at_ft": {m: {f: max((rc for _, ft, rc in c if ft <= f + 1e-9), default=0.0)
                                         for f in (0.02, 0.05, 0.10)} for m, c in sweep.items()}}

    # sigma splits at the median of the episode-mean sigma; distance bins
    em = {k: np.array([np.mean([d.x[k][i] for i in e["idx"] if i is not None]) if any(i is not None for i in e["idx"])
                       else np.nan for e in d.eps]) for k in ("facing_sigma", "distance_sigma", "distance", "gaze_sigma")}
    med = {k: float(np.nanmedian(em[k])) for k in ("facing_sigma", "distance_sigma", "gaze_sigma")}
    bins = {"facing σ": [("low", em["facing_sigma"] <= med["facing_sigma"]), ("high", em["facing_sigma"] > med["facing_sigma"])],
            "distance σ": [("low", em["distance_sigma"] <= med["distance_sigma"]), ("high", em["distance_sigma"] > med["distance_sigma"])],
            "gaze σ": [("low", em["gaze_sigma"] <= med["gaze_sigma"]), ("high", em["gaze_sigma"] > med["gaze_sigma"])],
            "distance": [("< 2 m", em["distance"] < 2), ("2–3 m", (em["distance"] >= 2) & (em["distance"] < 3)),
                         ("≥ 3 m", em["distance"] >= 3)]}
    res["splits"] = {"medians": med, "bins": {b: {n: {m: dict(zip(("false_trigger", "recall"), rates(acc[m], s, eps_lab=L)))
                                                      for m in ("type1", "it2", "type1_gaze", "it2_gaze", "facing_only")}
                                                  for n, s in bl} for b, bl in bins.items()}}
    res["episode_distance"] = {lab: [float(np.nanpercentile(em["distance"][mask], q)) for q in (5, 25, 50, 75, 95)]
                               for lab, mask in (("r", L), ("n", ~L))}

    # FOU scale k and ablations
    res["fou_k"] = {str(k): dict(zip(("false_trigger", "recall"),
                                     rates((d.gate_value(d.engage(True, k)) if k else V["type1"]) >= P.theta_e_motion, eps_lab=L)))
                    for k in (0, 0.5, 1, 2)}
    res["ablations"] = {}
    for name, drop in (("no approach", ("appr",)), ("facing only", ("dist", "appr")), ("distance only", ("face", "appr"))):
        res["ablations"][name] = {m: dict(zip(("false_trigger", "recall"),
                                              rates(d.gate_value(d.engage(m == "it2", drop=drop)) >= P.theta_e_motion, eps_lab=L)))
                                  for m in ("type1", "it2")}

    # noise injection (task6)
    fo_tau = np.array([fitted["facing_only"].get(d.takes[t], fitted["facing_only"].get("standing")) for t in d.take_of])
    levels = {"facing": [0, 0.1, 0.2, 0.3], "distance": [0, 0.3, 0.6, 1.0]}
    noise = {}
    for var, ss in levels.items():
        noise[var] = {}
        for s in ss:
            accn = defaultdict(list)
            for seed in range(SEEDS if s else 1):
                r_ = np.random.default_rng(seed)
                x = dict(d.x)
                v = d.x[var] + r_.normal(0, s, d.n)
                x[var] = np.clip(v, 0, 1) if var == "facing" else np.clip(v, 0.3, None)
                xs = dict(x)
                xs[var + "_sigma"] = np.sqrt(d.x[var + "_sigma"] ** 2 + s ** 2)
                for m, a in (("type1", d.gate_value(d.engage(False, X=d.X(x=x))) >= P.theta_e_motion),
                             ("it2", d.gate_value(d.engage(True, X=d.X(x=xs))) >= P.theta_e_motion),
                             ("it2_fixed_sigma", d.gate_value(d.engage(True, X=d.X(x=x))) >= P.theta_e_motion),
                             ("facing_only", d.best_window(x["facing"]) >= fo_tau)):
                    ft, rc = rates(a, eps_lab=L)
                    accn[m].append([ft["p"], rc["p"]])
            noise[var][str(s)] = {m: {"ft_mean": float(np.mean([a[0] for a in v])), "ft_sd": float(np.std([a[0] for a in v])),
                                      "recall_mean": float(np.mean([a[1] for a in v])),
                                      "recall_sd": float(np.std([a[1] for a in v]))} for m, v in accn.items()}
        print("noise", var, "done", flush=True)
    res["noise"] = {"seeds": SEEDS, "levels": levels, "results": noise}

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    pc = lambda r: f"{100 * r['p']:.1f}% ({r['k']}/{r['n']}) [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}]"
    md = ["# Full dataset evaluation (60 standing recordings)", "",
          f"Episodes: {L.sum()} r / {(~L).sum()} n; person-frames {d.n}; missing frames {dict(d.missing)}.", "",
          "| Policy | False triggers | Recall | Recall at FT ≤ 2 % |", "|---|---|---|---|"]
    for m in order:
        p = res["policies"][m]
        r2 = res["sweep"]["recall_at_ft"].get(m, {}).get(0.02)
        md.append(f"| {NAMES[m]} | {pc(p['false_trigger'])} | {pc(p['recall'])} | "
                  f"{'' if r2 is None else f'{100 * r2:.0f}%'} |")
    md += ["", "Median LR coefficients: " + ", ".join(f"{k} {v:+.2f}" for k, v in res["logreg_coef_median"].items()), ""]
    for ref, rows in res["stats"].items():
        md += [f"## {NAMES[ref]} vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)", "",
               "| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |", "|---|---|---|---|---|---|---|"]
        for m, row in rows.items():
            md.append(f"| {NAMES[m]} | {row['n']['only_' + ref]} / {row['n']['only_other']} | {row['n']['p_holm']:.3g} | "
                      f"{row['r']['only_' + ref]} / {row['r']['only_other']} | {row['r']['p_holm']:.3g} | "
                      f"{row['boot_ft_pts'][0]:+.1f} to {row['boot_ft_pts'][1]:+.1f} | "
                      f"{row['boot_recall_pts'][0]:+.1f} to {row['boot_recall_pts'][1]:+.1f} |")
        md.append("")
    md += ["## Splits (FT / recall %)", "", "| Split | Bin | Type-1 | IT2 | Type-1+gaze | IT2+gaze | Facing only |", "|---|---|---|---|---|---|---|"]
    q = lambda v: f"{100 * v['false_trigger']['p']:.0f} / {100 * v['recall']['p']:.0f}"
    for b, bl in res["splits"]["bins"].items():
        for n, v in bl.items():
            md.append(f"| {b} | {n} | " + " | ".join(q(v[m]) for m in ("type1", "it2", "type1_gaze", "it2_gaze", "facing_only")) + " |")
    md += ["", "## By scenario / label source / day (FT / recall %)", "",
           "| Group | Gesture only | Facing only | Gaze only | Type-1 | IT2 | IT2+gaze |", "|---|---|---|---|---|---|---|"]
    for g, vals in res["by"].items():
        for v, row in vals.items():
            md.append(f"| {g} {v} | " + " | ".join(q(row[m]) for m in ("gesture_only", "facing_only", "gaze_only", "type1", "it2", "it2_gaze")) + " |")
    md += ["", "## FOU scale k (FT / recall %)", "", " · ".join(f"k={k}: {q(v)}" for k, v in res["fou_k"].items()),
           "", "## Ablations (FT / recall %)", ""]
    for n, v in res["ablations"].items():
        md.append(f"- {n}: Type-1 {q(v['type1'])}, IT2 {q(v['it2'])}")
    md += ["", "## Noise (mean FT / recall %, 20 seeds)", ""]
    for var, lv in noise.items():
        for s, v in lv.items():
            md.append(f"- {var} s={s}: " + ", ".join(f"{m} {100 * a['ft_mean']:.1f} / {100 * a['recall_mean']:.1f}" for m, a in v.items()))
    md += ["", "Episode-mean depth distance (m) p5/25/50/75/95: " +
           "; ".join(f"{k}: " + ", ".join(f"{x:.2f}" for x in v) for k, v in res["episode_distance"].items())]
    open(os.path.join(OUT, "eval.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:20]))


if __name__ == "__main__":
    main()
