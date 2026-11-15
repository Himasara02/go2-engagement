#!/usr/bin/env python3
"""Two extra baselines from the literature, on the same episodes and rules as full_eval.py (nothing there changed).

  LOOK  (Belkada et al., BMVC 2021, github.com/vita-epfl/looking): pretrained MLP that says whether a person looks
        at the camera from 17 body keypoints (model LookingModel_LOOK+PIE, weights exported to look_weights.npz,
        forward pass re-implemented in numpy, input normalisation copied from utils_predict.normalize_by_image_).
        Keypoints: full_look_keypoints.py. Used zero-shot with the authors' default threshold 0.5 ("LOOK"), and,
        for completeness, with a threshold fitted like the single-cue baselines ("LOOK, fitted").
  Sequence model  in the spirit of intention-to-interact predictors (Arreghini et al. 2024; Abbate et al. 2023):
        a GRU over each episode's per-frame cues (distance, approach, facing, gaze and their sigmas), trained by
        full_seq_train.py (bench312 env, PyTorch). Leave-one-recording-out like the logistic regression; for the
        walking run (FIT_ON=standing) trained once on all standing recordings. Threshold: Youden J on training.
  Crisp rule  (Michalowski et al. 2006 style): inside Hall's social zone (d <= 3.6 m) and facing >= 0.5; no training.
  Random forest  on the same 7 cues (200 trees, leaf 20, balanced), LOO like the LR (computed, not in the paper).
Episode accepted = same 3-consecutive-candidate-frame rule (Data.best_window). Holm over the paper's family: IT2 + gaze
vs the 11 policies of full_eval + crisp rule -> stats_it2_gaze_with_extra (sequence model reported, not in the paper).

    ~/miniconda3/envs/go2/bin/python full_extra_baselines.py export      # writes seq features for the GRU
    ~/miniconda3/envs/bench312/bin/python -P full_seq_train.py           # writes seq probabilities
    ~/miniconda3/envs/go2/bin/python full_extra_baselines.py eval        # results/<EVAL_OUT>_extra/eval.{json,md}
Walking: same env as full_eval (TAKES, LABELS, CUES, PDIST, EVAL_OUT, FIT_ON=standing) + LK_OUT for keypoints.
"""
import csv
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

import full_eval as fe

HERE = os.path.dirname(os.path.abspath(__file__))
LK = os.environ.get("LK_OUT", os.path.join(fe.ROOT, "data", "look_kps_2026-11-13"))
OUT = fe.OUT.rstrip("/") + "_extra"
SEQ_FEATS = ["distance", "approach", "facing", "gaze", "distance_sigma", "facing_sigma", "gaze_sigma"]
STAND = dict(labels=os.path.join(fe.ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10", "labels_final.csv"),
             takes=os.path.join(fe.ROOT, "data", "dataset_2026-11-11", "takes_crop3"),
             cues=os.path.join(fe.ROOT, "data", "cues_2026-11-10"),
             pdist=os.path.join(fe.ROOT, "data", "person_distance_2026-11-12"))


def frames_of(d, labels):
    """(take, frame, tid) of every person-frame row of Data d, in d.pf order (rebuilt as Data.__init__ does)."""
    rows = [r for r in csv.DictReader(open(labels)) if r["label"] in ("r", "n")]
    by = defaultdict(list)
    for r in rows:
        by[r["take"]].append(r)
    out = [None] * d.n
    j = 0
    for take in sorted(by):
        if take in d.skipped:
            continue
        for r in by[take]:
            e = d.eps[j]
            assert e["take"] == take and e["clip"] == int(r["clip"])
            for k, i in enumerate(e["idx"]):
                if i is not None:
                    out[i] = (take, int(r["first_frame"]) + k, r["track_id"])
            j += 1
    assert j == len(d.eps) and None not in out
    return out


# ---------------- LOOK (numpy forward pass of LookingModel, eval mode) ----------------
def look_model():
    w = np.load(os.path.join(HERE, "look_weights.npz"))

    def bn(x, p):
        return (x - w[p + ".running_mean"]) / np.sqrt(w[p + ".running_var"] + 1e-5) * w[p + ".weight"] + w[p + ".bias"]

    def lin(x, p):
        return x @ w[p + ".weight"].T + w[p + ".bias"]

    def f(x):
        y = np.maximum(bn(lin(x, "w1"), "batch_norm1"), 0)
        for s in range(3):
            q = f"linear_stages.{s}"
            z = np.maximum(bn(lin(y, q + ".l1"), q + ".bn1"), 0)
            z = np.maximum(bn(lin(z, q + ".l2"), q + ".bn2"), 0)
            y = y + z
        return 1 / (1 + np.exp(-lin(y, "w2")[:, 0]))
    return f


def look_input(X, Y, C, W, H):
    """utils_predict.normalize_by_image_ (centre = mid-hips, x by image width + relative to centre / width)."""
    X, Y = np.array(X, float), np.array(Y, float)
    cx, cy = int((X[11] + X[12]) / 2), int((Y[11] + Y[12]) / 2)
    width, height = abs(X.max() - X.min()), abs(Y.max() - Y.min())
    Xn = X / W + (X - cx) / max(width, 1e-6)
    Yn = (Y - cy) / max(height, 1e-6)
    return np.concatenate([Xn, Yn, np.array(C, float)])


def look_scores(d, keys):
    f = look_model()
    cache, rows, miss = {}, [], 0
    for take, fr, tid in keys:
        if take not in cache:
            cache[take] = json.load(gzip.open(os.path.join(LK, take + ".json.gz"), "rt"))
        kp = cache[take]
        v = kp["frames"].get(str(fr), {}).get(tid)
        if v is None:
            miss += 1
            rows.append(None)
        else:
            rows.append(look_input(*v, kp["W"], kp["H"]))
    have = [i for i, r in enumerate(rows) if r is not None]
    s = np.zeros(d.n)
    s[have] = f(np.stack([rows[i] for i in have]))
    return s, miss


# ---------------- export for the sequence model ----------------
def export(d, name):
    F = np.stack([d.x[k] for k in SEQ_FEATS], 1).astype(np.float32)
    idx = [[-1 if i is None else i for i in e["idx"]] for e in d.eps]
    L = max(map(len, idx))
    I = np.full((len(idx), L), -1, np.int64)
    for j, r in enumerate(idx):
        I[j, :len(r)] = r
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"seq_{name}.npz"), F=F, I=I, lab=d.lab, take_of=d.take_of)
    print("exported", name, F.shape, I.shape)


def main():
    mode = sys.argv[1]
    walking = os.environ.get("FIT_ON") == "standing"
    d = fe.Data()
    if mode == "export":
        export(d, "test")
        if walking:
            export(fe.Data(**STAND), "train")
        return
    L = d.lab
    keys = frames_of(d, fe.LABELS)
    look, miss = look_scores(d, keys)
    acc, fitted = {}, {}
    lv = d.best_window(look)
    acc["look"] = lv >= 0.5
    # fitted threshold (Youden J on training recordings, as facing_only)
    grid = np.round(np.arange(0, 1.001, 0.02), 2)
    acc["look_fitted"] = np.zeros(len(d.eps), bool)
    seq = np.load(os.path.join(OUT, "seq_probs.npz"))
    acc["seq"] = np.zeros(len(d.eps), bool)
    sgrid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    if walking:
        ds = fe.Data(**STAND)
        ks = frames_of(ds, STAND["labels"])
        global LK
        lk_w, LK = LK, os.path.join(fe.ROOT, "data", "look_kps_2026-11-13")
        look_s, _ = look_scores(ds, ks)
        LK = lk_w
        alls = np.ones(len(ds.lab), bool)
        tau = fe.youden(ds.best_window(look_s), ds.lab, alls, grid)
        acc["look_fitted"] = lv >= tau
        fitted["look_fitted"] = {"standing": tau}
        tau = fe.youden(ds.best_window(seq["train"]), ds.lab, alls, sgrid)
        acc["seq"] = d.best_window(seq["test"]) >= tau
        fitted["seq"] = {"standing": tau}
    else:
        fitted = {"look_fitted": {}, "seq": {}}
        P = seq["loo"]                                    # (n_takes, n): fold t's model on every person-frame
        for ti, take in enumerate(d.takes):
            test, train = d.take_of == ti, d.take_of != ti
            tau = fe.youden(lv, L, train, grid)
            acc["look_fitted"][test] = lv[test] >= tau
            fitted["look_fitted"][take] = tau
            sv = d.best_window(P[ti])
            tau = fe.youden(sv, L, train, sgrid)
            acc["seq"][test] = sv[test] >= tau
            fitted["seq"][take] = tau
    # crisp proxemics + facing rule (Michalowski et al. 2006 style), fixed before results: inside Hall's social
    # zone (d <= 3.6 m) and facing >= 0.5, on 3 consecutive candidate frames. No training.
    acc["crisp"] = d.best_window(((d.x["distance"] <= 3.6) & (d.x["facing"] >= 0.5)).astype(float)) >= 1
    # random forest on the sequence model's 7 cues (per frame), threshold by Youden J on training, like the LR
    from sklearn.ensemble import RandomForestClassifier

    def forest():
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced",
                                      random_state=0, n_jobs=-1)
    Fx = np.stack([d.x[k] for k in SEQ_FEATS], 1)
    acc["forest"] = np.zeros(len(d.eps), bool)
    if walking:
        Fs = np.stack([ds.x[k] for k in SEQ_FEATS], 1)
        rows_s = [i for e in ds.eps for i in e["idx"] if i is not None]
        y_s = [l for e, l in zip(ds.eps, ds.lab) for i in e["idx"] if i is not None]
        clf = forest().fit(Fs[rows_s], y_s)
        tau = fe.youden(ds.best_window(clf.predict_proba(Fs)[:, 1]), ds.lab, alls, sgrid)
        acc["forest"] = d.best_window(clf.predict_proba(Fx)[:, 1]) >= tau
        fitted["forest"] = {"standing": tau}
    else:
        fitted["forest"] = {}
        ep_rows = [[i for i in e["idx"] if i is not None] for e in d.eps]
        for ti, take in enumerate(d.takes):
            test, train = d.take_of == ti, d.take_of != ti
            rows = [i for j in np.where(train)[0] for i in ep_rows[j]]
            y = [L[j] for j in np.where(train)[0] for _ in ep_rows[j]]
            clf = forest().fit(Fx[rows], y)
            fv = d.best_window(clf.predict_proba(Fx)[:, 1])
            tau = fe.youden(fv, L, train, sgrid)
            acc["forest"][test] = fv[test] >= tau
            fitted["forest"][take] = tau
    # reference: IT2 + gaze and the strongest earlier baselines, recomputed here for paired tests
    it2g = d.gate_value(d.engage(True, rules=fe.RULES4)) >= fe.P.theta_e_motion
    acc["it2_gaze"] = it2g
    res = {"episodes": {"r": int(L.sum()), "n": int((~L).sum())}, "look_missing_frames": miss,
           "fitted": fitted, "policies": {}, "mcnemar_vs_it2_gaze": {}}
    for m in ("look", "look_fitted", "seq", "crisp", "forest", "it2_gaze"):
        ft, rc = fe.rates(acc[m], eps_lab=L)
        res["policies"][m] = {"false_trigger": ft, "recall": rc}
    for m in ("look", "look_fitted", "seq", "crisp", "forest"):
        out = {}
        for part, sel in (("false_trigger", ~L), ("recall", L)):
            b = int((acc[m] & ~it2g & sel).sum())
            c = int((~acc[m] & it2g & sel).sum())
            out[part] = {"other_only": b, "it2_gaze_only": c, "p": fe.mcnemar(b, c)}
        res["mcnemar_vs_it2_gaze"][m] = out
    # paper family of tests: IT2 + gaze vs every policy in the main table = full_eval's 11 policies + the
    # crisp rule; Holm over all of them; bootstrap over recordings as full_eval
    base = json.load(open(os.path.join(fe.OUT, "eval.json")))["stats"]["it2_gaze"]
    rng = np.random.default_rng(0)
    nt = len(d.takes)
    boots = rng.integers(0, nt, (fe.BOOT, nt))

    def per_take(a, mask):
        return np.bincount(d.take_of[mask], weights=a[mask].astype(float), minlength=nt)
    nn_t, nr_t = per_take(np.ones(len(L), bool), ~L), per_take(np.ones(len(L), bool), L)
    merged = {m: {k: (dict(v) if isinstance(v, dict) else v) for k, v in row.items()} for m, row in base.items()}
    for m in ("crisp",):                                  # paper: only the proxemics rule added (user, 13 Nov)
        row = {}
        for lab, mask in (("n", ~L), ("r", L)):
            b = int((it2g & ~acc[m] & mask).sum())
            c = int((~it2g & acc[m] & mask).sum())
            row[lab] = {"only_it2_gaze": b, "only_other": c, "p": fe.mcnemar(b, c)}
        dn = per_take(it2g, ~L) - per_take(acc[m], ~L)
        dr = per_take(it2g, L) - per_take(acc[m], L)
        bft = dn[boots].sum(1) / np.maximum(nn_t[boots].sum(1), 1)
        brc = dr[boots].sum(1) / np.maximum(nr_t[boots].sum(1), 1)
        row["boot_ft_pts"] = [float(100 * np.percentile(bft, 2.5)), float(100 * np.percentile(bft, 97.5))]
        row["boot_recall_pts"] = [float(100 * np.percentile(brc, 2.5)), float(100 * np.percentile(brc, 97.5))]
        merged[m] = row
    keys = [(m, lab) for m in merged for lab in ("n", "r")]
    hp = fe.holm([merged[m][lab]["p"] for m, lab in keys])
    for (m, lab), p in zip(keys, hp):
        merged[m][lab]["p_holm"] = p
    res["stats_it2_gaze_with_extra"] = merged
    res["n_tests"] = len(keys)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "eval.json"), "w"), indent=1)
    md = [f"# Extra baselines ({'walking, fitted on standing' if walking else 'standing, leave-one-recording-out'})", "",
          f"Episodes r {res['episodes']['r']}, n {res['episodes']['n']}; LOOK keypoints missing on {miss} person-frames.", "",
          "| Policy | False triggers % | Recall % |", "|---|---|---|"]
    for m, v in res["policies"].items():
        md.append(f"| {m} | {100 * v['false_trigger']['p']:.1f} | {100 * v['recall']['p']:.1f} |")
    md += ["", "McNemar vs IT2 + gaze (episodes where only one policy accepts):", ""]
    for m, v in res["mcnemar_vs_it2_gaze"].items():
        md.append(f"- {m}: false triggers {v['false_trigger']['other_only']} vs {v['false_trigger']['it2_gaze_only']} "
                  f"(p={v['false_trigger']['p']:.2g}); recall {v['recall']['other_only']} vs "
                  f"{v['recall']['it2_gaze_only']} (p={v['recall']['p']:.2g})")
    open(os.path.join(OUT, "eval.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
