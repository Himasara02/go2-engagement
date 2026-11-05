#!/usr/bin/env python3
"""Task 4: single-cue and learned baselines next to the fuzzy policies.

Every baseline uses the same gate as the fuzzy policies except the engagement
test: a person's gesture counts in a frame if it is a command gesture with
conf >= theta_c (0.45) AND their score passes a threshold; then the same
3-frame sustain.
  facing_only   score = facing                        (higher = more engaged)
  distance_only score = -distance                     (closer = more engaged)
  logreg        score = P(robot-directed | distance, approach, facing,
                         sigma_d, sigma_f), standardised inputs, balanced classes
Leave-one-take-out (results/protocol.md): for each held-out take, the
threshold (and the logistic model) are fitted on the other 13 takes only. The
threshold maximises Youden's J = recall - false-trigger rate on the training
takes' episodes. Writes results/baselines.{json,md}.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis import (MAIN_DIR, NAMES, episode_records, fmt, load_all, rate, save,
                      scenario_takes, sustained)
from engage.fuzzy import DecisionParams
from engage.methods import COMMAND_MAP

THETA_C = DecisionParams().theta_conf
FEATS = ["distance", "approach", "facing", "distance_sigma", "facing_sigma"]
NAMES = {**NAMES, "facing_only": "Facing only", "distance_only": "Distance only",
         "logreg": "Logistic regression"}
ORDER = ["gesture_only", "nearest", "facing_only", "distance_only", "logreg", "type1", "it2"]


def candidates(recs):
    """Per frame: [(tid, is_candidate, person dict)] - candidate = command gesture, conf >= theta_c."""
    return [[(p["tid"], p.get("gesture") in COMMAND_MAP and p["gesture_conf"] >= THETA_C, p)
             for p in r["people"]] for r in recs]


def labelled_eps(labels, eps):
    return [(e, labels.get(str(e["id"]))) for e in eps if labels.get(str(e["id"])) in ("r", "n")]


def frame_sets(cands, score, tau):
    return sustained([{t for t, c, p in fr if c and score(p) >= tau} for fr in cands])


def youden(take_data, score, grid):
    """Best threshold on the given takes: max recall - FT over episodes."""
    best = (-2, None)
    for tau in grid:
        ar = nr = an = nn = 0
        for cands, leps in take_data:
            sus = frame_sets(cands, score, tau)
            for e, lab in leps:
                hit = any(e["tid"] in sus[i] for i in range(e["start"], e["end"] + 1))
                if lab == "r":
                    nr += 1
                    ar += hit
                else:
                    nn += 1
                    an += hit
        j = (ar / nr if nr else 0) - (an / nn if nn else 0)
        if j > best[0]:
            best = (j, tau)
    return best[1]


def frame_rows(recs, leps):
    """Training rows: every frame of a labelled episode, that person's features."""
    X, y = [], []
    for e, lab in leps:
        for i in range(e["start"], e["end"] + 1):
            for p in recs[i]["people"]:
                if p["tid"] == e["tid"]:
                    X.append([p[k] for k in FEATS])
                    y.append(lab == "r")
    return X, y


def main():
    takes = scenario_takes()
    loaded = load_all(takes)
    names = [m["take"] for m, *_ in loaded]
    cands = {n: candidates(recs) for n, (m, lab, recs, eps) in zip(names, loaded)}
    leps = {n: labelled_eps(lab, eps) for n, (m, lab, recs, eps) in zip(names, loaded)}
    rows = {n: frame_rows(recs, leps[n]) for n, (m, lab, recs, eps) in zip(names, loaded)}

    single = {"facing_only": (lambda p: p["facing"], np.round(np.arange(0.0, 1.001, 0.02), 2)),
              "distance_only": (lambda p: -p["distance"], -np.round(np.arange(0.5, 6.01, 0.1), 2))}
    held = {m: {} for m in ("facing_only", "distance_only", "logreg")}
    fitted = {m: {} for m in held}
    for test in names:
        train = [n for n in names if n != test]
        for m, (score, grid) in single.items():
            tau = youden([(cands[n], leps[n]) for n in train], score, grid)
            fitted[m][test] = float(tau)
            held[m][test] = frame_sets(cands[test], score, tau)
        X = sum((rows[n][0] for n in train), [])
        y = sum((rows[n][1] for n in train), [])
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        clf.fit(np.array(X), np.array(y))
        prob = {}

        def lr_score(p, prob=prob):
            return prob[id(p)]
        for n in names:   # probabilities for every person-frame, from this fold's model
            ps = [p for fr in cands[n] for _, _, p in fr]
            if ps:
                pr = clf.predict_proba(np.array([[p[k] for k in FEATS] for p in ps]))[:, 1]
                prob.update({id(p): v for p, v in zip(ps, pr)})
        tau = youden([(cands[n], leps[n]) for n in train], lr_score, np.round(np.arange(0.05, 0.96, 0.025), 3))
        lr = clf[-1]
        fitted["logreg"][test] = {"tau": float(tau), "coef": dict(zip(FEATS, map(float, lr.coef_[0]))),
                                  "intercept": float(lr.intercept_[0])}
        held["logreg"][test] = frame_sets(cands[test], lr_score, tau)

    eps, missed = episode_records(takes, loaded=loaded,
                                  extra=lambda recs, take: {m: held[m][take] for m in held})
    r = [e for e in eps if e["label"] == "r"]
    n = [e for e in eps if e["label"] == "n"]
    m_tot = sum(missed.values())
    res = {"data": MAIN_DIR, "protocol": "leave-one-take-out, Youden J threshold on training takes",
           "fitted": fitted, "policies": {}}
    lines = ["| Policy | Bystander false triggers | Recall (clips) | Recall (incl. missed) | Youden J |",
             "|---|---|---|---|---|"]
    for m in ORDER:
        ft = rate(sum(e["acc"][m] for e in n), len(n))
        rc = rate(sum(e["acc"][m] for e in r), len(r))
        rm = rate(rc["k"], len(r) + m_tot)
        res["policies"][m] = {"false_trigger": ft, "recall_clips": rc, "recall_incl_missed": rm}
        lines.append(f"| {NAMES[m]} | {fmt(ft, True)} | {fmt(rc, True)} | {fmt(rm)} | "
                     f"{rc['p'] - ft['p']:.2f} |")
    taus = {m: [fitted[m][t] if m != "logreg" else fitted[m][t]["tau"] for t in names] for m in fitted}
    notes = [f"- {NAMES[m]}: threshold per fold median {np.median(v):.2f} (range {min(v):.2f} to {max(v):.2f})"
             for m, v in taus.items()]
    coef = {k: np.median([fitted["logreg"][t]["coef"][k] for t in names]) for k in FEATS}
    notes.append("- Logistic regression median coefficients (standardised): " +
                 ", ".join(f"{k} {v:+.2f}" for k, v in coef.items()))
    md = "\n".join(["# Baselines (task 4)", "",
                    "Same gesture gate (command gesture, conf ≥ 0.45) and 3-frame sustain for all. "
                    "Learned thresholds/models: leave-one-take-out, Youden J on the 13 training takes. "
                    "Brackets: Wilson 95% CI.", "", *lines, "", *notes,
                    "- Distance threshold is shown as negative distance (accept if distance ≤ −τ)."])
    save("baselines", res, md)
    print(md)


if __name__ == "__main__":
    main()
