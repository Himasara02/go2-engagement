#!/usr/bin/env python3
"""Task 5: statistics for the policy comparison.

- Wilson 95% CIs per policy (episode level, as in tasks 3-4).
- Exact McNemar tests on paired episodes (same episode, two policies), separately
  for bystander (n) and robot-directed (r) episodes. Primary comparison: IT2 vs
  Type-1. Secondary: IT2 vs each other policy, Holm-corrected within each label.
- Bootstrap over takes (B = 10000, seed 0): resample the 14 takes with replacement,
  pool their episodes, recompute rates; percentile 95% CIs for each policy and for
  the differences IT2 - X. Captures take-to-take variation that the per-episode
  Wilson interval ignores.
Writes results/stats.{json,md}.
"""
import numpy as np
from scipy.stats import binomtest

from analysis import fmt, rate, save
from task4_baselines import NAMES, ORDER, compute

B, SEED = 10000, 0


def mcnemar(eps, a, b):
    """Exact two-sided McNemar: discordant pairs only."""
    only_a = sum(e["acc"][a] and not e["acc"][b] for e in eps)
    only_b = sum(e["acc"][b] and not e["acc"][a] for e in eps)
    n = only_a + only_b
    p = binomtest(min(only_a, only_b), n, 0.5).pvalue if n else 1.0
    return {"only_" + a: only_a, "only_" + b: only_b, "p": p}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def bootstrap(eps, names):
    """Per take, per policy: [accepted n, total n, accepted r, total r]."""
    idx = {t: i for i, t in enumerate(names)}
    cnt = np.zeros((len(names), len(ORDER), 4))
    for e in eps:
        if e["label"] not in ("r", "n"):
            continue
        off = 0 if e["label"] == "n" else 2
        for j, m in enumerate(ORDER):
            cnt[idx[e["take"]], j, off] += e["acc"][m]
            cnt[idx[e["take"]], j, off + 1] += 1
    rng = np.random.default_rng(SEED)
    samp = cnt[rng.integers(0, len(names), (B, len(names)))].sum(1)      # B x policy x 4
    with np.errstate(invalid="ignore", divide="ignore"):
        ft = samp[..., 0] / samp[..., 1]
        rc = samp[..., 2] / samp[..., 3]
    return ft, rc


def ci(x):
    x = x[~np.isnan(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    eps, missed, fitted, names = compute()
    groups = {"n": [e for e in eps if e["label"] == "n"], "r": [e for e in eps if e["label"] == "r"]}
    res = {"wilson": {}, "mcnemar": {}, "bootstrap": {"B": B, "seed": SEED, "unit": "take",
                                                     "policies": {}, "diff_vs_it2": {}}}
    for m in ORDER:
        res["wilson"][m] = {"false_trigger": rate(sum(e["acc"][m] for e in groups["n"]), len(groups["n"])),
                            "recall": rate(sum(e["acc"][m] for e in groups["r"]), len(groups["r"]))}

    others = [m for m in ORDER if m != "it2"]
    for lab, g in groups.items():
        tests = {m: mcnemar(g, "it2", m) for m in others}
        for m, p in zip(others, holm([tests[m]["p"] for m in others])):
            tests[m]["p_holm"] = p
        res["mcnemar"][lab] = tests

    ft, rc = bootstrap(eps, names)
    j_it2 = ORDER.index("it2")
    for j, m in enumerate(ORDER):
        res["bootstrap"]["policies"][m] = {"false_trigger": ci(ft[:, j]), "recall": ci(rc[:, j])}
        if m != "it2":
            res["bootstrap"]["diff_vs_it2"][m] = {"false_trigger": ci(ft[:, j_it2] - ft[:, j]),
                                                 "recall": ci(rc[:, j_it2] - rc[:, j])}

    pct = lambda v: f"{100 * v[0]:.0f}–{100 * v[1]:.0f}%"
    spct = lambda v: f"{100 * v[0]:+.0f} to {100 * v[1]:+.0f} pp"
    t1 = ["| Policy | False triggers | Wilson 95% | Bootstrap 95% (takes) | Recall | Wilson 95% | Bootstrap 95% (takes) |",
          "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        w, b = res["wilson"][m], res["bootstrap"]["policies"][m]
        t1.append(f"| {NAMES[m]} | {fmt(w['false_trigger'])} | {pct((w['false_trigger']['lo'], w['false_trigger']['hi']))} | "
                  f"{pct(b['false_trigger'])} | {fmt(w['recall'])} | {pct((w['recall']['lo'], w['recall']['hi']))} | "
                  f"{pct(b['recall'])} |")
    t2 = ["| IT2 vs | Bystander eps: only IT2 / only other | p | p (Holm) | Robot eps: only IT2 / only other | p | p (Holm) | "
          "Δ false triggers (boot.) | Δ recall (boot.) |", "|---|---|---|---|---|---|---|---|---|"]
    for m in others:
        a, b = res["mcnemar"]["n"][m], res["mcnemar"]["r"][m]
        d = res["bootstrap"]["diff_vs_it2"][m]
        t2.append(f"| {NAMES[m]} | {a['only_it2']} / {a['only_' + m]} | {a['p']:.3g} | {a['p_holm']:.3g} | "
                  f"{b['only_it2']} / {b['only_' + m]} | {b['p']:.3g} | {b['p_holm']:.3g} | "
                  f"{spct(d['false_trigger'])} | {spct(d['recall'])} |")
    md = "\n".join(["# Statistics (task 5)", "",
                    f"Episodes: {len(groups['n'])} bystander (n), {len(groups['r'])} robot-directed (r), "
                    f"{len(names)} takes. Bootstrap: {B} resamples of takes, seed {SEED}.", "",
                    "## Rates with confidence intervals", "", *t1, "",
                    "## Paired comparisons (exact McNemar) and bootstrap differences (IT2 − other)", "",
                    "Primary test: IT2 vs Type-1. Holm correction over the 6 comparisons within each label.", "",
                    *t2])
    save("stats", res, md)
    print(md)


if __name__ == "__main__":
    main()
