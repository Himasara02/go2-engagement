#!/usr/bin/env python3
"""Task 6: analysis of the fuzzy gate (no change to fuzzy.py / methods.py).

  1. theta_E sweep: false-trigger vs recall for Type-1 and IT2 (STOP keeps the
     frozen offset theta_stop = theta_motion - 0.15), facing-only for reference.
  2. Splits by measured sigma (median of sigma_f / sigma_d per episode), by
     distance bin and by scenario S1-S6, at the frozen operating point.
  3. Ablations: inputs removed from the rule base (antecedent dropped; rules left
     without antecedents dropped): no approach, facing only, distance only.
  4. FOU scale k in {0, 0.5, 1, 2} for all inputs (k = 0 is Type-1).
  5. Noise injection: i.i.d. Gaussian noise on distance or facing per person-frame,
     20 seeds. IT2 gets sigma' = sqrt(sigma^2 + s^2) (as a live sigma estimate would)
     and, as a control, the unchanged logged sigma. Facing-only keeps its
     leave-one-take-out thresholds from task 4.
The engagement system is rebuilt here from fuzzy.py's own membership functions,
rules and Karnik-Mendel code; with no input removed it reproduces methods.engagement()
exactly (checked at start-up).
Writes results/analysis.{json,md}, results/fig_sweep.{pdf,png}, results/fig_noise.{pdf,png}.
"""
import json
import math
import os

import numpy as np

from analysis import NAMES, RESULTS, load_all, rate, save, scenario_takes, sustained
from engage.fuzzy import RULES, DecisionParams, STOP_LABELS, _km, mf_approach, mf_distance, mf_facing
from engage.methods import COMMAND_MAP

P = DecisionParams()
STOP_OFF = P.theta_e_motion - P.theta_e_stop        # 0.15, frozen
SEEDS = 20
COL = {"it2": "#2a78d6", "type1": "#eb6834", "facing_only": "#1baf7a", "ref": "#8a8984"}


# ----------------------------------------------------------------- engagement
def fls(dist, sd, face, sf, appr, sa=0.08, it2=True, k=1.0, drop=()):
    """Engagement with inputs in `drop` removed from every rule. Returns E (T1) or E_low (IT2)."""
    z = k if it2 else 0.0
    sets = {"dist": mf_distance(dist, z * sd), "face": mf_facing(face, z * sf),
            "appr": mf_approach(appr, z * sa)}
    y, fl, fu = [], [], []
    for ante, cons in RULES:
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


class Data:
    """Flattened person-frames of all takes + labelled episodes."""

    def __init__(self):
        self.loaded = load_all(scenario_takes())
        self.takes = []
        self.pf = []                      # flat list of person dicts
        for meta, labels, recs, eps in self.loaded:
            frames = []
            for r in recs:
                frames.append([len(self.pf) + i for i in range(len(r["people"]))])
                self.pf.extend(r["people"])
            leps = [(e, labels[str(e["id"])]) for e in eps if labels.get(str(e["id"])) in ("r", "n")]
            self.takes.append({"name": meta["take"], "frames": frames, "eps": leps})
        g = self.pf
        self.tid = np.array([p["tid"] for p in g])
        self.cand = np.array([p.get("gesture") in COMMAND_MAP and p["gesture_conf"] >= P.theta_conf for p in g])
        self.stop = np.array([p.get("gesture") in STOP_LABELS for p in g])
        self.x = {k: np.array([p[k] for p in g], float)
                  for k in ("distance", "distance_sigma", "facing", "facing_sigma", "approach")}

    def engage(self, it2=True, k=1.0, drop=(), x=None):
        x = x or self.x
        return np.array([fls(x["distance"][i], x["distance_sigma"][i], x["facing"][i], x["facing_sigma"][i],
                             x["approach"][i], it2=it2, k=k, drop=drop) for i in range(len(self.pf))])

    def score(self, ok, sel=None):
        """ok: bool per person-frame (gate passed). Returns per-episode accepted list
        [(take, episode, label, accepted)] restricted by sel(take, e)."""
        out = []
        for t in self.takes:
            sus = sustained([{self.tid[i] for i in fr if ok[i]} for fr in t["frames"]])
            for e, lab in t["eps"]:
                if sel and not sel(t["name"], e):
                    continue
                out.append((t["name"], e, lab, any(e["tid"] in sus[i] for i in range(e["start"], e["end"] + 1))))
        return out

    def gate(self, e, th=P.theta_e_motion):
        return self.cand & (e >= np.where(self.stop, th - STOP_OFF, th))


def rates(res):
    n = [a for *_, lab, a in res if lab == "n"]
    r = [a for *_, lab, a in res if lab == "r"]
    return rate(sum(n), len(n)), rate(sum(r), len(r))


def main():
    d = Data()
    # sanity: rebuilt system == methods.engagement() logged values
    chk = [(fls(p["distance"], p["distance_sigma"], p["facing"], p["facing_sigma"], p["approach"], it2=False),
            fls(p["distance"], p["distance_sigma"], p["facing"], p["facing_sigma"], p["approach"]),
            p["e_t1"], p["e2_low"]) for p in d.pf[::50]]
    assert all(abs(a - c) < 1e-3 and abs(b - e) < 1e-3 for a, b, c, e in chk), "rebuild mismatch"
    E = {"type1": d.engage(it2=False), "it2": d.engage()}
    face = d.x["facing"]
    fo_tau = json.load(open(os.path.join(RESULTS, "baselines.json")))["fitted"]["facing_only"]
    fo_th = np.concatenate([np.full(sum(len(fr) for fr in t["frames"]), fo_tau[t["name"]]) for t in d.takes])
    fo_ok = lambda f: d.cand & (f >= fo_th)     # per-take leave-one-take-out threshold
    out = {}

    # 1. sweep --------------------------------------------------------------
    grid = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    sweep = {m: [] for m in ("type1", "it2", "facing_only")}
    for th in grid:
        for m in ("type1", "it2"):
            ft, rc = rates(d.score(d.gate(E[m], th)))
            sweep[m].append([float(th), ft["p"], rc["p"]])
        ft, rc = rates(d.score(d.cand & (face >= th)))
        sweep["facing_only"].append([float(th), ft["p"], rc["p"]])

    def best_recall(curve, max_ft):
        return max((rc for _, ft, rc in curve if ft <= max_ft + 1e-9), default=0.0)
    out["sweep"] = {"grid": grid.tolist(), "curves": sweep, "stop_offset": STOP_OFF,
                    "recall_at_ft": {m: {f: best_recall(c, f) for f in (0.02, 0.05, 0.10, 0.20)}
                                     for m, c in sweep.items()}}
    op = {m: rates(d.score(d.gate(E[m]))) for m in ("type1", "it2")}
    op["facing_only"] = rates(d.score(fo_ok(face)))
    op["gesture_only"] = rates(d.score(d.cand))

    # 2. splits -------------------------------------------------------------
    epm = {}
    for t in d.takes:
        for e, lab in t["eps"]:
            idx = [i for fr in t["frames"][e["start"]:e["end"] + 1] for i in fr if d.tid[i] == e["tid"]]
            epm[(t["name"], e["id"])] = {k: float(np.mean(d.x[k][idx])) for k in d.x}
    med = {k: float(np.median([v[k] for v in epm.values()])) for k in ("facing_sigma", "distance_sigma")}
    bins = {
        "facing σ": [("low", lambda v: v["facing_sigma"] <= med["facing_sigma"]),
                     ("high", lambda v: v["facing_sigma"] > med["facing_sigma"])],
        "distance σ": [("low", lambda v: v["distance_sigma"] <= med["distance_sigma"]),
                       ("high", lambda v: v["distance_sigma"] > med["distance_sigma"])],
        "distance": [("< 2 m", lambda v: v["distance"] < 2), ("2–3 m", lambda v: 2 <= v["distance"] < 3),
                     ("≥ 3 m", lambda v: v["distance"] >= 3)],
        "scenario": [(f"S{i}", None) for i in range(1, 7)],
    }
    splits = {}
    for split, bl in bins.items():
        splits[split] = {}
        for name, f in bl:
            if split == "scenario":
                sel = lambda take, e, name=name: take.split("_")[1] == name
            else:
                sel = lambda take, e, f=f: f(epm[(take, e["id"])])
            splits[split][name] = {m: rates(d.score(ok, sel)) for m, ok in
                                   (("type1", d.gate(E["type1"])), ("it2", d.gate(E["it2"])),
                                    ("facing_only", fo_ok(face)))}
    out["splits"] = {"medians": med, "bins": splits}

    # 3-4. ablations and FOU scale --------------------------------------------
    abl = {}
    for name, drop in (("full", ()), ("no approach", ("appr",)), ("facing only", ("dist", "appr")),
                       ("distance only", ("face", "appr"))):
        abl[name] = {m: rates(d.score(d.gate(d.engage(it2=(m == "it2"), drop=drop)))) for m in ("type1", "it2")}
    fou = {str(k): rates(d.score(d.gate(d.engage(k=k) if k else E["type1"]))) for k in (0, 0.5, 1, 2)}
    out["ablations"], out["fou_k"] = abl, fou

    # 5. noise --------------------------------------------------------------
    levels = {"facing": [0, 0.1, 0.2, 0.3], "distance": [0, 0.3, 0.6, 1.0]}
    noise = {}
    for var, ss in levels.items():
        sk = var + "_sigma"
        noise[var] = {}
        for s in ss:
            acc = {m: [] for m in ("type1", "it2", "it2_fixed_sigma", "facing_only")}
            for seed in range(SEEDS if s else 1):
                rng = np.random.default_rng(seed)
                x = dict(d.x)
                v = d.x[var] + rng.normal(0, s, len(d.pf))
                x[var] = np.clip(v, 0, 1) if var == "facing" else np.clip(v, 0.3, None)
                xs = dict(x)
                xs[sk] = np.sqrt(d.x[sk] ** 2 + s ** 2)
                for m, ok in (("type1", d.gate(d.engage(it2=False, x=x))), ("it2", d.gate(d.engage(x=xs))),
                              ("it2_fixed_sigma", d.gate(d.engage(x=x))), ("facing_only", fo_ok(x["facing"]))):
                    ft, rc = rates(d.score(ok))
                    acc[m].append([ft["p"], rc["p"]])
            noise[var][str(s)] = {m: {"ft_mean": float(np.mean([a[0] for a in v])),
                                      "ft_sd": float(np.std([a[0] for a in v])),
                                      "recall_mean": float(np.mean([a[1] for a in v])),
                                      "recall_sd": float(np.std([a[1] for a in v]))} for m, v in acc.items()}
    out["noise"] = {"seeds": SEEDS, "levels": levels, "results": noise}
    out["operating_point"] = op

    # ---------------------------------------------------------------- report
    pc = lambda r: f"{100 * r['p']:.0f}% ({r['k']}/{r['n']})" if r["n"] else "–"
    L = ["# Analysis (task 6)", "",
         "Frozen operating point unless stated. FT = bystander false triggers, R = recall (clips).", "",
         "## 1. θ_E sweep (STOP threshold = θ_E − 0.15)", "",
         "Best recall reachable at a false-trigger budget:", "",
         "| Policy | FT ≤ 2% | FT ≤ 5% | FT ≤ 10% | FT ≤ 20% |", "|---|---|---|---|---|"]
    for m in ("type1", "it2", "facing_only"):
        L.append(f"| {NAMES.get(m, 'Facing only')} | " +
                 " | ".join(f"{100 * out['sweep']['recall_at_ft'][m][f]:.0f}%" for f in (0.02, 0.05, 0.10, 0.20)) + " |")
    L += ["", "Figure: `fig_sweep.pdf`.", "", "## 2. Splits", "",
          f"Median episode σ_f = {med['facing_sigma']:.3f}, σ_d = {med['distance_sigma']:.3f} m.", "",
          "| Split | Bin | Type-1 FT | Type-1 R | IT2 FT | IT2 R | Facing-only FT | Facing-only R |",
          "|---|---|---|---|---|---|---|---|"]
    for split, bl in splits.items():
        for name, v in bl.items():
            L.append(f"| {split} | {name} | {pc(v['type1'][0])} | {pc(v['type1'][1])} | {pc(v['it2'][0])} | "
                     f"{pc(v['it2'][1])} | {pc(v['facing_only'][0])} | {pc(v['facing_only'][1])} |")
    L += ["", "## 3. Ablations (inputs removed from the rule base)", "",
          "| Inputs | Type-1 FT | Type-1 R | IT2 FT | IT2 R |", "|---|---|---|---|---|"]
    for name, v in abl.items():
        L.append(f"| {name} | {pc(v['type1'][0])} | {pc(v['type1'][1])} | {pc(v['it2'][0])} | {pc(v['it2'][1])} |")
    L += ["", "## 4. FOU scale k (all inputs; k = 0 ≡ Type-1)", "", "| k | FT | R |", "|---|---|---|"]
    for k, v in fou.items():
        L.append(f"| {k} | {pc(v[0])} | {pc(v[1])} |")
    L += ["", f"## 5. Noise injection ({SEEDS} seeds, mean ± sd)", "",
          "IT2 = σ updated with the noise; IT2 (fixed σ) = logged σ, a control.", "",
          "| Noise | s | Type-1 FT | Type-1 R | IT2 FT | IT2 R | IT2 fixed σ FT | IT2 fixed σ R | Facing-only FT | Facing-only R |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for var, lv in noise.items():
        for s, v in lv.items():
            cell = lambda m: (f"{100 * v[m]['ft_mean']:.1f} ± {100 * v[m]['ft_sd']:.1f}%",
                              f"{100 * v[m]['recall_mean']:.1f} ± {100 * v[m]['recall_sd']:.1f}%")
            L.append(f"| {var} | {s} | " + " | ".join(" | ".join(cell(m)) for m in
                                                      ("type1", "it2", "it2_fixed_sigma", "facing_only")) + " |")
    save("analysis", out, "\n".join(L))
    figures(out)
    print("\n".join(L))


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
    w = 12.2 / 2.54
    style = {"it2": ("IT2 fuzzy", COL["it2"], "-", "o"), "type1": ("Type-1 fuzzy", COL["type1"], "--", "s"),
             "facing_only": ("Facing only", COL["facing_only"], ":", "^")}

    fig, ax = plt.subplots(figsize=(w, w * 0.55))
    for m, (lab, c, ls, mk) in style.items():
        cur = out["sweep"]["curves"][m]
        ax.plot([100 * a[1] for a in cur], [100 * a[2] for a in cur], ls, color=c, lw=1.5, label=lab)
        if m != "facing_only":
            o = out["operating_point"][m]
            ax.plot(100 * o[0]["p"], 100 * o[1]["p"], mk, color=c, ms=6, mec="white", mew=1)
    g = out["operating_point"]["gesture_only"]
    ax.plot(100 * g[0]["p"], 100 * g[1]["p"], "D", color=COL["ref"], ms=6, mec="white", mew=1)
    ax.annotate("gesture only", (100 * g[0]["p"], 100 * g[1]["p"]), xytext=(-6, 6), textcoords="offset points",
                ha="right", fontsize=7, color="#52514e")
    ax.set_xlabel("Bystander false-trigger rate (%)")
    ax.set_ylabel("Command recall (%)")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 80)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_sweep.{ext}"), dpi=200)

    fig, axs = plt.subplots(2, 2, figsize=(w, w * 0.8), sharey="row", sharex="col")
    lv = out["noise"]["levels"]
    for col, var, unit in ((0, "facing", ""), (1, "distance", " (m)")):
      for row, key in ((0, "ft"), (1, "recall")):
        ax = axs[row, col]
        for m, lab, c, ls, mk in (("type1", "Type-1", COL["type1"], "--", "s"), ("it2", "IT2", COL["it2"], "-", "o"),
                                  ("it2_fixed_sigma", "IT2, σ not updated", COL["it2"], ":", "x"),
                                  ("facing_only", "Facing only", COL["facing_only"], ":", "^")):
            v = [out["noise"]["results"][var][str(s)][m] for s in lv[var]]
            ax.errorbar(lv[var], [100 * a[key + "_mean"] for a in v], yerr=[100 * a[key + "_sd"] for a in v],
                        fmt=ls, marker=mk, color=c, lw=1.2, ms=4, capsize=2, label=lab)
        if row:
            ax.set_xlabel(f"Added noise s on {var}{unit}")
        ax.grid(alpha=0.25, lw=0.5)
    axs[0, 0].set_ylabel("False triggers (%)")
    axs[1, 0].set_ylabel("Recall (%)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"fig_noise.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
