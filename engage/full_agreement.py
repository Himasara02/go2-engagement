#!/usr/bin/env python3
"""Inter-annotator agreement: annotator A (labels_final.csv: 36 recordings episode by episode, 24 by choosing the
commander) vs. annotator C (data/labels_C_2026-11-12, every episode of all 60 recordings, labelled independently).

    python3 full_agreement.py

Cohen's kappa on the episodes both labelled r or n (x = bad detection is reported separately), raw agreement,
confusion matrix, per labelling mode of A, per scenario, per day. Also writes labels_final_C.csv (A's file with C's
label) and labels_final_agree.csv (only episodes where A and C agree) for re-running full_eval.py with
LABELS=<file>. Writes results/full/agreement.{json,md}. Standard library only.
"""
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LA = os.path.join(ROOT, "data", "depthpack_2026-11-11", "labels_pack_2026-11-10", "labels_final.csv")
LC = os.path.join(ROOT, "data", "labels_C_2026-11-12", "labels_C")
OUT = os.path.join(HERE, "results", "full")

rowsA = list(csv.DictReader(open(LA)))
C, missedC = {}, {}
for f in glob.glob(os.path.join(LC, "*", "labels_C.json")):
    take = os.path.basename(os.path.dirname(f))
    for k, v in json.load(open(f)).items():
        if k == "_missed":
            missedC[take] = v
        elif not k.startswith("_"):
            C[(take, k)] = v


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == c for a, c in pairs) / n
    ca, cc = Counter(a for a, _ in pairs), Counter(c for _, c in pairs)
    pe = sum(ca[k] * cc[k] for k in set(ca) | set(cc)) / n ** 2
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    # large-sample standard error (Cohen 1960) -> 95 % CI
    se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)) if pe < 1 else 0.0
    return {"n": n, "agreement": po, "kappa": k, "ci": [k - 1.96 * se, k + 1.96 * se]}


both = [(r, C[(r["take"], r["clip"])]) for r in rowsA]
rn = [(r["label"], c) for r, c in both if r["label"] in "rn" and c in "rn"]
groups = defaultdict(list)
for r, c in both:
    if r["label"] in "rn" and c in "rn":
        groups["source " + r["label_source"]].append((r["label"], c))
        groups["scenario " + r["take"].split("_")[1]].append((r["label"], c))
        groups["day " + r["take"].split("_")[0]].append((r["label"], c))
conf = Counter((r["label"], c) for r, c in both)
res = {"episodes": len(both), "overall_rn": kappa(rn), "confusion_A_C": {f"{a}->{c}": v for (a, c), v in sorted(conf.items())},
       "x": {"A": sum(r["label"] == "x" for r, _ in both), "C": sum(c == "x" for _, c in both)},
       "missed_commands_C": sum(v for v in missedC.values() if isinstance(v, int)),
       "missed_not_given_C": sorted(t for t, v in missedC.items() if not isinstance(v, int)), "groups": {g: kappa(p) for g, p in sorted(groups.items())},
       "labels_A": dict(Counter(r["label"] for r, _ in both)), "labels_C": dict(Counter(c for _, c in both))}
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "agreement.json"), "w"), indent=1)

fields = list(rowsA[0].keys())
with open(os.path.join(OUT, "labels_final_C.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    for r, c in both:
        w.writerow({**r, "label": c, "label_source": r["label_source"]})
with open(os.path.join(OUT, "labels_final_agree.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fields)
    w.writeheader()
    for r, c in both:
        w.writerow({**r, "label": r["label"] if r["label"] == c else "x"})

o = res["overall_rn"]
md = ["# Inter-annotator agreement (A vs C)", "",
      f"Episodes {res['episodes']}; both r/n: {o['n']}; raw agreement {100 * o['agreement']:.1f} %; "
      f"Cohen's kappa {o['kappa']:.3f} [{o['ci'][0]:.3f}, {o['ci'][1]:.3f}].",
      f"x labels: A {res['x']['A']}, C {res['x']['C']}. Missed commands reported by C: {res['missed_commands_C']}.",
      "", "Confusion (A -> C): " + ", ".join(f"{k} {v}" for k, v in res["confusion_A_C"].items()), "",
      "| Group | n | agreement | kappa |", "|---|---|---|---|"]
md += [f"| {g} | {v['n']} | {100 * v['agreement']:.1f} % | {v['kappa']:.3f} |" for g, v in res["groups"].items() if v]
open(os.path.join(OUT, "agreement.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
