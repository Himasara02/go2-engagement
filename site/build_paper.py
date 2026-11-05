#!/usr/bin/env python3
"""Fill paper_template.html from results.json -> paper.pdf (A4, two-column).
Usage: python build_paper.py <results_dir>
"""
import json
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person",
         "type1": "Type-1 engagement", "it2": "IT2 engagement (ours)"}


def f(r, k):
    p, lo, hi = r[k]
    n = r[k.replace("false_trigger", "false_n").replace("recall", "recall_n")]
    if n[1] == 0:
        return "—"
    return f"{100*p:.0f}% ({n[0]}/{n[1]})"


def main():
    R = json.load(open(os.path.join(sys.argv[1], "results.json")))
    T = R["table"]
    rows = ["<table><tr><th rowspan=2>Policy</th><th colspan=2>Stationary</th><th colspan=2>Walking</th></tr>"
            "<tr><th>False trig.↓</th><th>Recall↑</th><th>False trig.↓</th><th>Recall↑</th></tr>"]
    for m, nm in NAMES.items():
        c = [f(T[f"{sc}/{m}"], k) for sc in ("stationary", "walking") for k in ("false_trigger", "recall")]
        b = " style='font-weight:bold'" if m == "it2" else ""
        rows.append(f"<tr{b}><td>{nm}</td>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>")
    nst = T["stationary/gesture_only"]["false_n"][1] + T["stationary/gesture_only"]["recall_n"][1]
    nwk = T["walking/gesture_only"]["false_n"][1] + T["walking/gesture_only"]["recall_n"][1]
    rows.append(f"</table><div style='font-size:8pt'>TABLE I. Bystander false-trigger rate and commander recall. "
                f"Episodes: {nst} stationary, {nwk} walking.</div>")
    pc = lambda sc, m, k: f"{100*T[f'{sc}/{m}'][k][0]:.0f}%"
    n_eps = sum(t["robot_eps"] + t["other_eps"] for t in R["takes"])
    abs_res = (f"the IT2 gate reduced bystander false triggers from {pc('walking','gesture_only','false_trigger')} "
               f"(gesture only) and {pc('walking','type1','false_trigger')} (Type-1) to {pc('walking','it2','false_trigger')} "
               f"while the robot walked, with {pc('walking','it2','recall')} commander recall.")
    res_txt = (f"Table I summarises {n_eps} labelled episodes. With the robot stationary, gesture-only control executed "
               f"{pc('stationary','gesture_only','false_trigger')} of bystander gestures, nearest-person "
               f"{pc('stationary','nearest','false_trigger')}, Type-1 {pc('stationary','type1','false_trigger')} and IT2 "
               f"{pc('stationary','it2','false_trigger')}, with commander recall of {pc('stationary','gesture_only','recall')}, "
               f"{pc('stationary','nearest','recall')}, {pc('stationary','type1','recall')} and {pc('stationary','it2','recall')}. "
               f"While walking, false triggers were {pc('walking','gesture_only','false_trigger')}, {pc('walking','nearest','false_trigger')}, "
               f"{pc('walking','type1','false_trigger')} and {pc('walking','it2','false_trigger')}, and recall "
               f"{pc('walking','gesture_only','recall')}, {pc('walking','nearest','recall')}, {pc('walking','type1','recall')} "
               f"and {pc('walking','it2','recall')}.")
    html = open(os.path.join(HERE, "paper_template.html")).read()
    for k, v in {"TABLE": "\n".join(rows), "N_TAKES": str(len(R["takes"])), "N_EPS": str(n_eps),
                 "ABS_RESULT": abs_res, "RESULT_TEXT": os.environ.get("RESULT_TEXT", res_txt)}.items():
        html = html.replace("{{" + k + "}}", v)
    out = os.path.join(HERE, "paper.html")
    open(out, "w").write(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + out)
        pg.wait_for_timeout(400)
        pg.pdf(path=os.path.join(HERE, "paper.pdf"), format="A4", print_background=True,
               margin={"top": "18mm", "bottom": "18mm", "left": "15mm", "right": "15mm"})
        b.close()
    print("paper.pdf written")


if __name__ == "__main__":
    main()
