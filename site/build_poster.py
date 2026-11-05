#!/usr/bin/env python3
"""Fill poster_template.html -> poster.pdf (1440x810).
Usage: python build_poster.py <results_dir|-> [frame.png] [url]
"""
import json
import os
import shutil
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person",
         "type1": "+ Type-1 engagement", "it2": "+ Interval Type-2"}


def pct(x):
    return "—" if x != x else f"{100*x:.0f}%"


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else "-"
    frame = sys.argv[2] if len(sys.argv) > 2 else None
    url = sys.argv[3] if len(sys.argv) > 3 else ""
    R = json.load(open(os.path.join(res, "results.json"))) if res != "-" and os.path.exists(os.path.join(res, "results.json")) else None
    if R:
        for f in ("bars.png", "engagement_hist.png"):
            shutil.copy(os.path.join(res, f), os.path.join(HERE, "assets", f))
    fps = [t["fps"] for t in R["takes"] if t.get("fps")] if R else []
    stats = [("0.864", "gesture macro-F1 (±0.018), 71 held-out subjects"),
             ("0.91", "STOP recall with safety-biased threshold"),
             (f"~{sum(fps)/len(fps):.0f} fps" if fps else "— fps", "full multi-person pipeline, laptop CPU"),
             (str(len(R["takes"])) if R else "9", "crowd scenarios recorded on the Go2")]
    stats_html = "".join(f"<div class='st'><b>{a}</b><span>{b}</span></div>" for a, b in stats)
    rows = ["<table><tr><th></th><th class='n' colspan=2>Stationary</th><th class='n' colspan=2>Walking</th></tr>"
            "<tr><th></th><th class='n'>false trig.</th><th class='n'>recall</th><th class='n'>false trig.</th><th class='n'>recall</th></tr>"]
    for m, nm in NAMES.items():
        cls = " class='hl'" if m == "it2" else ""
        cells = []
        for sc in ("stationary", "walking"):
            if R:
                r = R["table"][f"{sc}/{m}"]
                cells += [pct(r["false_trigger"][0]), pct(r["recall"][0])]
            else:
                cells += ["[ — ]", "[ — ]"]
        rows.append(f"<tr{cls}><td>{nm}</td>" + "".join(f"<td class='n'>{c}</td>" for c in cells) + "</tr>")
    rows.append("</table>")
    if R:
        T = R["table"]
        key = (f"While the Go2 was walking, the Interval Type-2 gate cut bystander false triggers from "
               f"{pct(T['walking/gesture_only']['false_trigger'][0])} (gesture only) to "
               f"{pct(T['walking/it2']['false_trigger'][0])}, keeping {pct(T['walking/it2']['recall'][0])} commander recall.")
        key = os.environ.get("POSTER_KEY", key)
        n_eps = sum(t["robot_eps"] + t["other_eps"] for t in R["takes"])
        fig, figcap = "bars.png", f"Real Go2 runs: {n_eps} hand-labelled gesture episodes. Red = bystander false triggers, blue = commander recall."
    else:
        key = "[One sentence: how much Interval Type-2 cut bystander false triggers while walking]"
        fig, figcap = "fusion_check.png", "Placeholder figure — replaced by tonight's results."
    if frame:
        shutil.copy(frame, os.path.join(HERE, "assets", "poster_frame.png"))
        fr, frcap = "poster_frame.png", "Go2 camera: every person gets a box, a distance, a facing score and an engagement interval. Green = gesture accepted."
    else:
        fr, frcap = "go2_frame.png", "Real Go2 frame: two people in view. Whose gesture should count?"
    html = open(os.path.join(HERE, "poster_template.html")).read()
    for k, v in {"STATS": stats_html, "TABLE": "\n".join(rows), "KEY": key, "FIG": fig, "FIG_CAP": figcap,
                 "FRAME": fr, "FRAME_CAP": frcap, "URL": url}.items():
        html = html.replace("{{" + k + "}}", v)
    out_html = os.path.join(HERE, "poster.html")
    open(out_html, "w").write(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 810})
        pg.goto("file://" + out_html)
        pg.wait_for_timeout(500)
        pg.pdf(path=os.path.join(HERE, "poster.pdf"), width="1440px", height="810px", print_background=True)
        pg.screenshot(path=os.path.join(HERE, "poster.png"))
        b.close()
    print("poster.pdf written")


if __name__ == "__main__":
    main()
