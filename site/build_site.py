#!/usr/bin/env python3
"""Fill template.html from results.json + videos -> dist/.
Usage: python build_site.py <results_dir> <takes_dir_or_videos_dir> [stats.json]
"""
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
NAMES = {"gesture_only": "Gesture only", "nearest": "Nearest person",
         "type1": "+ Type-1 engagement", "it2": "+ Interval Type-2"}
CAPS = {"S1": "Single commander", "S2": "Bystanders gesturing to each other",
        "S3": "Simultaneous gestures", "S4": "Commander turns to a friend",
        "S5": "Two people command at once", "S6": "Bystanders walking past",
        "W1": "Walking: commander drives the dog", "W2": "Walking: bystander gestures nearby",
        "W3": "Walking: occlusion"}


def pct(x):
    return "–" if x != x else f"{100*x:.0f}%"


def main():
    res_dir, vid_dir = sys.argv[1], sys.argv[2]
    extra = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
    os.makedirs(os.path.join(DIST, "assets"), exist_ok=True)
    for f in glob.glob(os.path.join(HERE, "assets", "*")):
        shutil.copy(f, os.path.join(DIST, "assets"))
    for f in ("bars.png", "engagement_hist.png"):
        if os.path.exists(os.path.join(res_dir, f)):
            shutil.copy(os.path.join(res_dir, f), os.path.join(DIST, "assets", f))
    rpath = os.path.join(res_dir, "results.json")
    R = json.load(open(rpath)) if os.path.exists(rpath) else None
    if R is None:
        return build_pending(vid_dir)
    T = R["table"]

    rows = ["<table><tr><th>Policy</th><th>Stationary · false triggers</th><th>Stationary · recall</th>"
            "<th>Walking · false triggers</th><th>Walking · recall</th></tr>"]
    for m in NAMES:
        cls = ' class="hl"' if m == "it2" else ""
        cells = []
        for sc in ("stationary", "walking"):
            r = T[f"{sc}/{m}"]
            cells.append(f"{pct(r['false_trigger'][0])} <span class='aff'>({r['false_n'][0]}/{r['false_n'][1]})</span>")
            cells.append(f"{pct(r['recall'][0])} <span class='aff'>({r['recall_n'][0]}/{r['recall_n'][1]})</span>")
        rows.append(f"<tr{cls}><td>{NAMES[m]}</td><td>" + "</td><td>".join(cells) + "</td></tr>")
    rows.append("</table>")

    n_eps = sum(t["robot_eps"] + t["other_eps"] for t in R["takes"])
    mins = sum(R["minutes"].values())
    fps = [t["fps"] for t in R["takes"] if t.get("fps")]
    stats = [(str(len(R["takes"])), "takes recorded on the Go2"),
             (str(n_eps), "labelled gesture episodes"),
             (f"{mins:.0f} min", "of real-robot video"),
             (f"~{sum(fps)/len(fps):.0f} fps" if fps else "–", "full pipeline on a laptop CPU")]
    stats.extend(extra.get("stats", []))
    stats_html = "".join(f"<div class='stat'><b>{a}</b><span>{b}</span></div>" for a, b in stats[:4])

    key = extra.get("key")
    if not key:
        g = T["walking/gesture_only"]["false_trigger"][0]
        i2 = T["walking/it2"]["false_trigger"][0]
        rc = T["walking/it2"]["recall"][0]
        key = (f"While the robot was walking, the IT2 gate cut bystander false triggers from "
               f"{pct(g)} (gesture only) to {pct(i2)}, while still executing {pct(rc)} of the commander's gestures.")

    vids, hero = [], None
    order = ["W1", "S3", "S4", "S2", "W2", "S5", "W3", "S1", "S6"]
    found = {}
    for p in glob.glob(os.path.join(vid_dir, "**", "annot.mp4"), recursive=True) + \
            glob.glob(os.path.join(vid_dir, "*.mp4")):
        name = os.path.basename(os.path.dirname(p)) if p.endswith("annot.mp4") else os.path.splitext(os.path.basename(p))[0]
        found.setdefault(name, p)
    for name in sorted(found, key=lambda n: (order.index(n[:2]) if n[:2] in order else 99, n)):
        out = f"take_{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", found[name], "-t", "40",
                        "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                        os.path.join(DIST, "assets", out)], check=True)
        hero = hero or out
        cap = CAPS.get(name[:2], name)
        vids.append(f"<figure><video src='assets/{out}' controls muted playsinline preload='metadata'></video>"
                    f"<figcaption><b>{name}</b> · {cap}</figcaption></figure>")

    write_html(stats_html, "\n".join(rows), key, vids, hero, found)


def take_cards(vids_html, found):
    cards = list(vids_html)
    for k, cap in CAPS.items():
        if not any(n.startswith(k) for n in found):
            cards.append(f"<figure><video onerror=\"ph(this)\" data-ph=\"{k}\" data-sub=\"{cap}\" "
                         f"src='assets/take_{k}.mp4' controls muted playsinline preload='metadata'></video>"
                         f"<figcaption><b>{k}</b> · {cap}</figcaption></figure>")
    return cards


def write_html(stats_html, table, key, vids, hero, found):
    team = os.path.join(HERE, "assets", "team.jpg")
    team_html = "<img onerror=\"ph(this)\" data-ph=\"Team photo\" src='assets/team.jpg' alt='The team'>"
    hero_v = hero or "vid_operator_lock.mp4"
    hero_p = "go2_frame.png" if hero else "poster_operator.jpg"
    html = open(os.path.join(HERE, "template.html")).read()
    html = (html.replace("{{TABLE}}", table).replace("{{STATS}}", stats_html)
            .replace("{{KEY}}", key).replace("{{VIDEOS}}", "\n".join(take_cards(vids, found)))
            .replace("{{HERO_VIDEO}}", hero_v).replace("{{HERO_POSTER}}", hero_p)
            .replace("{{TEAM_PHOTO}}", team_html).replace("{{FIGS_START}}", "")
            .replace("{{HERO_CAP}}", "The Go2's own front camera. Green = gesture accepted, orange = gesture seen but rejected "
                     "(person not engaged). Each box shows distance, facing score and the IT2 engagement interval." if hero else
                     "First prototype: a single operator locks on with an open palm. Tonight's multi-person runs on the Go2 replace this video."))
    if not os.path.exists(os.path.join(DIST, "assets", "bars.png")):
        html = html.replace('<img src="assets/bars.png" alt="false triggers vs recall" style="background:#fff">',
                            "<div class='ph'><b>Figure</b>false triggers vs recall per policy</div>")
        html = html.replace('<img src="assets/engagement_hist.png" alt="engagement histogram" style="background:#fff">',
                            "<div class='ph'><b>Figure</b>engagement of robot-directed vs other gestures</div>")
    open(os.path.join(DIST, "index.html"), "w").write(html)
    if os.path.exists(os.path.join(HERE, "poster.pdf")):
        shutil.copy(os.path.join(HERE, "poster.pdf"), DIST)
    print("built", DIST, "videos:", len(vids))


def build_pending(vid_dir):
    for f in ("bars.png", "engagement_hist.png"):
        src = os.path.join(HERE, "assets", f)
        if not os.path.exists(src):
            pass
    stats_html = "".join(f"<div class='stat'><b>{a}</b><span>{b}</span></div>" for a, b in [
        ("9", "crowd scenarios on the Go2"), ("4", "policies compared on identical inputs"),
        ("18", "gesture classes, 5 commands"), ("0.3 m/s", "speed cap, kill switch held")])
    table = "<div class='ph' style='aspect-ratio:auto;padding:40px'><b>Results table</b>being recorded on the Go2 tonight</div>"
    key = "Results from tonight's real-robot runs will appear here."
    write_html(stats_html, table, key, [], None, {})
    print("built pending site")


if __name__ == "__main__":
    main()
