#!/usr/bin/env python3
"""Overlay clips for the supplementary video: the same face-blurred, levelled clip as video_clip.py, with
per-person decisions drawn on top, once for the gesture-only policy and once for our IT2 + gaze gate.

    ~/miniconda3/envs/go2/bin/python video_overlay.py TAKE F0 F1 OUT_DIR

Decisions are recomputed exactly as in full_eval.py (paper numbers): candidate frame = command gesture with
confidence >= theta_c; gesture-only executes after 3 consecutive candidate frames; our gate additionally needs
E_low (IT2 + gaze, RULES4) + 0.15 * STOP >= theta_E motion on those 3 frames. A decision stays on screen for
0.5 s (the robot's stop timeout). Box colour = annotated addressee (blue: robot, pink: someone else);
the text says what the policy does. Robot line: STOP wins, otherwise the most engaged (ours) or the most
confident (gesture only) person.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from full_eval import LABELS, OFF, P, RULES4, TAKES, Data, fls   # noqa: E402
from engage.methods import COMMAND_MAP                                # noqa: E402
from full_cues import models                                          # noqa: E402
from video_clip import OUT_H, OUT_W, ROT, frames, times               # noqa: E402

CMD = {"palm": "STOP", "stop": "STOP", "fist": "FORWARD", "stop_inverted": "FORWARD", "call": "BACK",
       "mute": "BACK", "one": "LEFT", "two_up": "RIGHT"}
BLUE, PINK, WHITE, INK = (35, 145, 255), (247, 0, 99), (255, 255, 255), (26, 26, 26)
FONT = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
F = {k: ImageFont.truetype(FONT.format(w), s) for k, (w, s) in
     {"tag": ("Bold", 22), "sub": ("Medium", 18), "ban": ("Bold", 30), "leg": ("Medium", 19), "rob": ("Bold", 24)}.items()}
HOLD_S = 0.5


def decisions(take, f0, f1):
    """{frame: {tid: dict(gesture, conf, label, E, go, ours)}} for frames f0..f1."""
    rows = [r for r in csv.DictReader(open(LABELS)) if r["take"] == take and r["label"] in ("r", "n")]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    w = csv.DictWriter(tmp, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
    tmp.close()
    d = Data(labels=tmp.name)
    os.unlink(tmp.name)
    X = d.X()
    recs = [json.loads(l) for l in open(os.path.join(TAKES, take, "log.jsonl"))]
    t = [r["t"] for r in recs]
    out = {f: {} for f in range(f0, f1 + 1)}
    last = {}                                    # (policy, tid) -> (time, info) of last sustained frame
    per = {}                                     # frame -> list of (tid, idx, label)
    for r, e in zip(rows, d.eps):
        a = int(r["first_frame"])
        for k, i in enumerate(e["idx"]):
            per.setdefault(a + k, []).append((int(r["track_id"]), e, k))
    E = {}

    def eng(i):
        if i not in E:
            E[i] = fls(X[i], True, 1.0, RULES4)
        return E[i]
    for f in range(f0 - 3, f1 + 1):
        for tid, e, k in per.get(f, []):
            w3 = e["idx"][k - 2:k + 1] if k >= 2 else []
            if len(w3) < 3 or None in w3 or not all(d.cand[i] for i in w3):
                continue
            i = e["idx"][k]
            p = next(q for q in recs[f]["people"] if q["tid"] == tid)
            info = {"gesture": p["gesture"], "conf": p["gesture_conf"], "label": e["label"], "E": eng(i)}
            last[("go", tid)] = (t[f], info)
            if all(eng(j) + OFF * d.stop[j] >= P.theta_e_motion for j in w3):
                last[("ours", tid)] = (t[f], info)
            else:
                last[("rej", tid)] = (t[f], info)
        if f < f0:
            continue
        for (pol, tid), (tt, info) in list(last.items()):
            if t[f] - tt <= HOLD_S:
                out[f].setdefault(tid, {}).setdefault(pol, info)
    return out


def box_map(take, crop, W=1280, H=720):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ROT, 1.0)
    x0, y0, x1, y1 = crop
    s = OUT_W / (x1 - x0)

    def f(b):
        cx, cy = M @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
        w, h = b[2] - b[0], b[3] - b[1]
        return [(cx - w / 2 - x0) * s, (cy - h / 2 - y0) * s, (cx + w / 2 - x0) * s, (cy + h / 2 - y0) * s]
    return f


def pill(dr, xy, text, font, fill, fg=WHITE, pad=(10, 5)):
    x, y = xy
    l, t_, r, b = dr.textbbox((x + pad[0], y + pad[1]), text, font=font)
    dr.rounded_rectangle((x, y, r + pad[0], b + pad[1] + 2), radius=8, fill=fill)
    dr.text((x + pad[0], y + pad[1]), text, font=font, fill=fg)
    return r + pad[0], b + pad[1] + 2


def draw(img_bgr, boxes, dec, policy, bmap):
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    executed = []
    for tid, b in boxes.items():
        x0, y0, x1, y1 = bmap(b)
        x0, x1 = max(4, x0), min(OUT_W - 4, x1)
        y0, y1 = max(4, y0), min(OUT_H - 4, y1)
        st = dec.get(tid, {})
        if policy == "go":
            info, ok = st.get("go"), "go" in st
        else:
            info = st.get("ours") or st.get("rej")
            ok = "ours" in st
        if info is None:
            dr.rectangle((x0, y0, x1, y1), outline=WHITE + (150,), width=2)
            continue
        col = BLUE if info["label"] == "r" else PINK
        dr.rectangle((x0, y0, x1, y1), outline=col + (255,), width=5)
        cmd = CMD.get(info["gesture"], info["gesture"])
        tag = f"{cmd} → {'executed' if ok else 'ignored'}"
        yb = min(y1, OUT_H - 70) - 64
        bx, by = pill(dr, (x0 + 6, yb), tag, F["tag"], col + (235,))
        sub = "to the robot" if info["label"] == "r" else "to a friend"
        if policy != "go":
            sub += f"   E = {info['E']:.2f}"
        pill(dr, (x0 + 6, by + 2), sub, F["sub"], col + (200,))
        if ok:
            executed.append((cmd, info))
    # banner
    ban = "Gesture-only robot: obeys every hand" if policy == "go" else "Our addressee gate (IT2 + gaze)"
    pill(dr, (16, 14), ban, F["ban"], INK + (210,), pad=(14, 8))
    # robot line
    if not executed:
        rob = "Robot: no command"
    elif policy == "go":
        cmds = sorted({c for c, _ in executed})
        rob = (f"Robot obeys {len(executed)} people: {' + '.join(c for c, _ in executed)}" if len(executed) > 1
               else "Robot: " + cmds[0])
    else:
        stops = [c for c, _ in executed if c == "STOP"]
        rob = "Robot: " + ("STOP" if stops else max(executed, key=lambda z: z[1]["E"])[0])
    pill(dr, (16, OUT_H - 58), rob, F["rob"], INK + (210,), pad=(14, 8))
    leg = "blue = gesture for the robot    pink = gesture to someone else"
    lw = dr.textlength(leg, font=F["leg"])
    pill(dr, (OUT_W - lw - 44, 18), leg, F["leg"], INK + (170,), pad=(14, 6))
    return cv2.cvtColor(np.array(Image.alpha_composite(im, ov).convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    take, f0, f1, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    dec = decisions(take, f0, f1)
    t = times(take)
    pose, _ = models()
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lists = {"go": [], "ours": []}
        for f, img, crop, boxes in frames(take, f0, f1, pose):
            bmap = box_map(take, crop)
            dur = (t[f + 1] - t[f]) if f < f1 else (t[f] - t[f - 1])
            for pol in lists:
                p = os.path.join(tmp, f"{pol}_{f:05d}.png")
                cv2.imwrite(p, draw(img, boxes, dec[f], pol, bmap))
                lists[pol].append(f"file '{p}'\nduration {dur:.4f}\n")
        for pol, name in (("go", "gesture_only"), ("ours", "ours")):
            lists[pol].append(lists[pol][-1].split("\n")[0] + "\n")
            lp = os.path.join(tmp, f"{pol}.txt")
            open(lp, "w").writelines(lists[pol])
            out = os.path.join(out_dir, f"clipA_{take}_overlay_{name}.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lp, "-vf",
                            "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                            "-movflags", "+faststart", "-an", out], check=True)
            print("wrote", out)


if __name__ == "__main__":
    main()
