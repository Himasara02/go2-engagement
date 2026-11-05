#!/usr/bin/env python3
"""Label a take by choosing the commander once.

Every tracked person who made a command gesture gets one picture (their box in
orange, big #ID, and the gestures they made). Click the picture(s) that show
the COMMANDER - track IDs can change during a take, so the commander may appear
as several IDs; click all of them. Then every gesture by a chosen ID is saved as
"for the dog" (r) and every gesture by anyone else as "not for the dog" (n).

    click = choose / unchoose      or type the ID number, then Enter
    v = play the take with all IDs n / b = next / previous page
    Enter (nothing typed) = save   q = quit without saving

    python label_commander.py takes_v2/D1_S2_r1        -> labels_cmd.json

Saves to labels_cmd.json, so labels.json (A) and labels_B.json (B) are not touched.
Note: in S4 the commander also gestures to a friend; this tool marks those r too.
"""
import argparse
import json
import os

import cv2
import numpy as np

from label_episodes import Clips, episodes

TW, COLS, ROWS = 320, 4, 3
ORANGE, GREEN, BLUE = (0, 140, 255), (0, 220, 0), (255, 200, 0)


def tile(clips, recs, tid, eps):
    """Picture of one person: frame from the middle of their longest gesture."""
    e = max(eps, key=lambda e: len(e["frames"]))
    i = e["frames"][len(e["frames"]) // 2]
    fr = clips.clip(i, i)
    img = fr[0].copy() if fr else np.zeros((540, 960, 3), np.uint8)
    for p in recs[i]["people"] if i < len(recs) else []:
        x0, y0, x1, y1 = p["bbox"]
        if p["tid"] == tid:
            cv2.rectangle(img, (x0, y0), (x1, y1), ORANGE, 5)
            cv2.putText(img, f"#{tid}", (x0 + 5, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
            cv2.putText(img, f"#{tid}", (x0 + 5, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, ORANGE, 3)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 1)
    img = cv2.resize(img, (TW, int(img.shape[0] * TW / img.shape[1])))
    gest = {}
    for x in eps:
        gest[x["gesture"]] = gest.get(x["gesture"], 0) + 1
    txt = f"#{tid}: " + " ".join(f"{g}x{c}" for g, c in gest.items())
    band = np.zeros((26, TW, 3), np.uint8)
    cv2.putText(band, txt[:40], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def play(clips, recs):
    """Whole take with every ID drawn, so you can follow who is who (q/Esc stops)."""
    step = 2
    for a in range(0, clips.n, 50):
        for j, fr in enumerate(clips.clip(a, min(clips.n - 1, a + 49))):
            if j % step:
                continue
            i = a + j
            for p in recs[i]["people"] if i < len(recs) else []:
                x0, y0, x1, y1 = p["bbox"]
                cv2.rectangle(fr, (x0, y0), (x1, y1), ORANGE, 2)
                cv2.putText(fr, f"#{p['tid']}", (x0 + 3, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, ORANGE, 2)
            cv2.putText(fr, f"frame {i}/{clips.n}   q = stop", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow("take", fr)
            if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                cv2.destroyWindow("take")
                return
    cv2.destroyWindow("take")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    a = ap.parse_args()
    recs, eps = episodes(a.take)
    path = os.path.join(a.take, "labels_cmd.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    by_tid = {}
    for e in eps:
        by_tid.setdefault(e["tid"], []).append(e)
    tids = sorted(by_tid, key=lambda t: by_tid[t][0]["start"])
    print(f"{a.take}: {len(eps)} gestures by {len(tids)} IDs")
    if not eps:
        print("no gestures in this take - nothing to label")
        return
    if "_S4_" in os.path.basename(os.path.normpath(a.take)):
        print("WARNING: S4 - the commander's gestures to a friend will be saved as r too.")
    clips = Clips(a.take)
    tiles = {t: tile(clips, recs, t, by_tid[t]) for t in tids}
    th = max(x.shape[0] for x in tiles.values())
    chosen = set(old.get("_commander_tids", []))
    per = COLS * ROWS
    state = {"page": 0, "click": None}
    typed, msg = "", ""

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            print(f"  click at {x},{y}", flush=True)
            if y > 60:
                state["click"] = (x // TW, (y - 60) // th)

    win = "click the COMMANDER (all their IDs), then Enter"
    # resizable window that fits the screen (with Windows display scaling a fixed-size
    # window can be larger than the screen); OpenCV maps clicks back to image pixels
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1000, int(1000 * (60 + ROWS * th) / (COLS * TW)))
    cv2.setMouseCallback(win, on_mouse)
    while True:
        page_tids = tids[state["page"] * per:(state["page"] + 1) * per]
        if state["click"]:
            c, r = state["click"]
            state["click"] = None
            k = r * COLS + c
            if c < COLS and k < len(page_tids):
                chosen ^= {page_tids[k]}
                print(f"  #{page_tids[k]} {'chosen' if page_tids[k] in chosen else 'unchosen'}", flush=True)
        canvas = np.zeros((60 + ROWS * th, COLS * TW, 3), np.uint8)
        pages = (len(tids) - 1) // per + 1
        cv2.putText(canvas, f"Click the COMMANDER (every ID that is them). Chosen: "
                    f"{' '.join('#%d' % t for t in sorted(chosen)) or 'none'}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, msg or f"Type ID + Enter = choose   Enter alone = SAVE   page {state['page'] + 1}/"
                    f"{pages} (n/b)   v = play   q = quit      > {typed}_", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255) if msg else (0, 255, 255), 1, cv2.LINE_AA)
        for k, t in enumerate(page_tids):
            r, c = divmod(k, COLS)
            im = tiles[t]
            y0, x0 = 60 + r * th, c * TW
            canvas[y0:y0 + im.shape[0], x0:x0 + TW] = im
            if t in chosen:
                cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + TW - 3, y0 + im.shape[0] - 3), GREEN, 5)
                cv2.putText(canvas, "COMMANDER", (x0 + 10, y0 + im.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key == 255:
            continue
        msg = ""
        if key in (13, 10):
            if not typed:
                break
            t = int(typed)
            typed = ""
            if t in by_tid:
                chosen ^= {t}
                print(f"  #{t} {'chosen' if t in chosen else 'unchosen'}", flush=True)
            else:
                msg = f"#{t} has no gestures here - IDs: {' '.join('#%d' % i for i in tids)}"
            continue
        if chr(key).isdigit():
            typed += chr(key)
            continue
        if key in (8, 127):
            typed = typed[:-1]
            continue
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            print("quit - nothing saved")
            return
        if key == ord("n"):
            state["page"] = min(pages - 1, state["page"] + 1)
        elif key == ord("b"):
            state["page"] = max(0, state["page"] - 1)
        elif key == ord("v"):
            play(clips, recs)
    cv2.destroyAllWindows()
    labels = {str(e["id"]): ("r" if e["tid"] in chosen else "n") for e in eps}
    labels["_commander_tids"] = sorted(chosen)
    json.dump(labels, open(path, "w"), indent=0)
    nr = sum(v == "r" for k, v in labels.items() if not k.startswith("_"))
    print(f"saved {path}: commander {' '.join('#%d' % t for t in sorted(chosen)) or 'none'} -> "
          f"{nr} for the dog (r), {len(eps) - nr} not for the dog (n)")


if __name__ == "__main__":
    main()
