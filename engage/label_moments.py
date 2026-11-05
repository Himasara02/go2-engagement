#!/usr/bin/env python3
"""Faster labelling: one question per moment - WHO is commanding the dog?

Gestures that happen at the same time are grouped into one "moment". The clip
plays once and then freezes. Everyone who is making a command gesture has an
ORANGE box with a big #ID; other people have thin blue boxes.

Type the ID(s) of the person/people gesturing TO THE DOG, then press Enter:
    3          -> #3 is commanding the dog, everyone else in this moment is not
    3 5        -> both #3 and #5 command the dog (scenario S5)
    0          -> nobody in this moment is commanding the dog
    3 x5       -> #3 commands the dog; #5's box is a bad detection
    Backspace  -> fix typing      p -> play the clip again
    b          -> back one moment q -> save and quit (continue later)

Saves the same files as label_episodes.py (labels.json / labels_B.json), so
check_dataset.py and the analysis work unchanged.

    python label_moments.py takes/D1_S2_r1                 # annotator A
    python label_moments.py takes/D1_S2_r1 --annotator B   # annotator B
"""
import argparse
import json
import os

import cv2
import numpy as np

from label_episodes import Clips, episodes

GAP = 6          # gestures closer than this many frames belong to the same moment
ORANGE, BLUE, WHITE = (0, 140, 255), (255, 200, 0), (255, 255, 255)


def build_moments(eps, gap=GAP):
    """Group episodes that overlap in time into moments (lists of episodes)."""
    moments, cur, cur_end = [], [], -10 ** 9
    for e in sorted(eps, key=lambda e: e["start"]):
        if cur and e["start"] <= cur_end + gap:
            cur.append(e)
            cur_end = max(cur_end, e["end"])
        else:
            if cur:
                moments.append(cur)
            cur, cur_end = [e], e["end"]
    if cur:
        moments.append(cur)
    return moments


def parse_answer(text, moment):
    """'3 x5' -> (labels dict ep_id->r/n/x, error or None)."""
    tids = {e["tid"] for e in moment}
    robot, bad = set(), set()
    toks = text.replace(",", " ").split()
    if not toks:
        return None, "type the ID(s), or 0 for nobody"
    for t in toks:
        is_bad = t.lower().startswith("x")
        t = t[1:] if is_bad else t
        if not t.isdigit():
            return None, f"'{t}' is not a number"
        n = int(t)
        if n == 0 and not is_bad:
            continue
        if n not in tids:
            return None, f"#{n} is not an orange box here (orange: {', '.join('#%d' % i for i in sorted(tids))})"
        (bad if is_bad else robot).add(n)
    if robot & bad:
        return None, "the same ID is both 'robot' and 'bad'"
    out = {}
    for e in moment:
        out[str(e["id"])] = "x" if e["tid"] in bad else ("r" if e["tid"] in robot else "n")
    return out, None


def current_answer(labels, moment):
    """Rebuild the typed answer from saved labels (for going back)."""
    if not all(str(e["id"]) in labels for e in moment):
        return ""
    r = sorted({e["tid"] for e in moment if labels[str(e["id"])] == "r"})
    x = sorted({e["tid"] for e in moment if labels[str(e["id"])] == "x"})
    return " ".join([str(i) for i in r] + ["x%d" % i for i in x]) or "0"


def draw(img, recs, frame_i, moment, typed, msg, k, n):
    gest = {e["tid"]: e for e in moment}
    people = recs[frame_i]["people"] if frame_i < len(recs) else []
    shown = set()
    for p in people:
        x0, y0, x1, y1 = p["bbox"]
        if p["tid"] in gest:
            shown.add(p["tid"])
            cv2.rectangle(img, (x0, y0), (x1, y1), ORANGE, 3)
            t = f"#{p['tid']} {gest[p['tid']]['gesture']}"
            cv2.putText(img, t, (x0 + 3, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(img, t, (x0 + 3, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 1)
            cv2.putText(img, f"#{p['tid']}", (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLUE, 1)
    # gesturing people not visible on this exact frame: draw their box from their own gesture
    for tid, e in gest.items():
        if tid in shown:
            continue
        mid = e["frames"][len(e["frames"]) // 2]
        for p in (recs[mid]["people"] if mid < len(recs) else []):
            if p["tid"] == tid:
                x0, y0, x1, y1 = p["bbox"]
                cv2.rectangle(img, (x0, y0), (x1, y1), ORANGE, 2)
                cv2.putText(img, f"#{tid} {e['gesture']}", (x0 + 3, y0 + 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2)
    lines = [f"moment {k + 1}/{n}   WHO is gesturing TO THE DOG?  Type ID(s), then Enter   (0 = nobody)",
             f"> {typed}_", msg or "p = replay    b = back    q = save & quit    x5 = #5 is a bad box"]
    band = np.zeros((96, img.shape[1], 3), np.uint8)
    for j, s_ in enumerate(lines):
        col = (0, 0, 255) if (j == 2 and msg) else ((0, 255, 255) if j == 1 else WHITE)
        cv2.putText(band, s_, (10, 26 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if j != 1 else 0.8, col,
                    2 if j == 1 else 1, cv2.LINE_AA)
    return np.vstack([band, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--annotator", default="A", choices=["A", "B", "C"])
    a = ap.parse_args()
    recs, eps = episodes(a.take)
    moments = build_moments(eps)
    path = os.path.join(a.take, "labels.json" if a.annotator == "A" else f"labels_{a.annotator}.json")
    labels = json.load(open(path)) if os.path.exists(path) else {}
    clips = Clips(a.take)
    print(f"{a.take}: {len(eps)} gestures in {len(moments)} moments (annotator {a.annotator})")
    k = next((i for i, m in enumerate(moments) if not current_answer(labels, m)), len(moments))
    while 0 <= k < len(moments):
        m = moments[k]
        start, end = min(e["start"] for e in m), max(e["end"] for e in m)
        a0, b0 = max(0, start - 8), min(clips.n - 1, end + 4)
        clip = clips.clip(a0, b0)
        if not clip:
            k += 1
            continue
        freeze_i = (start + end) // 2
        typed, msg, play = current_answer(labels, m), "", True
        while True:
            if play:
                for j, fr in enumerate(clip):
                    img = draw(fr.copy(), recs, a0 + j, m, typed, msg, k, len(moments))
                    cv2.imshow("who is commanding the dog?", img)
                    cv2.waitKey(70)
                play = False
            img = draw(clip[min(len(clip) - 1, max(0, freeze_i - a0))].copy(), recs, freeze_i, m,
                       typed, msg, k, len(moments))
            cv2.imshow("who is commanding the dog?", img)
            key = cv2.waitKey(0) & 0xFF
            msg = ""
            if key in (13, 10):                       # Enter
                out, err = parse_answer(typed, m)
                if err:
                    msg = err
                    continue
                labels.update(out)
                json.dump(labels, open(path, "w"), indent=0)
                k += 1
                break
            if key in (8, 127):                       # Backspace
                typed = typed[:-1]
            elif chr(key).isdigit() or chr(key) in " ,":
                typed += chr(key)
            elif chr(key) in "xX":
                typed += "x"
            elif chr(key) == "p":
                play = True
            elif chr(key) == "b":
                k = max(0, k - 1)
                break
            elif chr(key) in "qQ" or key == 27:
                k = -1
                break
    cv2.destroyAllWindows()
    done = all(str(e["id"]) in labels for e in eps)
    if done and "_missed" not in labels:
        print("\nAll moments labelled. One last question about the WHOLE take:")
        print("How many times did someone clearly gesture TO THE DOG that never appeared with an orange box?")
        ans = input("Number (just press Enter for 0; type ? if you are not sure): ").strip()
        labels["_missed"] = 0 if ans == "" else (int(ans) if ans.isdigit() else ans)
    json.dump(labels, open(path, "w"), indent=0)
    left = len([e for e in eps if str(e["id"]) not in labels])
    print(f"saved {path}: {len(eps) - left}/{len(eps)} gestures labelled"
          + (f"  ({left} left - run again to continue)" if left else ""))


if __name__ == "__main__":
    main()
