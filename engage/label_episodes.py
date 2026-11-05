#!/usr/bin/env python3
"""Label every gesture episode in a take: was it meant for the robot?

An episode = one tracked person holding one command gesture (conf >= 0.4)
for >= 3 frames (gaps <= 4 frames merged). For each episode a short clip of
the raw video plays with that person boxed in red. Press:
   r = the gesture WAS meant for the robot
   n = NOT meant for the robot (to another person, or nobody)
   x = bad detection (not really a gesture / wrong person / no hand)
   SPACE = replay the clip      b = back one      q = save and quit
The video shown has no system decisions on it, so labels are not biased.

Two people label every take separately:
   python label_episodes.py takes/D1_S2_r1                  -> labels.json   (annotator A)
   python label_episodes.py takes/D1_S2_r1 --annotator B    -> labels_B.json (annotator B)
Do not look at each other's labels.
"""
import argparse
import json
import os

import cv2

from engage.methods import COMMAND_MAP
from engage.takeio import to_proc_view

MIN_LEN, MAX_GAP, CONF = 3, 4, 0.4


def episodes(take_dir):
    recs = [json.loads(l) for l in open(os.path.join(take_dir, "log.jsonl"))]
    open_ep, done = {}, []
    for r in recs:
        seen = set()
        for p in r["people"]:
            g = p.get("gesture")
            if g in COMMAND_MAP and p["gesture_conf"] >= CONF:
                key = p["tid"]
                seen.add(key)
                ep = open_ep.get(key)
                if ep and ep["gesture"] == g and r["i"] - ep["end"] <= MAX_GAP:
                    ep["end"] = r["i"]
                    ep["frames"].append(r["i"])
                else:
                    if ep:
                        done.append(ep)
                    open_ep[key] = {"tid": key, "gesture": g, "start": r["i"],
                                    "end": r["i"], "frames": [r["i"]]}
        for key in list(open_ep):
            if key not in seen and r["i"] - open_ep[key]["end"] > MAX_GAP:
                done.append(open_ep.pop(key))
    done += list(open_ep.values())
    done = [e for e in done if len(e["frames"]) >= MIN_LEN]
    done.sort(key=lambda e: e["start"])
    for k, e in enumerate(done):
        e["id"] = k
    return recs, done


class Clips:
    """Reads short clips from raw.mp4 on demand (whole takes don't fit in RAM)."""

    def __init__(self, take_dir):
        self.meta = json.load(open(os.path.join(take_dir, "meta.json")))
        self.new_format = self.meta.get("raw_format", 1) == 2
        self.cap = cv2.VideoCapture(os.path.join(take_dir, "raw.mp4"))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def clip(self, a, b):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        out = []
        for _ in range(a, b + 1):
            ok, img = self.cap.read()
            if not ok:
                break
            if self.new_format:             # same view as the logged boxes
                img, _ = to_proc_view(img, None, mirror=self.meta.get("mirror", True))
            out.append(img)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--annotator", default="A", choices=["A", "B", "C"])
    args = ap.parse_args()
    take = args.take
    recs, eps = episodes(take)
    name = "labels.json" if args.annotator == "A" else f"labels_{args.annotator}.json"
    path = os.path.join(take, name)
    labels = json.load(open(path)) if os.path.exists(path) else {}
    clips = Clips(take)
    print(f"{len(eps)} clips in {take}  (annotator {args.annotator}, {len([k for k in labels if not k.startswith('_')])} already labelled)")
    print("Decide ONLY about the person in the RED box: was THIS gesture meant for the dog?")
    k = 0
    while 0 <= k < len(eps):
        e = eps[k]
        a, b = max(0, e["start"] - 8), min(clips.n - 1, e["end"] + 4)
        clip = clips.clip(a, b)
        key = None
        while key is None:
            for j, frame in enumerate(clip):
                i = a + j
                img = frame.copy()
                for p in recs[i]["people"] if i < len(recs) else []:
                    x0, y0, x1, y1 = p["bbox"]
                    me = p["tid"] == e["tid"]
                    col = (0, 0, 255) if me else (255, 200, 0)
                    cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 if me else 2)
                    cv2.putText(img, f"#{p['tid']}", (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                cur = labels.get(str(e["id"]), "?")
                cv2.putText(img, f"{k+1}/{len(eps)}  red person: {e['gesture']}  label={cur}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(img, f"{k+1}/{len(eps)}  red person: {e['gesture']}  label={cur}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(img, "r=for robot  n=not for robot  x=bad  SPACE=replay  b=back  q=quit",
                            (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow("label", img)
                kk = cv2.waitKey(70) & 0xFF
                if kk != 255:
                    key = kk
                    break
            if key is None:
                kk = cv2.waitKey(0) & 0xFF
                key = None if kk == ord(" ") else kk
        if key in (ord("r"), ord("n"), ord("x")):
            labels[str(e["id"])] = chr(key)
            k += 1
        elif key == ord("b"):
            k = max(0, k - 1)
        elif key in (ord("q"), 27):
            break
        json.dump(labels, open(path, "w"), indent=0)
    cv2.destroyAllWindows()
    done = all(str(e["id"]) in labels for e in eps)
    if done and "_missed" not in labels:
        # gestures the system never detected cannot become clips - count them so recall stays honest
        print("\nAll clips labelled. One last question about the WHOLE take:")
        print("How many times did someone clearly gesture TO THE DOG that you never saw as a red-box clip?")
        ans = input("Number (just press Enter for 0; type ? if you are not sure): ").strip()
        labels["_missed"] = 0 if ans == "" else (ans if ans == "?" else int(ans) if ans.isdigit() else ans)
    json.dump(labels, open(path, "w"), indent=0)
    left = len([e for e in eps if str(e["id"]) not in labels])
    print(f"saved {path}: {len(eps) - left}/{len(eps)} labelled" + (f"  ({left} left - run again to finish)" if left else ""))


if __name__ == "__main__":
    main()
