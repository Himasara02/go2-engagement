#!/usr/bin/env python3
"""Find gestures per person by cropping each person and looking for hands in the crop.

In the full frame a hand is only ~30-40 px, so the hand detector misses most of
them. Here every tracked person (boxes from the take's log.jsonl) is cropped from
the full-resolution raw.mp4, enlarged, and the hand detector + gesture classifier
run on the crop. People, IDs and boxes stay exactly as in the log.

    python find_gestures_crop.py takes_v2/D1_S2_r1      # one take
    python find_gestures_crop.py                        # every take in takes_v2/

Writes takes_crop/<take>/ (log.jsonl with the new gestures, meta.json, raw.mp4,
frames.csv) so label_commander.py / label_episodes.py can open it.
perception.py is not changed; this is a labelling aid only.
"""
import glob
import json
import math
import os
import shutil
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from engage.methods import COMMAND_MAP
from engage.perception import MODELS, GestureClassifier
from engage.takeio import proc_size

SRC, OUT = "takes_v2", "takes_crop"
PAD_X, PAD_Y = 0.35, 0.10     # grow each box so raised/outstretched hands are inside
MIN_H = 640                   # enlarge crops to at least this height (px)
CONF = 0.4


def hands_model():
    return mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))


def owner(wx, wy, boxes):
    """Person whose box holds the wrist; if several, the one whose centre is nearest."""
    best, bd = None, 1e9
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        w = x1 - x0
        if x0 - 0.25 * w <= wx <= x1 + 0.25 * w and y0 - 0.05 * (y1 - y0) <= wy <= y1:
            d = abs(wx - 0.5 * (x0 + x1)) / max(w, 1)
            if d < bd:
                bd, best = d, k
    return best


def run(take, hl, clf):
    name = os.path.basename(os.path.normpath(take))
    meta = json.load(open(os.path.join(take, "meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(take, "log.jsonl"))]
    cap = cv2.VideoCapture(os.path.join(take, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    pw, _ = proc_size(W, H)
    s = W / pw                                   # log boxes are in the processing view
    mirror = meta.get("mirror", True)
    n_old = n_new = 0
    for r in recs:
        ok, img = cap.read()
        if not ok:
            break
        if mirror:
            img = cv2.flip(img, 1)               # same orientation as the log (and the classifier)
        boxes = [[v * s for v in p["bbox"]] for p in r["people"]]
        found = []                               # (wrist x, wrist y, label, conf) in full-res pixels
        for x0, y0, x1, y1 in boxes:
            bw, bh = x1 - x0, y1 - y0
            cx0, cy0 = int(max(0, x0 - PAD_X * bw)), int(max(0, y0 - PAD_Y * bh))
            cx1, cy1 = int(min(W, x1 + PAD_X * bw)), int(min(H, y1 + PAD_Y * bh))
            if cx1 - cx0 < 20 or cy1 - cy0 < 20:
                continue
            crop = img[cy0:cy1, cx0:cx1]
            k = max(1.0, MIN_H / crop.shape[0])
            if k > 1:
                crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            ch, cw = crop.shape[:2]
            res = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            for lm in res.hand_landmarks or []:
                pts = [[min(int(p.x * cw), cw - 1), min(int(p.y * ch), ch - 1)] for p in lm]
                g, c = clf(pts)
                wx, wy = cx0 + lm[0].x * cw / k, cy0 + lm[0].y * ch / k
                # the same hand seen in two overlapping crops: keep one
                if any(math.hypot(wx - f[0], wy - f[1]) < 25 for f in found):
                    continue
                found.append((wx, wy, g, c))
        per = {}
        for wx, wy, g, c in found:
            k = owner(wx, wy, boxes)
            if k is not None:
                per.setdefault(k, []).append((g, round(c, 3)))
        for k, p in enumerate(r["people"]):
            if p.get("gesture") in COMMAND_MAP and p.get("gesture_conf", 0) >= CONF:
                n_old += 1
            hs = per.get(k, [])
            # a raised command hand beats a relaxed hand on the other side
            cmd = [h for h in hs if h[0] in COMMAND_MAP and h[1] >= CONF]
            best = max(cmd or hs, key=lambda h: h[1]) if hs else (None, 0.0)
            p["gesture_full_frame"], p["gesture_conf_full_frame"] = p.get("gesture"), p.get("gesture_conf")
            p["gesture"], p["gesture_conf"] = best[0], float(best[1])
            p["hands"] = hs
            if best[0] in COMMAND_MAP and best[1] >= CONF:
                n_new += 1
    cap.release()
    dst = os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "log.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    for fn in ("raw.mp4", "frames.csv"):
        if os.path.exists(os.path.join(take, fn)):
            shutil.copy2(os.path.join(take, fn), os.path.join(dst, fn))
    meta.update(gestures_from="person crops (find_gestures_crop.py)", crop_source=take.replace("\\", "/"))
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    print(f"{name}: command-gesture person-frames  full frame {n_old}  ->  crops {n_new}", flush=True)


def main():
    takes = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(SRC, "*")) if os.path.isdir(d))
    takes = [t for t in takes if os.path.exists(os.path.join(t, "log.jsonl"))
             and json.load(open(os.path.join(t, "meta.json"))).get("scenario") not in ("calibration", "test")]
    hl, clf = hands_model(), GestureClassifier()
    for t in takes:
        run(t, hl, clf)


if __name__ == "__main__":
    main()
