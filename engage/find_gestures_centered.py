#!/usr/bin/env python3
"""Find gestures one person at a time, each person centred in their own crop.

For every tracked person in every frame (boxes from the take's log.jsonl):
  1. cut a SQUARE crop centred on the person's upper body (head to just below the
     hips, where command gestures happen), with a margin on every side (black
     padding at the frame edge, so the person always stays in the centre),
  2. enlarge it and run the hand detector + gesture classifier on that crop only,
  3. keep a hand for this person only if its wrist is nearer to them than to anyone else.

    python find_gestures_centered.py takes_v2/D1_S2_r1      # one take
    python find_gestures_centered.py                        # every take in takes_v2/

Writes takes_crop2/<take>/ (log.jsonl with the new gestures, meta.json, raw.mp4,
frames.csv), ready for label_episodes.py. perception.py is not changed; this is a
labelling aid only.
"""
import glob
import json
import os
import shutil
import sys

import cv2
import mediapipe as mp

from engage.methods import COMMAND_MAP
from engage.perception import GestureClassifier
from engage.takeio import proc_size
from find_gestures_crop import hands_model, owner

SRC, OUT = "takes_v2", "takes_crop2"
MARGIN_X = 0.6                  # margin left/right of the box, as a fraction of its width
UPPER = 0.7                     # crop height = this fraction of the body (head to below the hips)
SIZE = 640                      # crops are resized to SIZE x SIZE
CONF = 0.4


def centred_crop(img, box):
    """Square crop centred on the upper body, with margin; returns crop, x0, y0, scale.
    The hand detector shrinks the whole crop to 192 px, so a whole-body crop makes a
    raised hand ~10 px and its finger points too rough to tell the gestures apart."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    side = int(max(bw * (1 + 2 * MARGIN_X), bh * UPPER, 32))
    cx, cy = (x0 + x1) / 2, y0 + bh * UPPER / 2
    sx, sy = int(cx - side / 2), int(cy - side / 2)
    H, W = img.shape[:2]
    pad = cv2.copyMakeBorder(img[max(0, sy):min(H, sy + side), max(0, sx):min(W, sx + side)],
                             max(0, -sy), max(0, sy + side - H), max(0, -sx), max(0, sx + side - W),
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    k = SIZE / side
    return cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA), sx, sy, k


def run(take, hl, clf):
    name = os.path.basename(os.path.normpath(take))
    meta = json.load(open(os.path.join(take, "meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(take, "log.jsonl"))]
    cap = cv2.VideoCapture(os.path.join(take, "raw.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    s = W / proc_size(W, H)[0]                   # log boxes are in the processing view
    mirror = meta.get("mirror", True)
    n_old = n_new = 0
    for r in recs:
        ok, img = cap.read()
        if not ok:
            break
        if mirror:
            img = cv2.flip(img, 1)               # same orientation as the log (and the classifier)
        boxes = [[v * s for v in p["bbox"]] for p in r["people"]]
        for k, p in enumerate(r["people"]):
            crop, sx, sy, sc = centred_crop(img, boxes[k])
            res = hl.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            hs = []
            for lm in res.hand_landmarks or []:
                wx, wy = sx + lm[0].x * SIZE / sc, sy + lm[0].y * SIZE / sc
                # this person's hand only if the wrist is in their box and no one else's box fits better
                if owner(wx, wy, boxes) != k:
                    continue
                pts = [[min(int(q.x * SIZE), SIZE - 1), min(int(q.y * SIZE), SIZE - 1)] for q in lm]
                g, c = clf(pts)
                hs.append((g, round(c, 3)))
            if p.get("gesture") in COMMAND_MAP and p.get("gesture_conf", 0) >= CONF:
                n_old += 1
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
    meta.update(gestures_from="centred person crops (find_gestures_centered.py)",
                crop_source=take.replace("\\", "/"))
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    print(f"{name}: command-gesture person-frames  full frame {n_old}  ->  centred crops {n_new}", flush=True)


def main():
    takes = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(SRC, "*")) if os.path.isdir(d))
    takes = [t for t in takes if os.path.exists(os.path.join(t, "log.jsonl"))
             and json.load(open(os.path.join(t, "meta.json"))).get("scenario") not in ("calibration", "test")]
    hl, clf = hands_model(), GestureClassifier()
    for t in takes:
        run(t, hl, clf)


if __name__ == "__main__":
    main()
