#!/usr/bin/env python3
"""Check how many people the detector finds in a recorded take - old vs new mode.

    python count_people.py takes/D1_S2_r1            # compares video vs image mode
    python count_people.py takes/D1_S2_r1 --frames 150

Prints, for each mode, how often 0/1/2/3/... people were found, and writes
takes/<take>/_people_check.mp4 (new mode, boxes drawn) so you can look at it.
Nothing in the take is changed.
"""
import argparse
import os
from collections import Counter

import cv2

from engage.perception import Perception
from engage.takeio import TakeReader, to_proc_view


def run(take, mode, n, video_out=None):
    perc = Perception(mode=mode)
    hist, w = Counter(), None
    for i, bgr, depth in TakeReader(take):
        if i >= n:
            break
        frame, d = to_proc_view(bgr, depth, mirror=True)
        feats, _ = perc.process(frame, d, t=i / 12.0)
        hist[len(feats)] += 1
        if video_out:
            if w is None:
                w = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), 12,
                                    (frame.shape[1], frame.shape[0]))
            for f in feats:
                x0, y0, x1, y1 = f.bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 200, 0), 2)
                cv2.putText(frame, f"#{f.tid}", (x0 + 3, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.putText(frame, f"{len(feats)} people", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            w.write(frame)
    if w is not None:
        w.release()
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    for mode in ("video", "image"):
        out = os.path.join(a.take, "_people_check.mp4") if mode == "image" else None
        h = run(a.take, mode, a.frames, out)
        tot = sum(h.values()) or 1
        dist = "  ".join(f"{k} people: {100 * v / tot:.0f}%" for k, v in sorted(h.items()))
        print(f"{mode:5s} mode -> {dist}")
    print(f"\nLook at {os.path.join(a.take, '_people_check.mp4')}: everyone in view should have a green box most of the time.")


if __name__ == "__main__":
    main()
