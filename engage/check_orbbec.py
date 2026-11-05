#!/usr/bin/env python3
"""Quick Orbbec test: shows colour and depth side by side.

    python check_orbbec.py

You should see your colour image on the left and a coloured depth image on
the right, and the distance (in metres) to whatever is in the centre of the
image. Stand 2 m in front of the camera: the number should read about 2.0.
Press q to quit.
"""
import time

import cv2
import numpy as np

from engage.orbbec import OrbbecSource


def main():
    cam = OrbbecSource()
    print("camera profile:", cam.profile)
    n, t0 = 0, time.time()
    while True:
        ok, bgr, depth, _ = cam.read()
        if not ok:
            time.sleep(0.005)
            continue
        n += 1
        h, w = depth.shape
        c = depth[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
        c = c[c > 0]
        dist = f"{np.median(c) / 1000:.2f} m" if c.size else "no reading"
        valid = 100.0 * np.count_nonzero(depth) / depth.size
        vis_d = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / 6000.0), cv2.COLORMAP_JET)
        vis_d[depth == 0] = 0
        both = np.hstack([bgr, vis_d])
        fps = n / (time.time() - t0)
        txt = f"centre: {dist}   valid depth: {valid:.0f}%   {fps:.1f} fps   {w}x{h}"
        cv2.circle(both, (w // 2, h // 2), 6, (0, 255, 0), 2)
        cv2.putText(both, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(both, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        s = 1280 / both.shape[1]
        cv2.imshow("orbbec check (q to quit)", cv2.resize(both, None, fx=s, fy=s))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cam.close()
    print(f"OK: {n} frames, {fps:.1f} fps, colour+depth {w}x{h}")


if __name__ == "__main__":
    main()
