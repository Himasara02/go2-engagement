#!/usr/bin/env python3
"""Second privacy pass for the supplementary frames: FaceLandmarker on overlapping 2x-upscaled tiles of each
(already blurred) frame finds small or missed faces; every face found is blurred again (ellipse, feathered).
Writes the frames back in place and prints what it found.   ~/miniconda3/envs/go2/bin/python supp_faces.py
"""
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mpv

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "cvpr2027_paper", "figures", "supp_src")
face = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(HERE, "models", "face_landmarker.task")),
    running_mode=mpv.RunningMode.IMAGE, num_faces=4, min_face_detection_confidence=0.2,
    min_face_presence_confidence=0.2))


def blur_ellipse(img, cx, cy, rx, ry):
    H, W = img.shape[:2]
    x0, x1, y0, y1 = int(max(0, cx - rx)), int(min(W, cx + rx)), int(max(0, cy - ry)), int(min(H, cy + ry))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, (x1 - x0) // 10), max(1, (y1 - y0) // 10)), interpolation=cv2.INTER_AREA)
    bl = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST), (0, 0), 5)
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(m, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 2)[..., None]
    img[y0:y1, x0:x1] = (m * bl + (1 - m) * roi).astype(np.uint8)


def main():
  for f in sorted(glob.glob(os.path.join(SRC, "ex*.png"))):
      img = cv2.imread(f)
      H, W = img.shape[:2]
      found = []
      T, step = 360, 180
      for y in range(0, H - T + 1, step):
          for x in range(0, W - T + 1, step):
              tile = cv2.resize(img[y:y + T, x:x + T], (2 * T, 2 * T), interpolation=cv2.INTER_CUBIC)
              r = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)))
              for lm in r.face_landmarks or []:
                  xs = [x + p.x * T for p in lm]
                  ys = [y + p.y * T for p in lm]
                  found.append((np.mean(xs), np.mean(ys), (max(xs) - min(xs)) * 0.9 + 6, (max(ys) - min(ys)) * 0.9 + 6))
      for cx, cy, rx, ry in found:
          blur_ellipse(img, cx, cy, rx, ry)
      cv2.imwrite(f, img)
      print(os.path.basename(f), "faces found by the tile pass:", [(int(a), int(b), int(c)) for a, b, c, _ in found])


if __name__ == "__main__":
    main()
