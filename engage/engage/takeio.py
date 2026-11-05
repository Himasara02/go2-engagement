"""Read/write the raw data of a take so everything can be recomputed offline.

Layout of takes/<take>/ (new format, meta["raw_format"] == 2):
  raw.mp4            colour frames exactly as the camera gave them (NOT mirrored,
                     native resolution), one frame per processed frame
  depth/000000.npz   aligned depth in mm, uint16, half resolution, chunks of
                     CHUNK frames: arrays `depth` (N,H/2,W/2) and `idx` (N,)
  frames.csv         frame index, wall-clock timestamp, take time
  log.jsonl / meta.json / annot.mp4 / labels*.json

Processing view: frames are resized to at most PROC_W wide and mirrored
(the gesture classifier was trained on mirrored images). Bounding boxes in
log.jsonl are in the processing view; use `to_proc_view` to get it.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np

CHUNK = 200
PROC_W = 960


def proc_size(w, h):
    if w <= PROC_W:
        return w, h
    return PROC_W, int(round(h * PROC_W / w))


def to_proc_view(bgr, depth=None, mirror=True):
    h, w = bgr.shape[:2]
    pw, ph = proc_size(w, h)
    if (pw, ph) != (w, h):
        bgr = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        if depth is not None:
            depth = cv2.resize(depth, (pw, ph), interpolation=cv2.INTER_NEAREST)
    if mirror:
        bgr = cv2.flip(bgr, 1)
        if depth is not None:
            depth = cv2.flip(depth, 1)
    return bgr, depth


class DepthWriter:
    def __init__(self, take_dir):
        self.dir = os.path.join(take_dir, "depth")
        os.makedirs(self.dir, exist_ok=True)
        self.buf, self.idx, self.start = [], [], None

    def add(self, i, depth_mm):
        if self.start is None:
            self.start = i
        half = depth_mm[::2, ::2]
        self.buf.append(np.clip(half, 0, 65535).astype(np.uint16))
        self.idx.append(i)
        if len(self.buf) >= CHUNK:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        np.savez_compressed(os.path.join(self.dir, f"{self.idx[0]:06d}.npz"),
                            depth=np.stack(self.buf), idx=np.array(self.idx))
        self.buf, self.idx = [], []


class TakeReader:
    """Iterate (i, bgr_raw, depth_mm_or_None) over a recorded take."""

    def __init__(self, take_dir):
        self.cap = cv2.VideoCapture(os.path.join(take_dir, "raw.mp4"))
        self.depth = {}
        for f in sorted(glob.glob(os.path.join(take_dir, "depth", "*.npz"))):
            z = np.load(f)
            for k, i in enumerate(z["idx"]):
                self.depth[int(i)] = (f, k)
        self._cache = (None, None)

    def _depth(self, i, shape):
        if i not in self.depth:
            return None
        f, k = self.depth[i]
        if self._cache[0] != f:
            self._cache = (f, np.load(f)["depth"])
        d = self._cache[1][k].astype(np.float32)
        return cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    def __iter__(self):
        i = 0
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            yield i, bgr, self._depth(i, bgr.shape)
            i += 1
