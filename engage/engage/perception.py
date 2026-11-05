"""Per-person perception: who is there, how far, are they facing the robot,
and which gesture (if any) is each of them making.

MediaPipe PoseLandmarker (multi-person) + HandLandmarker (multi-hand) +
the team's 18-class keypoint MLP (keypoint_classifier.tflite).
Hands are attributed to people by matching the hand wrist to the pose wrists.
"""
from __future__ import annotations

import copy
import csv
import itertools
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")

# Pose landmark indices (MediaPipe BlazePose 33)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24

SHOULDER_WIDTH_M = 0.38


# --------------------------------------------------------------------------
# Gesture classifier (TFLite) — tries LiteRT, tflite_runtime, tensorflow
# --------------------------------------------------------------------------
def _interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


class GestureClassifier:
    def __init__(self):
        self.it = _interpreter(os.path.join(MODELS, "keypoint_classifier.tflite"))
        self.i = self.it.get_input_details()[0]["index"]
        self.o = self.it.get_output_details()[0]["index"]
        with open(os.path.join(MODELS, "keypoint_classifier_label.csv"),
                  encoding="utf-8-sig") as f:
            self.labels = [r[0] for r in csv.reader(f)]

    def __call__(self, pts_px: list) -> tuple[str, float]:
        tmp = copy.deepcopy(pts_px)
        bx, by = tmp[0]
        for p in tmp:
            p[0] -= bx
            p[1] -= by
        flat = list(itertools.chain.from_iterable(tmp))
        mx = max(map(abs, flat)) or 1.0
        x = np.array([[v / mx for v in flat]], dtype=np.float32)
        self.it.set_tensor(self.i, x)
        self.it.invoke()
        prob = np.squeeze(self.it.get_tensor(self.o))
        k = int(np.argmax(prob))
        return self.labels[k], float(prob[k])


# --------------------------------------------------------------------------
# Facing score from pose (head + body), with an uncertainty estimate
# --------------------------------------------------------------------------
def facing_from_pose(lm, wlm) -> tuple[float, float]:
    """Returns (facing in [0,1], sigma).

    head: nose position between the ears in the image (frontal => centred,
          both ears visible). body: shoulder depth difference in world
          landmarks (frontal => shoulders at equal depth).
    """
    vis = lambda i: getattr(lm[i], "visibility", 1.0) or 0.0
    # --- head yaw proxy
    le, re, n = lm[L_EAR], lm[R_EAR], lm[NOSE]
    ear_dx = le.x - re.x       # in the mirrored image, a frontal face has le.x > re.x
    ear_w = abs(ear_dx) + 1e-6
    off = (n.x - 0.5 * (le.x + re.x)) / (0.5 * ear_w)      # 0 = centred
    head = max(0.0, 1.0 - abs(off)) if ear_dx > 0 else 0.0   # ear order flips when facing away
    # eyes visible relative to ears: when looking away eye visibility drops
    eye_v = 0.5 * (vis(L_EYE) + vis(R_EYE))
    head *= min(1.0, 0.3 + eye_v)
    # --- body yaw from world landmarks (metres, hip-centred)
    ls, rs = wlm[L_SH], wlm[R_SH]
    dx, dz = ls.x - rs.x, ls.z - rs.z
    body_yaw = math.degrees(math.atan2(abs(dz), abs(dx) + 1e-6))   # 0 = frontal
    body = max(0.0, math.cos(math.radians(min(90.0, body_yaw))))
    facing = 0.65 * head + 0.35 * body
    # --- uncertainty: low landmark visibility -> wide FOU
    v = np.mean([vis(i) for i in (NOSE, L_EAR, R_EAR, L_EYE, R_EYE, L_SH, R_SH)])
    sigma = 0.04 + 0.35 * (1.0 - float(v))
    return float(np.clip(facing, 0, 1)), float(sigma)


# --------------------------------------------------------------------------
# Tracking (greedy nearest-centre, pose gives few people so this is enough)
# --------------------------------------------------------------------------
@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    scale: float
    last_t: float
    dist_hist: deque = field(default_factory=lambda: deque(maxlen=12))
    face_hist: deque = field(default_factory=lambda: deque(maxlen=8))
    gest_hist: deque = field(default_factory=lambda: deque(maxlen=7))
    approach: float = 0.0
    prev_d: float | None = None
    prev_t: float | None = None


class Tracker:
    def __init__(self, max_age=1.0):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, dets, t):
        """dets: list of (cx, cy, scale). Returns list of track ids per det."""
        ids = [None] * len(dets)
        free = set(self.tracks)
        pairs = []
        for i, (cx, cy, s) in enumerate(dets):
            for tid in free:
                tr = self.tracks[tid]
                d = math.hypot(cx - tr.cx, cy - tr.cy) / max(s, tr.scale, 1e-3)
                pairs.append((d, i, tid))
        pairs.sort()
        used_d, used_t = set(), set()
        for d, i, tid in pairs:
            if d > 1.5 or i in used_d or tid in used_t:
                continue
            ids[i] = tid
            used_d.add(i)
            used_t.add(tid)
        for i, (cx, cy, s) in enumerate(dets):
            if ids[i] is None:
                ids[i] = self.next_id
                self.tracks[self.next_id] = Track(self.next_id, cx, cy, s, t)
                self.next_id += 1
            tr = self.tracks[ids[i]]
            tr.cx, tr.cy, tr.scale, tr.last_t = cx, cy, s, t
        for tid in list(self.tracks):
            if t - self.tracks[tid].last_t > self.max_age:
                del self.tracks[tid]
        return ids


# --------------------------------------------------------------------------
# Main perception
# --------------------------------------------------------------------------
@dataclass
class PersonFeat:
    tid: int
    bbox: tuple            # x0,y0,x1,y1 px
    distance: float
    distance_sigma: float
    approach: float
    facing: float
    facing_sigma: float
    gesture: str | None    # smoothed label, None if no hand
    gesture_conf: float

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class Perception:
    """mode="image" (default): detect every person from scratch on every frame.
    MediaPipe's VIDEO mode only re-runs its person detector when tracking is
    lost, so after locking on to the first person it never picks up the others.
    Identity across frames is handled by our own Tracker instead."""

    def __init__(self, max_people=6, fx=615.0, mode=None):
        self.max_people = max_people
        self.fx = fx
        self.mode = (mode or os.environ.get("PERCEPTION_MODE", "image")).lower()
        rm = mp_vision.RunningMode.IMAGE if self.mode == "image" else mp_vision.RunningMode.VIDEO
        bo = lambda p: mp_python.BaseOptions(model_asset_path=os.path.join(MODELS, p))
        self.pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=bo(os.environ.get("POSE_MODEL", "pose_landmarker_full.task")),
                running_mode=rm,
                num_poses=max_people,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3))
        self.hands = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=bo("hand_landmarker.task"),
                running_mode=rm,
                num_hands=2 * max_people,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        self.clf = GestureClassifier()
        self.tracker = Tracker()
        self._last_ms = -1

    def _distance(self, lm, w, h, depth):
        pts = [(lm[i].x * w, lm[i].y * h) for i in (L_SH, R_SH, L_HIP, R_HIP)]
        sh_px = math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
        d_pin = self.fx * SHOULDER_WIDTH_M / max(sh_px, 1.0)
        if depth is not None:
            vals = []
            for x, y in pts + [((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)]:
                xi, yi = int(x), int(y)
                if 4 <= xi < w - 4 and 4 <= yi < h - 4:
                    patch = depth[yi - 4:yi + 5, xi - 4:xi + 5].astype(np.float32)
                    patch = patch[(patch > 200) & (patch < 9000)]  # mm
                    if patch.size:
                        vals.append(float(np.median(patch)) / 1000.0)
            if len(vals) >= 2:
                med = float(np.median(vals))
                mad = float(np.median(np.abs(np.array(vals) - med)))
                return med, 0.05 + 1.5 * mad + 0.1 * (5 - len(vals)) / 5
            return d_pin, 0.45   # depth holes -> treat like occlusion
        return d_pin, 0.15 + 0.08 * d_pin

    def process(self, frame_bgr, depth=None, t=None):
        """frame must already be mirrored (cv2.flip(.,1)) like training data."""
        t = time.time() if t is None else t
        h, w = frame_bgr.shape[:2]
        ms = int(t * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if self.mode == "image":
            pr = self.pose.detect(img)
            hr = self.hands.detect(img)
        else:
            pr = self.pose.detect_for_video(img, ms)
            hr = self.hands.detect_for_video(img, ms)

        poses = pr.pose_landmarks or []
        wposes = pr.pose_world_landmarks or []
        dets = []
        for lm in poses:
            cx = 0.5 * (lm[L_SH].x + lm[R_SH].x) * w
            cy = 0.5 * (lm[L_SH].y + lm[R_SH].y) * h
            s = math.hypot((lm[L_SH].x - lm[R_SH].x) * w, (lm[L_SH].y - lm[R_SH].y) * h)
            dets.append((cx, cy, max(s, 10.0)))
        ids = self.tracker.update(dets, t)

        # hand -> person by wrist proximity
        hand_owner = {}
        for hi, hl in enumerate(hr.hand_landmarks or []):
            hx, hy = hl[0].x * w, hl[0].y * h
            best, bd = None, 1e9
            for pi, lm in enumerate(poses):
                for wi in (L_WR, R_WR):
                    d = math.hypot(hx - lm[wi].x * w, hy - lm[wi].y * h)
                    d /= dets[pi][2]                      # relative to shoulder width
                    if d < bd:
                        bd, best = d, pi
            # strict: the hand must sit on one of that person's own wrists,
            # otherwise a hand of an undetected person gets mis-attributed
            if best is not None and bd < 0.7:
                hand_owner.setdefault(best, []).append(hi)

        feats = []
        for pi, lm in enumerate(poses):
            tr = self.tracker.tracks[ids[pi]]
            d, ds = self._distance(lm, w, h, depth)
            tr.dist_hist.append(d)
            if len(tr.dist_hist) >= 4:
                ds = max(ds, float(np.std(tr.dist_hist)))
            dsm = float(np.median(tr.dist_hist))
            if tr.prev_d is not None and t > tr.prev_t:
                v = -(dsm - tr.prev_d) / (t - tr.prev_t)
                tr.approach = 0.8 * tr.approach + 0.2 * float(np.clip(v, -2, 2))
            tr.prev_d, tr.prev_t = dsm, t

            f, fs = facing_from_pose(lm, wposes[pi]) if pi < len(wposes) else (0.0, 0.4)
            tr.face_hist.append(f)
            fsm = float(np.mean(tr.face_hist))
            if len(tr.face_hist) >= 3:
                fs = max(fs, float(np.std(tr.face_hist)))

            g, gc = None, 0.0
            if pi in hand_owner:
                cands = []
                for hi in hand_owner[pi]:
                    pts = [[min(int(p.x * w), w - 1), min(int(p.y * h), h - 1)]
                           for p in hr.hand_landmarks[hi]]
                    cands.append(self.clf(pts))
                g, gc = max(cands, key=lambda c: c[1])
                tr.gest_hist.append((g, gc))
            else:
                tr.gest_hist.clear()
            if tr.gest_hist:
                lab = Counter(x[0] for x in tr.gest_hist).most_common(1)[0][0]
                conf = float(np.mean([c for l, c in tr.gest_hist if l == lab]))
                frac = sum(1 for l, _ in tr.gest_hist if l == lab) / len(tr.gest_hist)
                g, gc = lab, conf * frac

            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            bbox = (int(max(0, min(xs))), int(max(0, min(ys))),
                    int(min(w - 1, max(xs))), int(min(h - 1, max(ys))))
            feats.append(PersonFeat(tr.tid, bbox, dsm, ds, tr.approach, fsm, fs, g, gc))
        return feats, (poses, hr)
