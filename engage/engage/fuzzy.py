"""Two-layer engagement gate: Type-1 and Interval Type-2 fuzzy logic.

Layer 1 (engagement): E = f(distance, approach speed, facing)
Layer 2 (decision):   act = [E_low >= theta_E(cmd)] and [gesture_conf >= theta_c]

The IT2 system uses Gaussian membership functions with an *uncertain mean*:
the FOU half-width for each input is set at runtime from that person's
measured uncertainty (distance jitter / occlusion for distance, landmark
visibility + temporal jitter for facing). With zero uncertainty the IT2
system collapses exactly to the Type-1 system, so T1 vs IT2 is a clean
ablation of "using measured uncertainty".

Pure numpy-free python so it runs anywhere (laptop, Jetson, offline eval).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Membership functions
# --------------------------------------------------------------------------
def gauss(x: float, m: float, s: float) -> float:
    return math.exp(-0.5 * ((x - m) / s) ** 2)


def gauss_um(x: float, m: float, s: float, d: float) -> tuple[float, float]:
    """Gaussian with uncertain mean in [m-d, m+d]. Returns (lower, upper)."""
    if d <= 0:
        v = gauss(x, m, s)
        return v, v
    lo, hi = m - d, m + d
    upper = 1.0 if lo <= x <= hi else gauss(x, lo if x < lo else hi, s)
    lower = min(gauss(x, lo, s), gauss(x, hi, s))
    return lower, upper


def sigmoid_um(x: float, c: float, k: float, d: float) -> tuple[float, float]:
    """Shoulder (sigmoid) set with uncertain crossover c in [c-d, c+d]."""
    a = 1.0 / (1.0 + math.exp(-k * (x - (c - d))))
    b = 1.0 / (1.0 + math.exp(-k * (x - (c + d))))
    return (min(a, b), max(a, b))


# --------------------------------------------------------------------------
# Input sets (Hall's proxemics for distance)
#   distance [m]:   NEAR  (intimate/personal, < ~1.2)
#                   MID   (social, 1.2 - 3.6)
#                   FAR   (public, > ~3.6)
#   facing [0,1]:   LOW / HIGH   (1 = head + body toward robot)
#   approach [m/s]: AWAY / STILL / TOWARD (positive = moving toward robot)
# --------------------------------------------------------------------------
def mf_distance(d: float, u: float) -> dict:
    return {
        "NEAR": sigmoid_um(-d, -1.4, 4.0, u),     # high when d small
        "MID": gauss_um(d, 2.4, 0.8, u),
        "FAR": sigmoid_um(d, 3.6, 3.0, u),
    }


def mf_facing(f: float, u: float) -> dict:
    return {
        "LOW": sigmoid_um(-f, -0.45, 12.0, u),
        "HIGH": sigmoid_um(f, 0.62, 12.0, u),
    }


def mf_approach(v: float, u: float) -> dict:
    return {
        "AWAY": sigmoid_um(-v, 0.25, 12.0, u),
        "STILL": gauss_um(v, 0.0, 0.18, u),
        "TOWARD": sigmoid_um(v, 0.25, 12.0, u),
    }


# Rule base: (antecedents, consequent engagement singleton)
RULES = [
    ({"face": "HIGH", "dist": "NEAR"}, 0.95),
    ({"face": "HIGH", "dist": "MID"}, 0.85),
    ({"face": "HIGH", "dist": "FAR"}, 0.55),
    ({"face": "LOW", "dist": "NEAR"}, 0.30),
    ({"face": "LOW", "dist": "MID"}, 0.10),
    ({"face": "LOW", "dist": "FAR"}, 0.05),
    ({"face": "HIGH", "appr": "TOWARD"}, 0.95),
    ({"face": "HIGH", "appr": "STILL"}, 0.80),
    ({"appr": "AWAY"}, 0.10),
]


def _km(y: list[float], fl: list[float], fu: list[float]) -> tuple[float, float]:
    """Exact Karnik-Mendel type reduction (enumerate switch points)."""
    order = sorted(range(len(y)), key=lambda i: y[i])
    ys = [y[i] for i in order]
    lo = [fl[i] for i in order]
    up = [fu[i] for i in order]
    n = len(ys)
    yl, yr = math.inf, -math.inf
    for k in range(n + 1):
        # left: upper firing for small y, lower for large y
        wl = up[:k] + lo[k:]
        wr = lo[:k] + up[k:]
        sl, sr = sum(wl), sum(wr)
        if sl > 1e-9:
            yl = min(yl, sum(a * b for a, b in zip(ys, wl)) / sl)
        if sr > 1e-9:
            yr = max(yr, sum(a * b for a, b in zip(ys, wr)) / sr)
    if yl is math.inf:
        return 0.0, 0.0
    return yl, yr


@dataclass
class EngageInput:
    distance: float          # m
    distance_sigma: float    # m, measured
    facing: float            # 0..1
    facing_sigma: float      # measured
    approach: float          # m/s toward robot
    approach_sigma: float = 0.08


@dataclass
class EngageOutput:
    e_low: float
    e_high: float

    @property
    def e(self) -> float:
        return 0.5 * (self.e_low + self.e_high)


class EngagementFLS:
    """k_* scale measured sigma into FOU half-width. type1=True zeroes FOU."""

    def __init__(self, type1: bool = False, k_dist: float = 1.0,
                 k_face: float = 1.0, k_appr: float = 1.0):
        self.type1 = type1
        self.k_dist, self.k_face, self.k_appr = k_dist, k_face, k_appr

    def __call__(self, x: EngageInput) -> EngageOutput:
        z = 0.0 if self.type1 else 1.0
        sets = {
            "dist": mf_distance(x.distance, z * self.k_dist * x.distance_sigma),
            "face": mf_facing(x.facing, z * self.k_face * x.facing_sigma),
            "appr": mf_approach(x.approach, z * self.k_appr * x.approach_sigma),
        }
        y, fl, fu = [], [], []
        for ante, cons in RULES:
            lows = [sets[v][lab][0] for v, lab in ante.items()]
            ups = [sets[v][lab][1] for v, lab in ante.items()]
            y.append(cons)
            fl.append(min(lows))
            fu.append(min(ups))
        if self.type1:
            s = sum(fu)
            e = sum(a * b for a, b in zip(y, fu)) / s if s > 1e-9 else 0.0
            return EngageOutput(e, e)
        yl, yr = _km(y, fl, fu)
        return EngageOutput(yl, yr)


# --------------------------------------------------------------------------
# Layer 2 — decision
# --------------------------------------------------------------------------
STOP_LABELS = {"palm", "stop"}


@dataclass
class DecisionParams:
    theta_e_motion: float = 0.60   # engagement needed for motion commands
    theta_e_stop: float = 0.45     # safety bias: STOP accepted more easily
    theta_conf: float = 0.45       # gesture confidence (set on dev take dev1)


def decide(e_conservative: float, label: str, conf: float,
           p: DecisionParams = DecisionParams()) -> bool:
    th = p.theta_e_stop if label in STOP_LABELS else p.theta_e_motion
    return e_conservative >= th and conf >= p.theta_conf
