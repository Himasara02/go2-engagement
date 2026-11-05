"""The four command-acceptance policies compared in the paper.

All four see exactly the same per-person features, so differences in
results come only from the gating policy.

  gesture_only : any confident command gesture from anyone is executed
  nearest      : only the nearest person's gesture counts
  type1        : two-layer Type-1 fuzzy engagement gate
  it2          : two-layer Interval Type-2 gate, FOU from measured
                 uncertainty, decision on the lower bound E_low
"""
from __future__ import annotations

from .fuzzy import (DecisionParams, EngageInput, EngagementFLS, STOP_LABELS,
                    decide)

COMMAND_MAP = {
    "palm": (0.0, 0.0), "stop": (0.0, 0.0),
    "stop_inverted": (1.0, 0.0), "fist": (1.0, 0.0),
    "call": (-1.0, 0.0), "mute": (-1.0, 0.0),
    "one": (0.0, 1.0), "two_up": (0.0, -1.0),
}
METHODS = ["gesture_only", "nearest", "type1", "it2"]

_T1 = EngagementFLS(type1=True)
_IT2 = EngagementFLS(type1=False)


def engagement(p: dict) -> dict:
    x = EngageInput(p["distance"], p["distance_sigma"], p["facing"],
                    p["facing_sigma"], p["approach"])
    e1 = _T1(x)
    e2 = _IT2(x)
    return {"e_t1": round(e1.e, 4), "e2_low": round(e2.e_low, 4),
            "e2_high": round(e2.e_high, 4)}


def accept_per_person(people: list[dict], params=DecisionParams()) -> dict:
    """people: list of feature dicts (with engagement fields added).
    Returns {method: set(track ids whose gesture is accepted this frame)}."""
    out = {m: set() for m in METHODS}
    cmd = [p for p in people if p.get("gesture") in COMMAND_MAP]
    for p in cmd:
        if p["gesture_conf"] >= params.theta_conf:
            out["gesture_only"].add(p["tid"])
    if people:
        near = min(people, key=lambda p: p["distance"])
        if near in cmd and near["gesture_conf"] >= params.theta_conf:
            out["nearest"].add(near["tid"])
    for p in cmd:
        if decide(p["e_t1"], p["gesture"], p["gesture_conf"], params):
            out["type1"].add(p["tid"])
        if decide(p["e2_low"], p["gesture"], p["gesture_conf"], params):
            out["it2"].add(p["tid"])
    return out


def arbitrate(people: list[dict], accepted: set, method: str):
    """Several people accepted at once: STOP always wins, otherwise the most
    engaged person wins. Returns (tid, label) or (None, None)."""
    cands = [p for p in people if p["tid"] in accepted]
    if not cands:
        return None, None
    stops = [p for p in cands if p["gesture"] in STOP_LABELS]
    if stops:
        p = stops[0]
        return p["tid"], p["gesture"]
    key = {"it2": "e2_low", "type1": "e_t1"}.get(method)
    p = max(cands, key=lambda q: q[key] if key else -q["distance"])
    return p["tid"], p["gesture"]
