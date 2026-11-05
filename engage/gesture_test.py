#!/usr/bin/env python3
"""Step-by-step camera test: one short check per gesture / situation.

    python gesture_test.py                     # Orbbec (main setup)
    python gesture_test.py --source webcam:0   # laptop webcam (no depth)
    python gesture_test.py --only palm,call    # run only some tests

For each test the window shows what to do. Press ENTER, a 3-2-1 countdown
runs, then the camera is processed for a few seconds with a timer on screen.
It stops by itself and says PASS / FAIL and what the system saw.
Then: ENTER = next test, r = redo, q = quit.  (s = skip a test.)

Uses the same perception, engagement and sustain logic as run_live.py.
Nothing is written to takes/; a summary + snapshots go to
gesture_tests/<date-time>/.
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from collections import Counter

import cv2
import numpy as np

from engage.methods import COMMAND_MAP, METHODS, accept_per_person, engagement
from engage.perception import Perception
from engage.takeio import to_proc_view
from run_live import SUSTAIN

WIN = "gesture test"
CONF = 0.4          # same as label_episodes: a gesture "counts" from this confidence

# kind: idle   = person seen, nothing accepted
#       accept = the expected gesture is recognised and accepted by IT2
#       reject = nothing accepted by IT2
#       info   = no pass/fail, just report what happened
#       two    = two people boxed at once
TESTS = [
    dict(key="idle", kind="idle", title="Stand still, no gesture",
         lines=["Stand about 2 m from the camera.", "Face the camera, hands down.", "Do NOT gesture."]),
    dict(key="fist", kind="accept", expect=["fist"], title="FIST  (= forward)",
         lines=["Stand about 2 m away, face the camera.", "Show a FIST toward the camera,",
                "beside your body, and hold it until the timer ends."]),
    dict(key="palm", kind="accept", expect=["palm", "stop"], title="PALM  (= STOP)",
         lines=["Stand about 2 m away, face the camera.", "Open hand, fingers spread,",
                "PALM toward the camera. Hold it."]),
    dict(key="call", kind="accept", expect=["call"], title="CALL  (= backward)",
         lines=["Stand about 2 m away, face the camera.", "Thumb and little finger out,",
                "like a phone. Hold it."]),
    dict(key="one", kind="accept", expect=["one"], title="ONE  (= turn left)",
         lines=["Stand about 2 m away, face the camera.", "Index finger pointing up. Hold it."]),
    dict(key="two_up", kind="accept", expect=["two_up"], title="TWO-UP  (= turn right)",
         lines=["Stand about 2 m away, face the camera.", "Index and middle finger up,",
                "held together. Hold it."]),
    dict(key="sideways", kind="reject", title="Sideways FIST (should be ignored)",
         lines=["Stand about 2 m away.", "Turn your body sideways and look at the wall.",
                "Show a FIST toward the wall. Hold it."]),
    dict(key="back", kind="reject", title="Back turned (should be ignored)",
         lines=["Stand about 2 m away.", "Turn your back to the camera.",
                "Show a FIST in front of you. Hold it."]),
    dict(key="close", kind="accept", expect=["fist"], title="Close-up FIST (1.2 m)",
         lines=["Stand about 1.2 m from the camera, face it.", "Show a FIST. Hold it."]),
    dict(key="far", kind="info", expect=["fist"], title="Far FIST (3.5 m) - info only",
         lines=["Stand about 3.5 m from the camera, face it.", "Show a FIST. Hold it.",
                "(No pass/fail: shows how far the gate still accepts.)"]),
    dict(key="two", kind="two", title="Two people (needs a 2nd person, s = skip)",
         lines=["Two people, about 2 m away, side by side,", "both facing the camera.",
                "Both show a FIST. Hold it."]),
]


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, source):
        self.has_depth = source == "orbbec"
        if source == "orbbec":
            from engage.orbbec import OrbbecSource
            try:
                self.cam = OrbbecSource()
            except RuntimeError as e:
                sys.exit(f"Orbbec camera problem: {e}")
            self._read = self.cam.read
            self.close = self.cam.close
        else:
            cap = cv2.VideoCapture(int(source.split(":")[1]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._read = lambda: (*cap.read(), None, None)
            self.close = cap.release
        self.last = None

    def read(self):
        """Newest frame in the processing view (mirrored, <=960 wide), or None."""
        ok, raw, depth, _ = self._read()
        if not ok or raw is None:
            return None, None
        frame, dproc = to_proc_view(raw, depth, mirror=True)
        self.last = frame
        return frame, dproc

    def wait_frame(self, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            f, d = self.read()
            if f is not None:
                return f, d
            time.sleep(0.005)
        return self.last, None


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def text(img, s, org, scale=0.7, col=(255, 255, 255), th=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def panel(frame, title, lines, footer, title_col=(0, 220, 255)):
    img = (frame * 0.35).astype(np.uint8) if frame is not None else np.zeros((540, 960, 3), np.uint8)
    text(img, title, (30, 60), 1.0, title_col, 2)
    y = 110
    for s in lines:
        col = (0, 0, 255) if s.startswith("!") else (255, 255, 255)
        text(img, s.lstrip("!"), (30, y), 0.65, col, 2)
        y += 34
    text(img, footer, (30, img.shape[0] - 25), 0.65, (0, 255, 0), 2)
    return img


def draw_people(img, people, acc_it2):
    for p in people:
        x0, y0, x1, y1 = p["bbox"]
        ok = p["tid"] in acc_it2
        col = (40, 200, 40) if ok else ((0, 160, 255) if p.get("gesture") in COMMAND_MAP else (180, 180, 180))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        g = p.get("gesture") or "-"
        s = [f"{p['distance']:.1f}m face {p['facing']:.2f}  E_low {p['e2_low']:.2f}",
             f"{g} {p['gesture_conf']:.2f}" + ("  ACCEPT" if ok else "")]
        for i, line in enumerate(s):
            text(img, line, (x0, max(18, y0 - 26 + 18 * i)), 0.5, col, 1)


def key():
    k = cv2.waitKey(30) & 0xFF
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        return "q"
    return {13: "enter", 10: "enter", 32: "enter", ord("r"): "r", ord("s"): "s",
            ord("q"): "q", 27: "q"}.get(k)


# --------------------------------------------------------------------------
# one test
# --------------------------------------------------------------------------
def main_person(people):
    """The biggest box = the person doing the test."""
    if not people:
        return None
    return max(people, key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]))


def run_test(cam, perc, seconds, countdown):
    """Countdown, then process the camera for `seconds`. Returns frames or None if quit."""
    t0 = time.time()
    while time.time() - t0 < countdown:
        f, _ = cam.wait_frame()
        img = f.copy()
        n = countdown - int(time.time() - t0)
        text(img, str(n), (img.shape[1] // 2 - 30, img.shape[0] // 2 + 30), 4.0, (0, 220, 255), 8)
        text(img, "get ready", (img.shape[1] // 2 - 80, img.shape[0] // 2 + 90), 1.0, (0, 220, 255), 2)
        cv2.imshow(WIN, img)
        if key() == "q":
            return None
    frames, cnt = [], {m: {} for m in METHODS}
    t0 = time.time()
    while (left := seconds - (time.time() - t0)) > 0:
        f, d = cam.read()
        if f is None:
            time.sleep(0.005)
            continue
        feats, _ = perc.process(f, d)
        people = []
        for x in feats:
            p = x.as_dict()
            p.update(engagement(p))
            people.append(p)
        acc = accept_per_person(people)
        sus = {}
        for m in METHODS:
            c = cnt[m]
            for tid in list(c):
                if tid not in acc[m]:
                    del c[tid]
            for tid in acc[m]:
                c[tid] = c.get(tid, 0) + 1
            sus[m] = {tid for tid, v in c.items() if v >= SUSTAIN}
        frames.append({"t": time.time() - t0, "people": people, "sus": sus, "depth": d is not None})
        vis = f.copy()
        draw_people(vis, people, sus["it2"])
        fps = len(frames) / max(time.time() - t0, 1e-3)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        text(vis, f"PROCESSING  {left:4.1f} s left   {fps:.1f} fps   "
                  f"{'depth' if d is not None else 'NO DEPTH'}", (10, 28), 0.7, (0, 255, 255), 2)
        cv2.rectangle(vis, (0, 40), (int(vis.shape[1] * left / seconds), 46), (0, 255, 255), -1)
        cv2.imshow(WIN, vis)
        frames[-1]["vis"] = vis
        if key() == "q":
            return None
    return frames


def evaluate(test, frames, need_depth):
    """Returns (status, lines, stats)."""
    n = len(frames)
    if n == 0:
        return "FAIL", ["!No camera frames were processed."], {}
    dur = frames[-1]["t"] or 1e-3
    mains = [main_person(f["people"]) for f in frames]
    seen = [m for m in mains if m]
    pres = len(seen) / n
    two = sum(len(f["people"]) >= 2 for f in frames) / n
    depth = sum(f["depth"] for f in frames) / n
    gest = Counter(m["gesture"] for m in seen if m["gesture"] and m["gesture_conf"] >= CONF)
    acc = {meth: Counter(m["gesture"] for f, m in zip(frames, mains) if m and m["tid"] in f["sus"][meth])
           for meth in METHODS}
    acc_any_it2 = sum(bool(f["sus"]["it2"]) for f in frames)
    med = lambda k: statistics.median(m[k] for m in seen) if seen else float("nan")
    stats = {"frames": n, "fps": round(n / dur, 1), "person_seen": round(pres, 2), "two_people": round(two, 2),
             "depth": round(depth, 2), "distance": round(med("distance"), 2), "facing": round(med("facing"), 2),
             "e2_low": round(med("e2_low"), 2), "e_t1": round(med("e_t1"), 2),
             "gestures": dict(gest), "accepted": {k: dict(v) for k, v in acc.items()}}
    fmt = lambda c: ", ".join(f"{g} ({v})" for g, v in c.most_common(3)) or "none"
    lines = [f"person seen in {100 * pres:.0f}% of frames   {stats['fps']} fps   "
             f"depth {'yes' if depth > 0.5 else 'NO'}",
             f"distance {stats['distance']:.2f} m   facing {stats['facing']:.2f}   "
             f"E_low {stats['e2_low']:.2f}   Type-1 E {stats['e_t1']:.2f}",
             f"gestures seen: {fmt(gest)}",
             f"accepted by IT2: {fmt(acc['it2'])}   Type-1: {fmt(acc['type1'])}"]
    kind, exp = test["kind"], set(test.get("expect", []))
    why = []
    if kind == "idle":
        ok = pres >= 0.6 and not acc_any_it2
        if pres < 0.6:
            why.append("!Person not seen most of the time (light? distance? in frame?).")
        if acc_any_it2:
            why.append("!A command was accepted although nobody gestured.")
    elif kind in ("accept", "info"):
        got = set(acc["it2"])
        ok = bool(got & exp)
        anyc = Counter(m["gesture"] for m in seen if m["gesture"])      # any confidence
        top = (gest or anyc).most_common(1)[0][0] if (gest or anyc) else None
        low = ""
        if top and top not in gest:
            confs = [m["gesture_conf"] for m in seen if m["gesture"] == top]
            low = f" (low confidence {statistics.median(confs):.2f})"
        if got - exp:
            why.append(f"!WRONG command accepted: {', '.join(sorted(got - exp))} "
                       f"(expected {'/'.join(sorted(exp))})")
        if not ok:
            if top not in exp:
                why.append(f"!Gesture not recognised: mostly read as '{top or 'nothing'}'{low}.")
            else:
                why.append(f"!Recognised, but IT2 did not accept (E_low {stats['e2_low']:.2f}, "
                           f"needs 0.60 for motion / 0.45 for STOP).")
    elif kind == "reject":
        ok = not acc_any_it2
        if not ok:
            why.append(f"!IT2 accepted {fmt(acc['it2'])} although you were not facing the camera.")
    else:  # two
        ok = two >= 0.5
        if not ok:
            why.append("!Two people were boxed in less than half the frames.")
    if need_depth and depth < 0.5:
        ok = False
        why.append("!No depth: re-plug the camera (blue USB-3 port).")
    if stats["fps"] < 8:
        why.append(f"!Slow ({stats['fps']} fps, target 8+): charger plugged in? other programs closed?")
    status = "INFO" if kind == "info" else ("PASS" if ok else "FAIL")
    return status, lines + why, stats


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="orbbec", help="orbbec (default) or webcam:0")
    ap.add_argument("--seconds", type=float, default=6.0, help="processing time per test")
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated test keys: " + ",".join(t["key"] for t in TESTS))
    args = ap.parse_args()
    tests = [t for t in TESTS if not args.only or t["key"] in args.only.split(",")]
    if not tests:
        sys.exit("no tests match --only")

    print("starting camera and models ...")
    cam = Camera(args.source)
    perc = Perception()
    out = os.path.join("gesture_tests", dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    results, k = {}, 0
    while k < len(tests):
        t = tests[k]
        # 1. instructions over the live (dimmed) camera view
        while True:
            f, _ = cam.wait_frame()
            cv2.imshow(WIN, panel(f, f"Test {k + 1}/{len(tests)}: {t['title']}", t["lines"],
                                  "ENTER = start     s = skip     q = quit"))
            a = key()
            if a in ("enter", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            results[t["key"]] = {"status": "SKIPPED"}
            k += 1
            continue
        # 2. countdown + processing
        frames = run_test(cam, perc, args.seconds, args.countdown)
        if frames is None:
            break
        status, lines, stats = evaluate(t, frames, cam.has_depth)
        snap = max(frames, key=lambda fr: len(fr["sus"]["it2"]))["vis"] if frames else None
        if snap is not None:
            cv2.imwrite(os.path.join(out, f"{t['key']}.jpg"), snap)
        results[t["key"]] = {"status": status, "title": t["title"], "details": [s.lstrip("!") for s in lines],
                             **stats}
        print(f"[{status}] {t['title']}")
        for s in lines:
            print("    " + s.lstrip("!"))
        # 3. result screen
        col = {"PASS": (0, 220, 0), "FAIL": (0, 0, 255)}.get(status, (0, 220, 255))
        while True:
            cv2.imshow(WIN, panel(snap, f"{status}:  {t['title']}", lines,
                                  "ENTER = next test     r = redo     q = quit", col))
            a = key()
            if a in ("enter", "r", "q"):
                break
        if a == "q":
            break
        if a == "enter":
            k += 1

    cam.close()
    cv2.destroyAllWindows()
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": args.source, "seconds": args.seconds, "when": dt.datetime.now().isoformat(),
                   "results": results}, fh, indent=1)
    print("\nSummary")
    for t in tests:
        r = results.get(t["key"], {"status": "not run"})
        print(f"  {r['status']:8s} {t['title']}")
    print(f"saved {out}/results.json (+ a snapshot per test)")


if __name__ == "__main__":
    main()
