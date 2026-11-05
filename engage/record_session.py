#!/usr/bin/env python3
"""Guided recording session: calibration + the stationary scenarios S1-S6.

    python record_session.py --day 1 --location lab --people P01,P02,P03,P04
    python record_session.py --day 1 --location lab --people P01,P02 --practice
    python record_session.py --day 1 --location lab --people P01,P02,P03 --only S3

For every take the window explains who stands where and what to do. Press
ENTER, a 5-second countdown lets everyone get into position, then the take is
recorded for ~75 s. Cues at the top of the screen (with a beep) say exactly
when to gesture; the person at the laptop reads them out loud. The take stops
by itself, a quick check says whether it looks usable, then:
ENTER = next take, r = redo (saved as the next _r number), q = quit.
On the instruction screen: c = change commander, s = skip this scenario.

Takes are written exactly like run_live.py (takes/<take>/: raw.mp4, depth/,
frames.csv, log.jsonl, annot.mp4, meta.json), so label_episodes.py and
check_dataset.py work as usual. Nothing is ever overwritten.
--practice records test_... takes (scenario "test"), ignored by the dataset.
"""
import argparse
import csv
import datetime as dt
import json
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from engage.fuzzy import DecisionParams
from engage.methods import METHODS, accept_per_person, arbitrate, engagement
from engage.perception import Perception
from engage.takeio import DepthWriter, to_proc_view
from label_episodes import episodes
from run_live import SUSTAIN, draw, git_commit

WIN = "record session"
GESTURES = ["PALM", "FIST", "CALL", "ONE", "TWO-UP"]
HI, LO = 1200, 700          # beep pitches: HI = commander / main cue, LO = others


# --------------------------------------------------------------------------
# scenarios: instructions + timed cues (start_s, end_s, text, beep)
# --------------------------------------------------------------------------
def scenario(code, A, B):
    ev, idle = [], "hands down, wait for the next cue"
    if code == "CAL":
        title = "Calibration (1 person only)"
        lines = [f"Only {A} in view; everyone else out of the picture.",
                 "Tape marks at 1, 2, 3 and 4 m in front of the dog.",
                 "At each mark: FACE the dog 5 s, SIDEWAYS 5 s, BACK 5 s.",
                 "Follow the cues; a beep means: change now.",
                 "The person at the laptop reads every cue OUT LOUD."]
        t = 2
        for m in (1, 2, 3, 4):
            ev += [(t, t + 4, f"Walk to the {m} m mark", LO),
                   (t + 4, t + 9, f"{m} m: FACE the dog", HI),
                   (t + 9, t + 14, f"{m} m: turn SIDEWAYS", HI),
                   (t + 14, t + 19, f"{m} m: turn your BACK to the dog", HI)]
            t += 19
        idle = "stand still"
    elif code == "S1":
        title = "S1  Commander alone"
        lines = [f"Commander {A}: about 2 m in front of the dog, facing it.",
                 "Everyone else: stand to the side (still in view), still, hands down.",
                 "Each cue names a gesture: show it TO THE DOG and HOLD it",
                 "while the cue is shown, then drop the hand.",
                 "PALM = open hand   FIST   CALL = thumb + little finger",
                 "ONE = index up   TWO-UP = index + middle finger together",
                 "Hand beside or in front of the body, never hidden."]
        t = 3
        for g in GESTURES * 2:
            ev += [(t, t + 4, f"{A}: show {g} to the dog - HOLD", HI)]
            t += 7
    elif code == "S2":
        title = "S2  Bystanders gesture to each other"
        lines = [f"Commander {A}: 1.5-4 m from the dog, facing it.",
                 "Bystanders: stand in pairs (or a group of 3) facing",
                 "EACH OTHER, not the dog. Spread out left to right.",
                 "Cue 'to the DOG'  -> commander gestures to the dog.",
                 "Cue 'to EACH OTHER' -> bystanders gesture to their partner.",
                 "EVERYONE uses the same 5 gestures: PALM FIST CALL ONE TWO-UP",
                 "(no waving or talking gestures). Hold while the cue is shown (4 s),",
                 "then 2 s hands down before the next cue."]
        # 4 s gesture, then 2 s hands down to switch over
        for k in range(7):
            t = 3 + 12 * k
            ev += [(t, t + 4, f"{A}: gesture to the DOG", HI),
                   (t + 6, t + 10, "Bystanders: gesture to EACH OTHER", LO)]
        idle = "hands down - get ready for the next cue"
    elif code == "S3":
        title = "S3  Same moment: commander + bystander"
        lines = [f"Commander {A}: 2 m from the dog, facing it.",
                 "One bystander beside the commander (1 m apart), facing a friend.",
                 "The screen counts 3-2-1-NOW: count it OUT LOUD.",
                 "On NOW, at the same moment:",
                 "  commander gestures TO THE DOG, bystander gestures TO THE FRIEND.",
                 "Hold about 2 s, then hands down. Repeats 8 times."]
        for k in range(8):
            t = 3 + 9 * k
            ev += [(t, t + 1, "3 ...", 0), (t + 1, t + 2, "2 ...", 0), (t + 2, t + 3, "1 ...", 0),
                   (t + 3, t + 6, f"NOW!  {A} -> dog,  bystander -> friend", HI)]
    elif code == "S4":
        title = "S4  Commander: friend first, then the dog"
        lines = [f"Commander {A}: 2 m from the dog. A friend stands 1-2 m to the side.",
                 "Cue 'AT THE FRIEND': turn to the friend and gesture to them.",
                 "Cue 'AT THE DOG': turn back to the dog and gesture to it.",
                 "Everyone else: still, hands down. Repeats 6 times."]
        for k in range(6):
            t = 3 + 12 * k
            ev += [(t, t + 6, f"{A}: turn to the friend, gesture AT THE FRIEND", LO),
                   (t + 6, t + 12, f"{A}: turn back, gesture AT THE DOG", HI)]
    elif code == "S5":
        title = "S5  Two people gesture to the dog together"
        lines = [f"{A} and {B}: side by side, about 1 m apart, 2 m from the dog,",
                 "both facing the dog. Everyone else: at the side, hands down.",
                 "On each cue BOTH gesture TO THE DOG at the same time:",
                 "the cue says who shows PALM and who shows FIST.",
                 "Hold while the cue is shown. Repeats 6 times, swapping."]
        for k in range(6):
            t = 3 + 12 * k
            a, b = ("PALM", "FIST") if k % 2 == 0 else ("FIST", "PALM")
            ev += [(t, t + 4, f"BOTH to the dog:  {A} {a}  +  {B} {b}", HI)]
    elif code == "S6":
        title = "S6  Bystanders walk across"
        lines = [f"Commander {A}: stands still, 2-3 m from the dog, facing it.",
                 "Bystanders: walk slowly back and forth ACROSS in front of the dog",
                 "(1.5-4 m away), waving or gesturing to someone OFF TO THE SIDE,",
                 "never to the dog. Keep walking the whole take.",
                 "Cue 'gesture to the DOG now' -> commander gestures to the dog."]
        for k in range(7):
            t = 5 + 10 * k
            ev += [(t, t + 3, f"{A}: gesture to the DOG now", HI)]
        idle = "Bystanders: keep walking across, waving to the side"
    else:
        raise ValueError(code)
    dur = max(e[1] for e in ev) + 2
    return title, lines, ev, idle, dur


def beep(freq):
    if not freq:
        return
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, 250), daemon=True).start()
    except ImportError:
        sys.stdout.write("\a")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# drawing / keys
# --------------------------------------------------------------------------
def text(img, s, org, scale=0.7, col=(255, 255, 255), th=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def fit(s, scale, th, maxw):
    """Largest scale <= `scale` at which `s` is at most `maxw` pixels wide."""
    while scale > 0.35 and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0][0] > maxw:
        scale -= 0.05
    return scale


def wrap(s, scale, th, maxw, max_lines=2):
    """Split `s` into at most `max_lines` lines that fit `maxw`; returns (lines, scale)."""
    while True:
        lines, cur = [], ""
        for w in s.split():
            t = (cur + " " + w).strip()
            if cur and cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0][0] > maxw:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        wide = max(cv2.getTextSize(x, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0][0] for x in lines)
        if (len(lines) <= max_lines and wide <= maxw) or scale <= 0.4:
            return lines, scale
        scale -= 0.05


def banner(img, rows):
    """Black band at the top with big wrapped text. rows: (text, scale, colour, max_lines).
    Returns the band height."""
    w, y, todo = img.shape[1], 10, []
    for s, scale, col, ml in rows:
        th = 3 if scale >= 1.0 else 2
        lines, sc = wrap(s, scale, th, w - 30, ml)
        hgt = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][1]
        for x in lines:
            y += int(hgt * 1.5)
            todo.append((x, y, sc, col, th))
        y += 6
    y += 8
    cv2.rectangle(img, (0, 0), (w, y), (0, 0, 0), -1)
    for x, yy, sc, col, th in todo:
        text(img, x, (15, yy), sc, col, th)
    return y


def panel(frame, title, lines, footer, title_col=(0, 220, 255)):
    img = (frame * 0.3).astype(np.uint8) if frame is not None else np.zeros((540, 960, 3), np.uint8)
    maxw = img.shape[1] - 40
    text(img, title, (20, 50), fit(title, 1.2, 3, maxw), title_col, 3)
    y = 98
    for s in lines:
        col = (80, 80, 255) if s.startswith("!") else (255, 255, 255)
        s = s.lstrip("!")
        text(img, s, (20, y), fit(s, 0.8, 2, maxw), col, 2)
        y += 36
    for i, s in enumerate(footer):
        text(img, s, (20, img.shape[0] - 18 - 34 * (len(footer) - 1 - i)), fit(s, 0.75, 2, maxw),
             (0, 255, 0), 2)
    return img


def key(wait=30):
    k = cv2.waitKey(wait) & 0xFF
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        return "q"
    if k == ord("f"):       # f = toggle full screen
        full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
    return {13: "enter", 10: "enter", 32: "enter", ord("r"): "r", ord("s"): "s",
            ord("c"): "c", ord("q"): "q", 27: "q"}.get(k)


class Camera:
    def __init__(self, source):
        self.source = source
        if source == "orbbec":
            from engage.orbbec import OrbbecSource
            try:
                cam = OrbbecSource()
            except RuntimeError as e:
                sys.exit(f"Orbbec camera problem: {e}")
            self.profile, self.read, self.close = cam.profile, cam.read, cam.close
        else:
            cap = cv2.VideoCapture(int(source.split(":")[1]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.profile, self.close = None, cap.release
            self.read = lambda: (*cap.read(), None, None)
        self.last = None

    def preview(self):
        ok, raw, _, _ = self.read()
        if ok and raw is not None:
            self.last, _ = to_proc_view(raw, None, mirror=True)
        return self.last


# --------------------------------------------------------------------------
# one take (same loop and files as run_live.py)
# --------------------------------------------------------------------------
STOP = {"take": False, "all": False}


def record_take(cam, out, take, scen, args, commander, people, notes, ev, idle, dur):
    os.makedirs(out, exist_ok=True)
    perc = Perception()
    logf = open(os.path.join(out, "log.jsonl"), "w")
    fcsv = open(os.path.join(out, "frames.csv"), "w", newline="")
    fw = csv.writer(fcsv)
    fw.writerow(["i", "wall", "t"])
    dwr = raw_w = ann_w = None
    sustain = {m: {} for m in METHODS}
    started = dt.datetime.now().isoformat(timespec="seconds")
    n, fps_ema, raw_size, has_depth = 0, 0.0, None, False
    want = len(people)
    full_frames = 0
    cur, early = None, False
    STOP["take"] = False
    t_start = time.time()

    while not STOP["take"]:
        el = time.time() - t_start
        if el >= dur:
            break
        ok, raw, depth, _ = cam.read()
        if not ok or raw is None:
            time.sleep(0.005)
            continue
        wall = time.time()
        t = wall - t_start
        frame, dproc = to_proc_view(raw, depth, mirror=True)
        tic = time.time()
        feats, _ = perc.process(frame, dproc, t=None)
        people_f = []
        for f in feats:
            d = f.as_dict()
            d.update(engagement(d))
            people_f.append(d)
        acc = accept_per_person(people_f)
        sus = {}
        for m in METHODS:
            cnt = sustain[m]
            for tid in list(cnt):
                if tid not in acc[m]:
                    del cnt[tid]
            for tid in acc[m]:
                cnt[tid] = cnt.get(tid, 0) + 1
            sus[m] = {tid for tid, c in cnt.items() if c >= SUSTAIN}
        tid, label = arbitrate(people_f, sus["it2"], "it2")
        active = f"#{tid} {label}" if tid else None
        logf.write(json.dumps({"i": n, "t": round(t, 3), "people": people_f,
                               "acc": {m: sorted(v) for m, v in acc.items()},
                               "sus": {m: sorted(v) for m, v in sus.items()},
                               "drive": {"tid": tid, "label": label, "on": False}}) + "\n")
        fw.writerow([n, f"{wall:.3f}", f"{t:.3f}"])
        if raw_w is None:
            raw_size = (raw.shape[1], raw.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            raw_w = cv2.VideoWriter(os.path.join(out, "raw.mp4"), fourcc, 12.0, raw_size)
            ann_w = cv2.VideoWriter(os.path.join(out, "annot.mp4"), fourcc, 12.0,
                                    (frame.shape[1], frame.shape[0]))
        raw_w.write(raw)
        if depth is not None:
            has_depth = True
            if dwr is None:
                dwr = DepthWriter(out)
            dwr.add(n, depth)
        vis = frame.copy()
        draw(vis, people_f, sus["it2"], False, active)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(time.time() - tic, 1e-3))
        cv2.putText(vis, f"{fps_ema:.1f} fps  {take}  {len(people_f)} people"
                    + ("  depth" if depth is not None else "  NO DEPTH"), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ann_w.write(vis)
        n += 1
        full_frames += len(people_f) >= want

        # screen only (not saved): the cue, timer and box count
        e = next((x for x in ev if x[0] <= el < x[1]), None)
        if e is not cur:
            cur = e
            if e is not None:
                beep(e[3])
        show = vis.copy()
        # yellow = commander / main cue, light blue = bystanders, grey = wait
        if e:
            rows = [("NOW: " + e[2], 1.6, (0, 255, 255) if e[3] != LO else (255, 255, 120), 2)]
        else:
            rows = [(idle, 1.1, (200, 200, 200), 1)]
        nxt = next((x for x in ev if x[0] > el), None)
        if nxt is not None:
            rows.append((f"NEXT in {nxt[0] - el:.0f} s: {nxt[2]}", 0.8, (255, 200, 120), 1))
        hb = banner(show, rows)
        left = dur - el
        cv2.rectangle(show, (0, hb), (int(show.shape[1] * left / dur), hb + 8), (0, 255, 255), -1)
        nb = len(people_f)
        text(show, f"REC {take}   {left:4.0f} s left   boxed {nb}/{want}",
             (12, show.shape[0] - 25), 0.8, (0, 0, 255) if nb < want else (0, 255, 0), 2)
        cv2.imshow(WIN, show)
        if key(1) == "q":
            early = True
            break
    if STOP["take"]:
        early = True

    if dwr is not None:
        dwr.flush()
    logf.close()
    fcsv.close()
    if raw_w is not None:
        raw_w.release()
        ann_w.release()
    duration = time.time() - t_start
    p = DecisionParams()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"take": take, "scenario": scen, "location": args.location,
                   "commander": commander, "people": people,
                   "notes": notes, "started": started,
                   "frames": n, "duration_s": round(duration, 1),
                   "mean_fps": round(fps_ema, 1), "source": args.source, "camera_profile": cam.profile,
                   "raw_format": 2, "raw_size": raw_size, "mirror": True,
                   "has_depth": has_depth, "max_people": getattr(perc, "max_people", None), "perception_mode": getattr(perc, "mode", "video"), "method_driving": None,
                   "params": {"theta_e_motion": p.theta_e_motion, "theta_e_stop": p.theta_e_stop,
                              "theta_conf": p.theta_conf, "sustain": SUSTAIN},
                   "code_version": git_commit()}, f, indent=1)
    print(f"saved {out}  ({n} frames, {'with' if has_depth else 'NO'} depth)")
    return {"frames": n, "duration": duration, "fps": n / max(duration, 1e-3), "depth": has_depth,
            "full": full_frames / max(n, 1), "want": want, "early": early}


def check(out, code, st, need_depth):
    """Quick usability check shown after each take. Returns (ok, lines)."""
    lines, ok = [], True
    lines.append(f"{st['frames']} frames in {st['duration']:.0f} s  ({st['fps']:.1f} fps)   "
                 f"depth: {'yes' if st['depth'] else 'NO'}")
    lines.append(f"all {st['want']} people boxed in {100 * st['full']:.0f}% of frames")
    if code != "CAL" and st["frames"]:
        _, eps = episodes(out)
        lines.append(f"gesture episodes found: {len(eps)}")
        if not eps:
            ok = False
            lines.append("!No gestures detected - hands visible? facing the camera?")
    if st["early"]:
        ok = False
        lines.append("!Stopped early - record it again.")
    if need_depth and not st["depth"]:
        ok = False
        lines.append("!No depth saved - re-plug the camera, then redo.")
    if st["full"] < 0.6:
        ok = False
        lines.append("!People often missing: move closer (1.5-4 m), 1 m apart, better light.")
    if st["fps"] < 5:
        lines.append(f"!Slow ({st['fps']:.1f} fps): plug in the charger, close other programs.")
    return ok, lines


# --------------------------------------------------------------------------
def next_name(day, code, practice):
    pre = "test_" if practice else ""
    if code == "CAL":
        base = f"{pre}D{day}_CAL"
        if not os.path.exists(os.path.join("takes", base)):
            return base
        r = 2
        while os.path.exists(os.path.join("takes", f"{base}_r{r}")):
            r += 1
        return f"{base}_r{r}"
    r = 1
    while os.path.exists(os.path.join("takes", f"{pre}D{day}_{code}_r{r}")):
        r += 1
    return f"{pre}D{day}_{code}_r{r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", type=int, help="session day number (1, 2, 3)")
    ap.add_argument("--location", help="e.g. lab")
    ap.add_argument("--people", help="participant IDs in this session, max 5, e.g. P01,P02,P03,P04,P05")
    ap.add_argument("--source", default="orbbec", help="orbbec (default) or webcam:0")
    ap.add_argument("--start", default="CAL", help="first scenario: CAL, S1 ... S6")
    ap.add_argument("--no-cal", action="store_true", help="skip the calibration take")
    ap.add_argument("--only", help="record just this one scenario (CAL or S1 ... S6), then stop")
    ap.add_argument("--practice", action="store_true", help="record test_ takes (not part of the dataset)")
    args = ap.parse_args()
    if args.day is None:
        args.day = int(input("Day number (1, 2, 3): ").strip())
    if not args.location:
        args.location = input("Location (e.g. lab): ").strip()
    if not args.people:
        args.people = input("Participant IDs, comma-separated (max 5, e.g. P01,P02,P03,P04,P05): ").strip()
    ppl = [x.strip().upper() for x in args.people.split(",") if x.strip()]
    if not 1 <= len(ppl) <= 5:
        sys.exit("give 1 to 5 participant IDs with --people")

    order = ["S1", "S2", "S3", "S4", "S5", "S6"]
    plan = ([] if args.no_cal or args.start != "CAL" else ["CAL"])
    first = order.index(args.start) if args.start in order else 0
    if args.only:
        args.only = args.only.upper()
        if args.only not in ["CAL"] + order:
            sys.exit("--only must be CAL or S1 ... S6")
        plan, first = [args.only], 0

    print("starting camera ...")
    cam = Camera(args.source)
    need_depth = args.source == "orbbec"
    signal.signal(signal.SIGINT, lambda *_: STOP.update(take=True, all=True))
    os.makedirs("takes", exist_ok=True)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)   # resizable; f = full screen
    cv2.resizeWindow(WIN, 1280, 720)
    done = []
    rnd, shift, k = 0, 0, first
    pending = list(plan)

    while not STOP["all"]:
        if pending:
            code = pending[0]
        else:
            code = order[k]
        A = ppl[(rnd + shift) % len(ppl)]
        B = ppl[(rnd + shift + 1) % len(ppl)] if len(ppl) > 1 else "(2nd person)"
        title, lines, ev, idle, dur = scenario(code, A, B)
        take = next_name(args.day, code, args.practice)
        people = [A] if code == "CAL" else ppl
        warn = []
        if code in ("S2", "S3", "S4", "S5", "S6") and len(ppl) < 2:
            warn.append("!This scenario needs at least 2 people (s = skip).")

        # 1. instructions
        foot = [f"Take {take}   Round {rnd + 1}   ~{dur:.0f} s   Commander {A} (c = change)   "
                f"People {','.join(people)}",
                "ENTER = start (5 s countdown)   s = skip   f = full screen   q = quit"]
        while True:
            img = panel(cam.preview(), title, lines + warn, foot)
            cv2.imshow(WIN, img)
            a = key()
            if a == "c":
                shift += 1
                A = ppl[(rnd + shift) % len(ppl)]
                B = ppl[(rnd + shift + 1) % len(ppl)] if len(ppl) > 1 else "(2nd person)"
                title, lines, ev, idle, dur = scenario(code, A, B)
                people = [A] if code == "CAL" else ppl
                foot[0] = (f"Take {take}   Round {rnd + 1}   ~{dur:.0f} s   Commander {A} (c = change)   "
                           f"People {','.join(people)}")
            if a in ("enter", "s", "q") or STOP["all"]:
                break
        if a == "q" or STOP["all"]:
            break

        if a == "enter":
            # 2. countdown, everyone into position
            t0, quit_ = time.time(), False
            while time.time() - t0 < 5:
                f = cam.preview()
                img = f.copy() if f is not None else np.zeros((540, 960, 3), np.uint8)
                n = 5 - int(time.time() - t0)
                text(img, str(n), (img.shape[1] // 2 - 30, img.shape[0] // 2 + 30), 4.0, (0, 220, 255), 8)
                text(img, "everyone into position", (img.shape[1] // 2 - 190, img.shape[0] // 2 + 90),
                     1.0, (0, 220, 255), 2)
                banner(img, [(f"FIRST: {ev[0][2]}", 1.4, (255, 200, 120), 2)])
                cv2.imshow(WIN, img)
                if key() == "q":
                    quit_ = True
                    break
            if quit_:
                continue
            beep(HI)
            # 3. record
            notes = "1-2-3-4 m: front 5s, side 5s, back 5s" if code == "CAL" else ""
            scen = "test" if args.practice else ("calibration" if code == "CAL" else "stationary")
            out = os.path.join("takes", take)
            st = record_take(cam, out, take, scen, args, A if code != "CAL" else "", people,
                             notes, ev, idle, dur)
            ok, res = check(out, code, st, need_depth)
            done.append((take, "OK" if ok else "CHECK"))
            beep(LO)
            # 4. result
            col = (0, 220, 0) if ok else (0, 0, 255)
            head = f"{'Looks good' if ok else 'Problem'}:  {take}"
            STOP["take"] = False
            while True:
                cv2.imshow(WIN, panel(cam.preview(), head, res + ["", "Write this take on the paper take log."],
                                      ["ENTER = next take    r = redo (new _r number)    q = quit"], col))
                a = key()
                if a in ("enter", "r", "q") or STOP["all"]:
                    break
            if a == "q" or STOP["all"]:
                break
            if a == "r":
                continue
        # advance (after ENTER on the result screen, or s = skip)
        if args.only:
            break
        if pending:
            pending.pop(0)
        else:
            k += 1
            if k == len(order):
                k, rnd = 0, rnd + 1

    cam.close()
    cv2.destroyAllWindows()
    print("\nThis session:")
    for t, s in done:
        print(f"  {s:6s} {t}")
    print("Next: label each take (python label_episodes.py takes/<take>), then python check_dataset.py")


if __name__ == "__main__":
    main()
