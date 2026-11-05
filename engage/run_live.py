#!/usr/bin/env python3
"""Record a take (or replay one) with the engagement-gated gesture pipeline.

All policies run every frame on the same features ("shadow mode"). Nothing
moves the robot unless --drive is given.

Every take writes takes/<take>/:
  raw.mp4      colour exactly as the camera gave it (native size, not mirrored)
  depth/       aligned depth in mm (Orbbec only), see engage/takeio.py
  frames.csv   per-frame timestamps
  annot.mp4    overlay video (for the website / poster, NOT for labelling)
  log.jsonl    per-frame per-person features + decisions
  meta.json    who / where / what / settings

Sources
  --source orbbec              Orbbec RGB-D camera on the dog (main setup)
  --source go2 --ip <IP>       the dog's own front camera over WebRTC
  --source webcam:0            laptop webcam (testing)
  --source take:takes/<name>   replay a recorded take (colour + depth)
  --source file:<video>        replay an old-format raw.mp4 (add --no-mirror)

Recording example
  python run_live.py --source orbbec --take D1_S2_r1 --scenario stationary \\
      --location lab --commander P03 --people P01,P03,P05 --notes "P05 partly hidden"

Keys in the window: q / ESC (or Ctrl+C) = stop and save.  (With --drive: SPACE toggles
driving, s = stop the robot now.)
"""
import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import cv2

from engage.fuzzy import DecisionParams
from engage.methods import (COMMAND_MAP, METHODS, accept_per_person, arbitrate,
                            engagement)
from engage.perception import Perception
from engage.takeio import DepthWriter, TakeReader, to_proc_view

SUSTAIN = 3          # frames a decision must persist before the robot acts
CMD_HZ = 10.0


def open_source(args):
    """Returns read() -> (ok, bgr_raw, depth_mm|None, t|None), info dict, link."""
    info = {"mirror": not args.no_mirror, "file_fps": None, "profile": None}
    link = None
    if args.source == "orbbec":
        from engage.orbbec import OrbbecSource
        try:
            cam = OrbbecSource()
        except RuntimeError as e:
            sys.exit(f"Orbbec camera problem: {e}")
        info["profile"] = cam.profile
        info["close"] = cam.close
        read = cam.read
    elif args.source == "go2":
        from engage.go2_link import Go2Link
        link = Go2Link(args.ip, ap_mode=args.ap)
        if not link.ok:
            sys.exit(f"Go2 WebRTC connection failed: {link.err}")
        seen = {"t": 0.0}

        def read():
            if link.latest_frame is None or link.frame_t == seen["t"]:
                return False, None, None, None
            seen["t"] = link.frame_t
            return True, link.latest_frame, None, link.frame_t
    elif args.source.startswith("take:"):
        tdir = args.source.split(":", 1)[1]
        it = iter(TakeReader(tdir))
        ts = []
        fcsv = os.path.join(tdir, "frames.csv")
        if os.path.exists(fcsv):
            ts = [float(r["t"]) for r in csv.DictReader(open(fcsv))]
        info["file_fps"] = 12.0
        info["replay_times"] = ts

        def read():
            try:
                i, bgr, depth = next(it)
            except StopIteration:
                return False, None, None, None
            t = ts[i] if i < len(ts) else i / 12.0
            return True, bgr, depth, t
    else:
        if args.source.startswith("webcam:"):
            cap = cv2.VideoCapture(int(args.source.split(":")[1]))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(args.source.split(":", 1)[1])
            info["file_fps"] = cap.get(cv2.CAP_PROP_FPS) or 12.0
        n = {"i": 0}

        def read():
            ok, bgr = cap.read()
            t = n["i"] / info["file_fps"] if info["file_fps"] else None
            n["i"] += 1
            return ok, bgr, None, t
    if args.drive and link is None and args.ip:
        from engage.go2_link import Go2Link
        link = Go2Link(args.ip, ap_mode=args.ap, video=False)
    return read, info, link


def draw(img, people, accepted_it2, driving, active):
    for p in people:
        x0, y0, x1, y1 = p["bbox"]
        ok = p["tid"] in accepted_it2
        col = (40, 200, 40) if ok else ((0, 160, 255) if p.get("gesture") in COMMAND_MAP else (180, 180, 180))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        g = p.get("gesture") or "-"
        lines = [f"#{p['tid']} {p['distance']:.1f}m face {p['facing']:.2f}",
                 f"E[{p['e2_low']:.2f},{p['e2_high']:.2f}] T1 {p['e_t1']:.2f}",
                 f"{g} {p['gesture_conf']:.2f}" + ("  ACCEPT" if ok else "")]
        for i, s in enumerate(lines):
            y = max(18, y0 - 44 + 16 * i)
            cv2.putText(img, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    st = f"{'DRIVING' if driving else 'RECORDING (robot not controlled)'}  cmd: {active or 'none'}"
    cv2.putText(img, st, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if driving else (255, 255, 255), 2, cv2.LINE_AA)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="orbbec")
    ap.add_argument("--ip", default=None, help="dog IP (only for --source go2 or --drive)")
    ap.add_argument("--ap", action="store_true", help="laptop on the dog's own hotspot")
    ap.add_argument("--take", required=True, help="take name, e.g. D1_S2_r1")
    ap.add_argument("--scenario", default="stationary", choices=["stationary", "walking", "calibration", "test"])
    ap.add_argument("--location", default="")
    ap.add_argument("--commander", default="", help="participant ID of the commander, e.g. P03")
    ap.add_argument("--people", default="", help="comma-separated participant IDs in the take")
    ap.add_argument("--notes", default="")
    ap.add_argument("--method", default="it2", choices=METHODS)
    ap.add_argument("--drive", action="store_true", help="let the system drive the dog (needs --ip)")
    ap.add_argument("--vmax", type=float, default=0.3)
    ap.add_argument("--wmax", type=float, default=0.6)
    ap.add_argument("--no-mirror", action="store_true", help="input already mirrored (old-format replay)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="allow reusing an existing take name")
    args = ap.parse_args()

    out = os.path.join("takes", args.take)
    if os.path.exists(os.path.join(out, "meta.json")) and not args.overwrite:
        sys.exit(f"{out} already exists. Use a new take name (e.g. _r2), or --overwrite to redo it.")
    os.makedirs(out, exist_ok=True)
    read, info, link = open_source(args)
    live = info["file_fps"] is None
    perc = Perception()
    logf = open(os.path.join(out, "log.jsonl"), "w")
    fcsv = open(os.path.join(out, "frames.csv"), "w", newline="")
    fw = csv.writer(fcsv)
    fw.writerow(["i", "wall", "t"])
    dwr = None
    raw_w = ann_w = None
    driving = bool(args.drive and link is not None)
    sustain = {m: {} for m in METHODS}
    last_cmd_t, last_motion, active = 0.0, False, None
    t_start = time.time()
    started = dt.datetime.now().isoformat(timespec="seconds")
    n, fps_ema, raw_size, has_depth = 0, 0.0, None, False
    stop = {"now": False}       # Ctrl+C stops and saves like q / ESC
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    while not stop["now"]:
        ok, raw, depth, tcam = read()
        if not ok or raw is None:
            if not live:
                break
            time.sleep(0.005)
            continue
        wall = time.time()
        t = (tcam if not live else wall - t_start)
        frame, dproc = to_proc_view(raw, depth, mirror=info["mirror"])
        tic = time.time()
        feats, _ = perc.process(frame, dproc, t=t if not live else None)
        people = []
        for f in feats:
            d = f.as_dict()
            d.update(engagement(d))
            people.append(d)
        acc = accept_per_person(people)
        sus = {}
        for m in METHODS:
            cnt = sustain[m]
            for tid in list(cnt):
                if tid not in acc[m]:
                    del cnt[tid]
            for tid in acc[m]:
                cnt[tid] = cnt.get(tid, 0) + 1
            sus[m] = {tid for tid, c in cnt.items() if c >= SUSTAIN}

        tid, label = arbitrate(people, sus[args.method], args.method)
        active = f"#{tid} {label}" if tid else None
        now = time.time()
        if link is not None and driving and now - last_cmd_t >= 1.0 / CMD_HZ:
            last_cmd_t = now
            if label is not None:
                fx, fz = COMMAND_MAP[label]
                if fx == 0 and fz == 0:
                    link.stop()
                    last_motion = False
                else:
                    link.move(fx * args.vmax, fz * args.wmax)
                    last_motion = True
            elif last_motion:
                link.stop()
                last_motion = False

        logf.write(json.dumps({"i": n, "t": round(t, 3), "people": people,
                               "acc": {m: sorted(v) for m, v in acc.items()},
                               "sus": {m: sorted(v) for m, v in sus.items()},
                               "drive": {"tid": tid, "label": label, "on": driving}}) + "\n")
        fw.writerow([n, f"{wall:.3f}", f"{t:.3f}"])

        if raw_w is None:
            raw_size = (raw.shape[1], raw.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rate = info["file_fps"] or 12.0
            raw_w = cv2.VideoWriter(os.path.join(out, "raw.mp4"), fourcc, rate, raw_size)
            ann_w = cv2.VideoWriter(os.path.join(out, "annot.mp4"), fourcc, rate,
                                    (frame.shape[1], frame.shape[0]))
        raw_w.write(raw)
        if depth is not None:
            has_depth = True
            if dwr is None:
                dwr = DepthWriter(out)
            dwr.add(n, depth)
        vis = frame.copy()
        draw(vis, people, sus["it2"], driving, active)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(time.time() - tic, 1e-3))
        cv2.putText(vis, f"{fps_ema:.1f} fps  {args.take}  {len(people)} people"
                    + ("  depth" if depth is not None else "  NO DEPTH"), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ann_w.write(vis)
        n += 1
        if not args.headless:
            cv2.imshow("engage", vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" ") and link is not None:
                driving = not driving
                if not driving:
                    link.stop()
            if k == ord("s") and link is not None:
                driving = False
                link.stop()

    if link is not None:
        link.stop()
    if "close" in info:
        info["close"]()
    if dwr is not None:
        dwr.flush()
    logf.close()
    fcsv.close()
    if raw_w is not None:
        raw_w.release()
        ann_w.release()
    p = DecisionParams()
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({"take": args.take, "scenario": args.scenario, "location": args.location,
                   "commander": args.commander,
                   "people": [x.strip() for x in args.people.split(",") if x.strip()],
                   "notes": args.notes, "started": started,
                   "frames": n, "duration_s": round(time.time() - t_start if live else n / (info["file_fps"] or 12), 1),
                   "mean_fps": round(fps_ema, 1), "source": args.source, "camera_profile": info["profile"],
                   "raw_format": 2 if info["mirror"] else 1, "raw_size": raw_size, "mirror": info["mirror"],
                   "has_depth": has_depth, "max_people": getattr(perc, "max_people", None), "perception_mode": getattr(perc, "mode", "video"), "method_driving": args.method if args.drive else None,
                   "params": {"theta_e_motion": p.theta_e_motion, "theta_e_stop": p.theta_e_stop,
                              "theta_conf": p.theta_conf, "sustain": SUSTAIN},
                   "code_version": git_commit()}, f, indent=1)
    print(f"saved {out}  ({n} frames, {'with' if has_depth else 'NO'} depth)")


if __name__ == "__main__":
    main()
