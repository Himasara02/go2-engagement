#!/usr/bin/env python3
"""Collect the portable runtime benchmark (bench_pkg/bench.py) from every device into one summary.

    python3 hardware_summary.py

Reads results/full/hardware/results_*.json (Raspberry Pi 5, laptop, desktop workstation; same code, models,
frames and library versions; CPU; GPU-delegate attempts recorded as errors). Writes
results/full/hardware/summary.{json,md}.
  networks_ms  = decode + people (one whole-frame pose pass, as the live system) + people x (hands + gesture
                 MLP + face/iris), i.e. the neural-network perception shared by every policy
  ours_ms      = everything after the networks for IT2 + gaze (tracking, cues, uncertainty, gate, sustain,
                 arbitration), measured on the networks' saved outputs (bench.py part C)
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "full", "hardware")
NAME = {"Raspberry Pi 5": "Raspberry Pi 5", "Ryzen 7 5825U": "Laptop CPU", "Ryzen 9 9950X": "Desktop CPU"}
ORDER = ["Raspberry Pi 5", "Laptop CPU", "Desktop CPU"]
rows, gpu = {}, []
for f in sorted(glob.glob(os.path.join(D, "results_*.json"))):
    r = json.load(open(f))
    sysi = r["system"]
    hw = sysi["board"] if "Raspberry" in (sysi["board"] or "") else sysi["cpu"]
    key = next(v for k, v in NAME.items() if k in hw)
    if "error" in r:
        gpu.append({"device": key, "gpu": sysi.get("gpu") or "integrated graphics", "error": r["error"][-160:]})
        continue
    ms, p = r["perception_median_ms"], r["mean_people"]
    nets = ms["decode"] + ms["people_whole_frame"] + p * (ms["hands_per_person"] + ms["gaze_per_person"])
    oe = r["our_end"]
    rows[key] = {"hardware": hw.replace("Model B Rev 1.0", "").strip(), "cores": sysi["cores"], "ram_gb": sysi["ram_gb"],
                 "gpu": sysi["gpu"], "people": p, "frames": r["frames"],
                 "networks_ms": nets, "networks_strips_ms": nets - ms["people_whole_frame"] + ms["people_strips"],
                 "ours_ms": oe["it2_gaze"]["ms_per_frame"], "type1_ms": oe["type1_gaze"]["ms_per_frame"],
                 "logreg_ms": oe["logreg"]["ms_per_frame"], "facing_only_ms": oe["facing_only"]["ms_per_frame"],
                 "gesture_only_ms": oe["gesture_only"]["ms_per_frame"],
                 "gate_us_per_person": r["decision_us_per_person"]["it2_gaze"],
                 "perception_ms": ms, "fps": 1000 / (nets + oe["it2_gaze"]["ms_per_frame"]),
                 "ours_share_pct": 100 * oe["it2_gaze"]["ms_per_frame"] / (nets + oe["it2_gaze"]["ms_per_frame"]),
                 "ours_max_fps": 1000 / oe["it2_gaze"]["ms_per_frame"],
                 "ours_core_pct_at_30fps": 3.0 * oe["it2_gaze"]["ms_per_frame"],
                 "versions": {k: sysi[k] for k in ("python", "mediapipe", "opencv", "numpy", "os")}}
out = {"devices": {k: rows[k] for k in ORDER if k in rows}, "gpu_delegate": gpu}
json.dump(out, open(os.path.join(D, "summary.json"), "w"), indent=1)
md = ["# Runtime on three devices (3 people in view, CPU, same code/models/frames/versions)", "",
      "| Device | Networks (ms) | Ours, after the networks (ms) | Type-1 instead (ms) | fps | Ours share | Ours max fps |",
      "|---|---|---|---|---|---|---|"]
md += [f"| {k} ({v['hardware']}) | {v['networks_ms']:.0f} | {v['ours_ms']:.2f} | {v['type1_ms']:.2f} | {v['fps']:.1f} | "
       f"{v['ours_share_pct']:.2f} % | {v['ours_max_fps']:.0f} |" for k, v in out["devices"].items()]
md += ["", "GPU delegate: " + "; ".join(f"{g['device']} ({g['gpu']}): failed" for g in gpu)]
open(os.path.join(D, "summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
