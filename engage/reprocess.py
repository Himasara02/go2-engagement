#!/usr/bin/env python3
"""Re-run detection on takes that were already recorded (no new recording needed).

Use this after the detection fix (every frame detected from scratch, up to 6 people). It reads each take's
saved raw.mp4 + depth, runs the current pipeline again, and replaces
log.jsonl and annot.mp4 in place. raw.mp4, depth/, frames.csv and the take
details are kept.

Because the detected gestures change, any existing labels no longer match.
They are moved to takes/<take>/labels_old/ and the take must be labelled again
(annotator A and B).

    python reprocess.py                  # all takes that need it
    python reprocess.py takes/D2_S3_r1   # one take
"""
import glob
import json
import os
import shutil
import subprocess
import sys

from engage.perception import Perception

HERE = os.path.dirname(os.path.abspath(__file__))
MAXP = Perception.__init__.__defaults__[0]


def needs(take):
    m = json.load(open(os.path.join(take, "meta.json")))
    return m.get("raw_format") == 2 and (m.get("max_people", 4) < MAXP or m.get("perception_mode") != "image")


def run(take):
    name = os.path.basename(os.path.normpath(take))
    tmp_name = f"_reprocess_{name}"
    tmp = os.path.join("takes", tmp_name)
    shutil.rmtree(tmp, ignore_errors=True)
    meta_path = os.path.join(take, "meta.json")
    old = json.load(open(meta_path))
    print(f"== {name}: re-running detection (up to {MAXP} people) ...", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_live.py"), "--source", f"take:{take}",
                        "--take", tmp_name, "--headless", "--overwrite",
                        "--scenario", old.get("scenario", "stationary")])
    if r.returncode != 0 or not os.path.exists(os.path.join(tmp, "log.jsonl")):
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"   FAILED for {name} - nothing changed")
        return False
    # move old labels aside, they belong to the old detections
    lab = [f for f in ("labels.json", "labels_B.json", "labels_C.json") if os.path.exists(os.path.join(take, f))]
    if lab:
        os.makedirs(os.path.join(take, "labels_old"), exist_ok=True)
        for f in lab:
            shutil.move(os.path.join(take, f), os.path.join(take, "labels_old", f))
    for f in ("log.jsonl", "annot.mp4"):
        shutil.move(os.path.join(tmp, f), os.path.join(take, f))
    new = json.load(open(os.path.join(tmp, "meta.json")))
    old.update(max_people=MAXP, perception_mode=new.get("perception_mode"), reprocessed=True, mean_fps_reprocess=new.get("mean_fps"),
               code_version=new.get("code_version"))
    json.dump(old, open(meta_path, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"   done. {'Old labels moved to labels_old/ - label this take again.' if lab else ''}")
    return True


def main():
    takes = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join("takes", "*"))
                                   if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
    todo = [t for t in takes if os.path.exists(os.path.join(t, "meta.json")) and (sys.argv[1:] or needs(t))]
    if not todo:
        print("Nothing to reprocess.")
        return
    print(f"{len(todo)} takes to reprocess (about 1-3 min each).")
    ok = sum(run(t) for t in todo)
    print(f"\nFinished: {ok}/{len(todo)} takes reprocessed. Now label them again (A and B), then run check_dataset.py.")


if __name__ == "__main__":
    main()
