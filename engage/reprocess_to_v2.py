#!/usr/bin/env python3
"""Like reprocess.py, but writes the re-detected takes to a separate folder.

    python reprocess_to_v2.py                    # every take in takes/ -> takes_v2/
    python reprocess_to_v2.py takes/D1_S2_r1     # one take

takes/ is not touched. Each takes_v2/<take>/ gets the new log.jsonl and annot.mp4
(current detection), the original raw.mp4, depth/ and frames.csv (copied), and
meta.json with perception_mode / max_people added. No labels are copied: label
takes_v2 again (A and B). Takes already in takes_v2/ are skipped.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "takes_v2"


def frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def run(take):
    name = os.path.basename(os.path.normpath(take))
    dst = os.path.join(OUT, name)
    if os.path.exists(os.path.join(dst, "meta.json")):
        print(f"== {name}: already in {OUT}, skipped", flush=True)
        return True
    tmp_name = f"_v2_{name}"
    tmp = os.path.join("takes", tmp_name)
    shutil.rmtree(tmp, ignore_errors=True)
    old = json.load(open(os.path.join(take, "meta.json")))
    print(f"== {name}: re-running detection ...", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_live.py"), "--source", f"take:{take}",
                        "--take", tmp_name, "--headless", "--overwrite",
                        "--scenario", old.get("scenario", "stationary")])
    if r.returncode != 0 or not os.path.exists(os.path.join(tmp, "log.jsonl")):
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"   FAILED for {name} - nothing written")
        return False
    new = json.load(open(os.path.join(tmp, "meta.json")))
    os.makedirs(dst, exist_ok=True)
    for f in ("log.jsonl", "annot.mp4"):
        shutil.move(os.path.join(tmp, f), os.path.join(dst, f))
    # keep the original recording when it lines up frame for frame with the new log
    if frames(os.path.join(take, "raw.mp4")) == new.get("frames"):
        shutil.copy2(os.path.join(take, "raw.mp4"), os.path.join(dst, "raw.mp4"))
        shutil.copy2(os.path.join(take, "frames.csv"), os.path.join(dst, "frames.csv"))
        shutil.copytree(os.path.join(take, "depth"), os.path.join(dst, "depth"))
    else:
        for f in ("raw.mp4", "frames.csv", "depth"):
            shutil.move(os.path.join(tmp, f), os.path.join(dst, f))
    old.update(max_people=new.get("max_people"), perception_mode=new.get("perception_mode"),
               reprocessed=True, reprocessed_from=take.replace("\\", "/"),
               mean_fps_reprocess=new.get("mean_fps"), code_version=new.get("code_version"))
    json.dump(old, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"   done -> {dst}", flush=True)
    return True


def main():
    takes = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join("takes", "*"))
                                   if os.path.isdir(d) and not os.path.basename(d).startswith("_"))
    takes = [t for t in takes if os.path.exists(os.path.join(t, "meta.json"))]
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(takes)} takes -> {OUT}/ (about 1-3 min each).", flush=True)
    ok = sum(run(t) for t in takes)
    print(f"\nFinished: {ok}/{len(takes)} takes in {OUT}/. Label them again (A and B).")


if __name__ == "__main__":
    main()
