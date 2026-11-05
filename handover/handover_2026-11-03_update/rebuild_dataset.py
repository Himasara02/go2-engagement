#!/usr/bin/env python3
"""Rebuild the labelled dataset of 3 Nov 2026 from the 31 Oct data + this small update.

The 31 Oct handover (handover_himasara_2026-10-31_data.zip) already holds every
recording (raw.mp4, depth/*.npz, frames.csv). This update only carries what is new:
logs, meta.json and labels for takes_v2/, takes_crop/, takes_crop2/. This script
copies each big file from the 31 Oct data, adds the new small files, and then checks
EVERY file of the result against manifest.json (size + SHA-256).

    python rebuild_dataset.py --old <31 Oct data: the .zip, or the folder it was unzipped to>
                              [--out dataset_2026-11-03] [--export-crops [TAKE ...]]

Result: <out>/engage/{takes, takes_v2, takes_crop, takes_crop2, takes_discarded}/...
--export-crops also writes the exact crop images the hand detector saw, as PNG, to
<out>/engage/takes_crop*/<take>/crops/ (from crop_boxes_*.csv.gz). Name takes to export only
those (e.g. --export-crops D1_S2_r1); with no names it exports all ~24,000 crops (~10 GB).
Exit code 0 = exact replica.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "engage/takes/D1_CAL_r3/meta.json"          # a file every 31 Oct copy has


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class OldData:
    """The 31 Oct data, either the zip itself or the folder it was unzipped to."""

    def __init__(self, path):
        self.zip = self.root = None
        if os.path.isfile(path) and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            names = set(self.zip.namelist())
            pre = [n[:-len(MARK)] for n in names if n.endswith(MARK)]
            if not pre:
                sys.exit(f"{path} is not the 31 Oct data zip (no {MARK} inside)")
            self.prefix = pre[0]
        else:
            # accept the unzip folder, its engage/ folder, or engage/takes itself
            for cand in (path, os.path.join(path, ".."), os.path.join(path, "..", "..")):
                if os.path.exists(os.path.join(cand, *MARK.split("/"))):
                    self.root = os.path.normpath(cand)
                    break
            else:
                # one level deeper (e.g. unzipped into a folder named after the zip)
                for sub in os.listdir(path) if os.path.isdir(path) else []:
                    if os.path.exists(os.path.join(path, sub, *MARK.split("/"))):
                        self.root = os.path.join(path, sub)
                        break
            if self.root is None:
                sys.exit(f"could not find the 31 Oct data in {path} (looked for {MARK})")

    def where(self):
        return self.zip.filename if self.zip else self.root

    def exists(self, rel):
        if self.zip:
            try:
                self.zip.getinfo(self.prefix + rel)
                return True
            except KeyError:
                return False
        return os.path.isfile(os.path.join(self.root, *rel.split("/")))

    def copy_to(self, rel, dst):
        if self.zip:
            with self.zip.open(self.prefix + rel) as s, open(dst, "wb") as d:
                shutil.copyfileobj(s, d, 1 << 20)
        else:
            shutil.copyfile(os.path.join(self.root, *rel.split("/")), dst)


def export_crops(out, upd, only=None):
    import cv2
    for folder in ("takes_crop", "takes_crop2"):
        rows = {}
        with gzip.open(os.path.join(upd, f"crop_boxes_{folder}.csv.gz"), "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if only and r["take"] not in only:
                    continue
                rows.setdefault(r["take"], {}).setdefault(int(r["frame"]), []).append(r)
        for take, by_frame in sorted(rows.items()):
            d = os.path.join(out, "engage", folder, take)
            os.makedirs(os.path.join(d, "crops"), exist_ok=True)
            cap = cv2.VideoCapture(os.path.join(d, "raw.mp4"))
            i, n = 0, 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                for r in by_frame.get(i, []):
                    im = cv2.flip(img, 1) if r["mirrored"] == "1" else img
                    H, W = im.shape[:2]
                    x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
                    crop = im[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
                    # parts outside the frame are black, as in find_gestures_centered.py
                    crop = cv2.copyMakeBorder(crop, max(0, -y), max(0, y + h - H), max(0, -x), max(0, x + w - W),
                                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    ow, oh = int(r["out_w"]), int(r["out_h"])
                    if r["scale"]:      # enlarged by a factor, as in find_gestures_crop.py
                        k = float(r["scale"])
                        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                    elif (ow, oh) != (crop.shape[1], crop.shape[0]):
                        crop = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_CUBIC
                                          if ow > crop.shape[1] else cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(d, "crops", f"f{i:05d}_id{r['tid']}.png"), crop)
                    n += 1
                i += 1
            cap.release()
            print(f"  {folder}/{take}: {n} crops", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="31 Oct data: the zip, or the folder it was unzipped to")
    ap.add_argument("--out", default="dataset_2026-11-03", help="where to build the dataset (must be new/empty)")
    ap.add_argument("--export-crops", nargs="*", metavar="TAKE",
                    help="also write the crop images (not in the manifest); optionally only these takes")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    files = man["files"]
    old = OldData(a.old)
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty - choose a new --out folder")
    print(f"31 Oct data: {old.where()}\nbuilding:    {out}\n")

    # 1. everything needed from the 31 Oct data must be there before anything is copied
    need = sorted({f["old"] for f in files.values() if "old" in f})
    missing = [r for r in need if not old.exists(r)]
    if missing:
        print(f"STOP: {len(missing)} of {len(need)} files are missing from the 31 Oct data, e.g.:")
        for r in missing[:10]:
            print("   ", r)
        sys.exit(2)
    print(f"step 1/3  31 Oct data found: all {len(need)} needed files are there")

    # 2. build: big files from the 31 Oct data (each read once), small files from this update
    done = {}
    for k, (rel, f) in enumerate(sorted(files.items()), 1):
        dst = os.path.join(out, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if "old" in f:
            if f["old"] in done:
                shutil.copyfile(done[f["old"]], dst)
            else:
                old.copy_to(f["old"], dst)
                done[f["old"]] = dst
        else:
            shutil.copyfile(os.path.join(HERE, "data", *rel.split("/")), dst)
        if k % 100 == 0:
            print(f"   {k}/{len(files)} files", flush=True)
    print(f"step 2/3  {len(files)} files written")

    # 3. verify every file, and that nothing extra is there
    bad = [rel for rel, f in sorted(files.items())
           if os.path.getsize(os.path.join(out, *rel.split("/"))) != f["size"]
           or sha256_file(os.path.join(out, *rel.split("/"))) != f["sha256"]]
    have = {os.path.relpath(os.path.join(d, x), out).replace("\\", "/")
            for d, _, fs in os.walk(out) for x in fs}
    extra = sorted(have - set(files))
    if bad or extra:
        print(f"step 3/3  FAILED: {len(bad)} files differ, {len(extra)} unexpected files")
        for r in (bad + extra)[:20]:
            print("   ", r)
        sys.exit(1)
    print(f"step 3/3  verified: all {len(files)} files are byte-identical to the 3 Nov dataset "
          f"({sum(f['size'] for f in files.values()) / 1e9:.2f} GB)")
    print(f"\nEXACT REPLICA OK -> {out}")

    if a.export_crops is not None:
        print("\nexporting crop images ...")
        export_crops(out, HERE, set(a.export_crops))


if __name__ == "__main__":
    main()
