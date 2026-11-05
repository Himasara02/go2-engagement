# Guide for Claude: rebuild the 3 Nov 2026 labelled dataset

Follow these steps exactly. The goal: rebuild the full labelled dataset of 3 Nov 2026,
byte-identical to the data collector's copy, from (a) the 31 Oct data the user already has
and (b) this small update folder. Nothing has to be detected or labelled again.

## What you need

1. **This update folder**: `handover_2026-11-03_update/`, unzipped from
   `handover_2026-11-03_update.zip`. It contains `rebuild_dataset.py`, `manifest.json`, `data/`,
   `crop_boxes_takes_crop.csv.gz`, `crop_boxes_takes_crop2.csv.gz`, `code/`, this guide, `README.md`
   and `labels_summary.csv`.
2. **The 31 Oct data**, sent earlier as **`handover_himasara_2026-10-31_data.zip`** (5.1 GB).
   Either the zip file itself or the folder it was unzipped to works. Inside, it has:
   - `engage/takes/`: 20 take folders: `D1_CAL`, `D1_CAL_r2`, `D1_CAL_r3`, `D1_S1_r1`, `D1_S1_r2`,
     `D1_S1_r3`, `D1_S2_r1`, `D2_S1_r1`, `D2_S2_r1`, `D2_S3_r1`, `D2_S3_r2`, `D2_S4_r1`, `D2_S4_r2`,
     `D2_S5_r1`, `D2_S5_r2`, `D2_S6_r1`, `D2_S6_r2`, `test_2`, `test_D2_S2_r1`, `test_win_1`.
     Each one holds `raw.mp4`, `annot.mp4`, `log.jsonl`, `meta.json`, `frames.csv`, `depth/*.npz`,
     and for labelled takes also `labels.json` / `labels_B.json`.
   - `engage/takes_discarded/`: 2 stopped takes (`D1_S2_r1`, `D2_S2_r1`).
   - `handover_himasara_2026-10-31/` (report, image-mode test) and `WINDOWS_COMPATIBILITY_REPORT.md`.
     These aren't needed here.
3. Python 3.9 or newer. The rebuild uses only the standard library. Step 4 (crop images) also needs
   `opencv-python`, which is already installed in the project's `.venv`.
4. About 7 GB of free disk space.

## Steps

**Step 1. Find the 31 Oct data.** Look for `handover_himasara_2026-10-31_data.zip`, or a folder that
contains `engage/takes/D1_CAL_r3/meta.json`. Check the Dog_model project folder, its parent folder,
Downloads and any external drive. If you can't find it, **stop and ask the user where it is.** The
project's own `engage/takes/` only works if it's still the unchanged 31 Oct data; the script checks
every file and refuses anything that doesn't match.

**Step 2. Run the rebuild.** From inside `handover_2026-11-03_update/`:

```
python rebuild_dataset.py --old "<path to the 31 Oct zip or its unzip folder>" --out "<new empty folder>"
```

Suggested `--out`: a new folder next to the project, e.g. `../dataset_2026-11-03`. It must not
exist yet, or must be empty. It takes about 2–5 minutes.

**Step 3. Check the output.** A successful run prints, apart from the progress lines:

```
step 1/3  31 Oct data found: all 235 needed files are there
step 2/3  490 files written
step 3/3  verified: all 490 files are byte-identical to the 3 Nov dataset (6.42 GB)

EXACT REPLICA OK -> <out>
```

and exits with code 0. Report this to the user. Step 3 compares the size and SHA-256 of every file
with `manifest.json`, and fails if any file differs or an extra file is present.

- `STOP: N of 235 files are missing from the 31 Oct data` (exit code 2): the folder or zip given
  isn't the complete 31 Oct data. Show the user the listed files and ask for the right copy.
- `step 3/3 FAILED: N files differ` (exit code 1): a file in the 31 Oct copy has been changed since
  it was sent, e.g. a label was edited or a take was reprocessed in place. Show the user the list.
  **Don't edit the manifest or the files to make it pass.**

**Step 4 (optional, only if the user wants the crop images).** Run the same command with a new
empty `--out` and `--export-crops <take> [<take> ...]`, e.g. `--export-crops D1_S2_r1`. This writes
one PNG per person per frame, for both crop versions, into
`<out>/engage/takes_crop*/<take>/crops/f<frame>_id<track id>.png`. These are the exact images the
hand detector was run on. `--export-crops` with no take names exports every take: ~24,000 images,
about 10 GB, so ask the user first.

## What the result contains

`<out>/engage/`:

| Folder | What it is | Labels |
|---|---|---|
| `takes/` | Original recordings (pose + hands in VIDEO mode, 1 person tracked); unchanged since 31 Oct | A `labels.json`, B `labels_B.json` (44 clips) |
| `takes_v2/` | Himasara's FIX_31OCT_v2 (pose + hands in IMAGE mode), re-run on the same recordings | A (34 clips), B (31 clips, 4 of 5 takes) |
| `takes_crop/` | takes_v2 people + gestures from a crop around each person | `labels_cmd.json`: commander IDs chosen per take (295 clips) |
| `takes_crop2/` | takes_v2 people + gestures from a square crop centred on each person's upper body | A `labels.json`, every clip, with `_missed` (253 clips) |
| `takes_discarded/` | 2 stopped takes, unchanged | none |

**Main label set: `takes_crop2/` (annotator A).** Per-take counts are in `labels_summary.csv`.

Left out on purpose: the `annot.mp4` overlay videos of `takes_v2/` (20 files, 287 MB) and the debug
video `takes/D1_S2_r1/_people_check.mp4`. Neither is used for labelling or evaluation. The full
list is under `left_out` in `manifest.json`. The overlays can be regenerated with
`code/engage/reprocess_to_v2.py`.

## Using the dataset with the project tools

The labelling tools take a take folder as their argument, so run them from the project's
`engage/` folder:

```
python label_episodes.py "<out>/engage/takes_crop2/D1_S2_r1"                 # view / continue labels (A)
python label_episodes.py "<out>/engage/takes_crop2/D1_S2_r1" --annotator B   # second annotator
```

`check_dataset.py` scans `./takes/*` only. To check another folder, run it from a copy where that
folder is named `takes/`.

## How the crop gestures were made (for the paper / methods)

- People, track IDs and boxes come from the takes_v2 `log.jsonl`. They're unchanged in both crop
  versions, so a track ID means the same person in takes_v2, takes_crop and takes_crop2.
- `code/engage/find_gestures_crop.py` (`takes_crop`): box + 35 % width / 10 % height margin,
  clipped to the frame; enlarged to ≥ 640 px tall.
- `code/engage/find_gestures_centered.py` (`takes_crop2`): square crop, side
  `max(2.2 × box width, 0.7 × box height)`, centred on the upper body (`y0 + 0.35 × box height`),
  black outside the frame, resized to 640×640.
- On each crop: MediaPipe HandLandmarker (IMAGE, 2 hands, detection/presence 0.3) + the project's
  keypoint MLP. A hand belongs to the person only if its wrist lies inside their box (+25 % width)
  and no other person's box fits better. Per person, a command-class hand with conf ≥ 0.4 is
  preferred over a relaxed hand.
- `crop_boxes_takes_crop*.csv.gz`, one row per person per frame:
  - `x, y, w, h`: the crop rectangle in full-resolution pixels of the horizontally mirrored raw.mp4
    frame (`mirrored` = 1). It may extend past the frame edge: pad with black.
  - `out_w, out_h`: the size fed to the detector.
  - `scale`: set when the crop was enlarged by that factor (`cv2.resize(fx=fy=scale, INTER_CUBIC)`);
    otherwise it was resized to `out_w × out_h` (cubic when enlarging, area when shrinking).
  - Tested: crops exported this way are pixel-identical to the detector's crops.
- `perception.py` was **not** changed. The live system still uses full-frame hands.
