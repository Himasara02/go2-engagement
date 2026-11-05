# Data + labels update: 3 Nov 2026

For Himasara. It adds to `handover_himasara_2026-10-31_data.zip`: the same recordings, with no new takes.

**To get the full labelled dataset, ask Claude to follow `GUIDE_FOR_CLAUDE.md`**, or run:

```
python rebuild_dataset.py --old <the 31 Oct zip or its unzip folder> --out ../dataset_2026-11-03
```

This update is small because every recording (raw.mp4, depth, frames.csv) is already in the 31 Oct
zip. Here are only the new logs, meta files and labels, a manifest with the SHA-256 of all 490 files
in the dataset, the exact crop rectangles (`crop_boxes_*.csv.gz`), and the code (`code/engage/`).
The rebuild was tested against the real 31 Oct zip and produced a byte-identical copy of all 490
files (6.42 GB). Only the takes_v2 overlay videos (annot.mp4) and one debug video are left out.

## The four data folders

| Folder | What detected the people and gestures | Labels | Clips (r / n / x) |
|---|---|---|---|
| `takes/` | Original recording, pose + hands in VIDEO mode (1 person tracked) | A `labels.json`, B `labels_B.json` | A 44 (26/18/0), B 44 (25/19/0) |
| `takes_v2/` | Your FIX_31OCT_v2: pose + hands in IMAGE mode (`reprocess_to_v2.py`) | A, B (B stopped after 4 of 5 takes) | A 34 (23/11/0), B 31 (19/12/0) |
| `takes_crop/` | takes_v2 people + gestures from a crop per person (`find_gestures_crop.py`) | `labels_cmd.json`: commander chosen per take, see below | 295 (122/173/0) |
| `takes_crop2/` | takes_v2 people + gestures from a centred crop per person (`find_gestures_centered.py`) | A `labels.json`, every clip labelled, with `_missed` | 253 (101/141/11), missed 15 |

**Use `takes_crop2` (A labels)** as the most complete set. It's labelled clip by clip, so S4 "gesture to a friend" clips are correct there, and it has the end-of-take missed-commands count.

## What changed since 31 Oct and why

1. **Hands are missed in full-frame IMAGE mode.** v2 found the people (3 per frame in D1_S2_r1), but HandLandmarker in IMAGE mode on the whole frame saw few hands. In D1_S2_r1 the commander's command gesture was read in only 16 of 911 frames.
2. **Gestures per person from crops (labelling aid only; `perception.py` is NOT changed).** The people, IDs and boxes come from the takes_v2 log. For each person in each frame, the person is cut out of the full-resolution raw.mp4 (1280×720), enlarged, and HandLandmarker (IMAGE) + the keypoint MLP run on that crop. A hand counts for a person only if its wrist is inside their box and no other box fits better. Per person, a command-class hand with conf ≥ 0.4 is preferred over a relaxed hand.
   - `takes_crop`: box + margin, clipped at the frame edge.
   - `takes_crop2`: square crop centred on the person's upper body (head to below the hips) with a margin, black-padded at the frame edges. A whole-body square crop made hands ~10 px after MediaPipe's 192 px resize and everything read as "dislike", so the crop covers the upper body only.
   - Command-gesture person-frames, full frame → crop: D1_S2_r1 51 → 486, D2_S3_r1 73 → 560, D2_S2_r1 2 → 310, D2_S3_r2 0 → 257. Commander #4 in D1_S2_r1: 25 frames → 248 (crop) / 174 (centred crop), in 30 / 29 separate bursts.
   - **Suggestion for the live system:** per-person crops for the hand stage find ~10× more gestures. That's a `perception.py` change, so it's your decision.
3. **New labelling tool `label_commander.py`** (`takes_crop`): pick the commander's track ID(s) once per take; their clips get r, everyone else's n. It's fast, but it marks S4 commander-to-friend gestures r as well, and `check_dataset.py` doesn't read `labels_cmd.json`. In `takes_crop/D2_S3_r2` no commander was chosen (all n).
4. Your 1 Nov update (missed-commands question) is merged, keeping the UTF-8 fix in `check_dataset.py`. The 2 Nov `label_moments.py` was checked but not installed: takes_v2 had 0 moments with 2+ people gesturing at once.

## Known problems that remain

- The classifier often reads call as "mute" (both backward) and rarely gives "one"/"two_up". Relaxed hands are mostly "dislike". This doesn't change r/n, but it does change which command would be executed.
- Track IDs change often (up to #76 in D2_S6_r1), so the commander has several IDs in a take.
- Day-2 takes are 3–7 fps (laptop on battery).
- Participant IDs in meta.json are placeholders (P01–P04); the real ones are on the paper log.
- Only one annotator (A) labelled `takes_crop2`; a B pass is still needed for agreement.

## Code in `code/engage/` (new or changed by us)

- `find_gestures_crop.py`, `find_gestures_centered.py`: gestures from per-person crops (new).
- `label_commander.py`: choose the commander per take (new).
- `reprocess_to_v2.py`: re-run detection into a separate folder (new).
- `rebuild_dataset.py` (top of this folder): rebuild + verify the dataset from the 31 Oct data (new).
- `record_session.py`, `gesture_test.py`, `setup.ps1`: recording tools (from before).
- `perception.py`, `label_episodes.py`, `check_dataset.py`, `reprocess.py`, `count_people.py`: your versions (31 Oct v2 / 1 Nov), plus Windows fixes only.
