# ICAISC 2027: plan and research tasks

**Submit Thu 12 Nov** at icaiscsystem.icaisc.eu. Deadline Sun 15 Nov (the day after the last exam, so backup only).
Data collection is finished (3 Nov). **Current task list: `START_HERE.md` section 3** (it replaces section 3 below).

## 1. Timeline

| Date | What happens |
|---|---|
| Sat 24 Oct | Handover package to the collector. LaTeX project in Overleaf. |
| Sun 25 Oct | Collector: setup and tests. |
| Mon 26 – Wed 28 Oct | Collector: 3 recording sessions + labelling (A and B) each evening. |
| Thu 29 Oct | Collector: `check_dataset.py` clean, then handover (SSD + Drive). |
| **Sat 31 Oct** | Himasara + Claude Code: tasks 5–12 below. Write Method / Experiments. |
| **Tue 3 Nov** | Write Results, Discussion, Conclusion; finish Abstract + Intro with real numbers. Full 12-page draft. |
| **Fri 6 – Sat 7 Nov** | Teammates + a lecturer review; fix. |
| **Thu 12 Nov** | Anonymity check, references, PDF check. **Submit.** |

Exam days (keep the day before free): 29 Oct, 30 Oct, 2 Nov, 5 Nov, 9 Nov, 11 Nov, 14 Nov.

## 2. Already done (before recording)

| # | Task | Where |
|---|---|---|
| 1 | Orbbec RGB-D source, depth aligned to colour | `engage/engage/orbbec.py`, `run_live.py --source orbbec`, `check_orbbec.py` |
| 2 | Raw colour (native, unmirrored) + depth (mm, half-res, chunked .npz) + timestamps saved per take; offline reader | `engage/engage/takeio.py`; replay with `run_live.py --source take:takes/<name>` |
| 3 | Take metadata (scenario, location, commander, people, notes, params, code version); annotator B mode; labeller plays clips with no system overlay | `run_live.py`, `label_episodes.py --annotator B` |
| 4 | Dataset completeness check + report | `check_dataset.py` |
| — | Settings frozen: θ_E 0.60 / 0.45 (STOP), θ_c 0.45, sustain 3 frames; recorded in every `meta.json` | `engage/engage/fuzzy.py`, `run_live.py` |

Not done: WebRTC driving (task 2b, not needed: the dog is driven with the remote during collection) and L1 LiDAR recording (3b, optional; the paper can do without it).

## 3. Research tasks after the data arrives (Claude Code)

Open Claude Code in `~/go2-engagement`. Do these in order. **Never tune anything on test takes.** If a setting changes, rerun every method on every take offline.

| # | Task | Done when | Prompt |
|---|---|---|---|
| 5 | **Offline replay** | `replay.py` reruns perception (colour + depth) on every take via `takeio.TakeReader` and writes `takes/<take>/replay/log.jsonl`; episodes and labels still line up (same frame indices) | "Write replay.py that recomputes all features for all takes from raw.mp4 + depth using TakeReader, keeping frame indices identical so labels still match." |
| 6 | **Gaze module** | Gaze score + σ_gaze per person (σ grows as eye distance in pixels shrinks); 4th fuzzy input with rules | "Add a gaze input from MediaPipe FaceLandmarker iris landmarks on the face crop, with distance-dependent sigma, as a 4th input to the Type-1 and IT2 systems." |
| 7 | **Dev/test split** | `splits.json`: first 3 non-calibration takes of D1 = dev, the rest = test | "Create splits.json and make evaluate.py report test-only numbers." |
| 8 | **Six policies** | gesture-only, nearest, **gaze-only**, **logistic regression (trained on dev)**, Type-1, IT2 | "Add gaze-only and a logistic-regression baseline to methods.py and evaluate.py." |
| 8b | **Missed commands** | Each labels file has `_missed` = robot-directed gestures the detector never turned into a clip. Report perception recall separately and add these to the recall denominator for all policies | "Use the _missed counts in labels*.json as undetected robot-directed gestures in the recall denominator." |
| 9 | **Statistics** | Wilson CIs, McNemar IT2 vs T1, Cohen's κ between labels.json and labels_B.json; disagreements listed for a third person to decide | "Add McNemar and Cohen's kappa to evaluate.py and write a disagreements list." |
| 10 | **Analysis** | θ_E sweep curves; results by σ (low/high), distance, scenario; noise injection; ablations (no gaze, facing only, distance only, FOU scale k) | "Add the operating curve, sigma-bin breakdown, noise test and ablations." |
| 11 | **Distance sources** | IT2 with σ_d from (a) monocular estimate (ignore depth), (b) Orbbec depth; calibration takes give distance/facing ground truth at 1–4 m | "Compare monocular vs depth-based distance and sigma, using the *_CAL takes as ground truth, and run evaluate.py with each." |
| 11b | (Optional) **ROS-bag check** — needs the team's fusion repo + bag: `git clone https://github.com/KaveeshwaraBandara/lidar-cam-fusion-lab ~/lidar-cam-fusion-lab`, bag from Kaveeshwara into its `data/go2_bags/` | Our distance vs LiDAR/RealSense on the team's `social_fusion` bag | "Run the pipeline on the bag's colour images and compare distance with LiDAR/RealSense." |
| 12 | **Paper figures/tables** | Vector PDFs 12.2 cm wide in `paper/figures/`; LaTeX tables in `paper/tables/` | "Make paper-ready figures and LaTeX tables from results/." |

## 4. Writing

1. Overleaf → New Project → **Upload Project** → zip of `paper/` (compiles as-is with llncs).
2. Red `\TODO{}` = to write; blue `\RES{}` = numbers from results. **Both must be gone before submitting.**
3. Check every `% VERIFY` entry in `paper/references.bib` on Google Scholar.
4. Anonymous: no names, no Rysera/WSO2, no website/GitHub link. The real author block is commented out in `main.tex`.
5. LNCS single column, ~12 pages including references. Don't change margins/fonts.
