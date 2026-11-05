# Go2 engagement-gated gesture commands (ICAISC 2027 paper)

Owner: Himasara Kulasinghe (UoM final-year EE, Rysera). Team: Kaveeshwara Bandara, Rivikula Fernando,
Agra Navarathna, Himasara Kulasinghe, Lasan Perera. Rysera Innovations × WSO2 × University of Moratuwa.
Paper: "Whose Gesture Is It? Uncertainty-Aware Interval Type-2 Fuzzy Engagement Gating for Gesture
Commands to a Quadruped Robot in Crowds" — ICAISC 2027 (Springer LNAI, double-blind, ~12 pp LNCS),
submit by **12 Nov 2026** (deadline 15 Nov). Owner prefers short, plain answers.

## Current phase (from 5 Nov 2026): analysis + writing. Read `START_HERE.md` first.
Data collection is finished (collector handover 3 Nov, in `handover/`). The user is Himasara.
Rules for this phase:
- **Main dataset: `../dataset_2026-11-03/engage/takes_crop2/`** (rebuilt per `handover/.../GUIDE_FOR_CLAUDE.md`).
  People/IDs/features come from the IMAGE-mode run (takes_v2); gestures come from a square crop centred
  on each person's upper body (`engage/find_gestures_centered.py`). Labels: `labels.json` = annotator A
  (every clip, with `_missed`); annotator B may be added later as `labels_B.json`.
- `takes/`, `takes_v2/`, `takes_crop/` are older/alternative runs: use only for the "full-frame vs
  per-person-crop" perception comparison, not for the main results.
- Evaluation is **offline** on the logged features (`evaluate.py` recomputes every policy with the
  frozen parameters θ_E 0.60/0.45, θ_c 0.45, sustain 3). Never tune anything on test takes.
- Never edit labels, logs or the rebuilt dataset. New outputs go to `engage/results/`.
- Every number written into `paper/` must come from a file in `engage/results/`. Don't invent numbers.
- Double-blind paper: no names, no Rysera/WSO2/University, no links.
- No walking takes exist yet (W1–W3). Don't report walking results unless walking takes are added.
- Ask before changing `fuzzy.py` / `methods.py`; any change means re-running every policy on every take.
- The user prefers short, plain answers. After each task, show the key numbers and stop for a check.

## The problem
In a crowd, gesture-controlled robots obey any hand. Per person we want to: (1) recognise and attribute
each gesture, even simultaneous ones; (2) obey only gestures meant for the robot; (3) ignore the same
person when they gesture to someone else.

## Method (`engage/`)
- Sensor: **Orbbec RGB-D on the dog** (`engage/engage/orbbec.py`, pyorbbecsdk2 v2 API, AlignFilter D2C).
  The dog (Go2 AIR/PRO, no EDU SDK) is driven by a human with the remote during data collection.
  WebRTC control (`go2_link.py`, go2_webrtc_connect) exists but is untested.
- Perception (`perception.py`): MediaPipe PoseLandmarker (≤6 people, IMAGE mode) + HandLandmarker (dataset: on per-person crops); hand→person by
  wrist distance < 0.7 shoulder widths; 18-class keypoint MLP (HaGRID-style labels), 7-frame vote.
  Distance = median depth on shoulders/hips (σ_d from spread + depth holes + temporal std), fallback
  monocular shoulder width. Facing = 0.65 head + 0.35 body yaw with σ_f from visibility + jitter.
  Gaze: not yet implemented (task 6).
- Layer 1 (`fuzzy.py`): engagement FLS, 9 rules, Hall proxemics; IT2 uncertain-mean sets with
  FOU half-width = k·σ per person per frame, exact Karnik–Mendel; σ=0 ⇒ IT2 ≡ Type-1.
- Layer 2: act ⇔ E_low ≥ θ_E(cmd) ∧ conf ≥ θ_c. θ_E 0.60 motion / 0.45 STOP, θ_c 0.45, 3-frame sustain.
  Frozen (set on dev take dev1). Arbitration: STOP wins, else most engaged.
- Policies run in shadow mode on identical features: gesture_only, nearest, type1, it2
  (gaze_only + logistic regression to add, task 8).

## Data format (takes/<take>/, raw_format 2)
raw.mp4 (native, unmirrored colour) · depth/*.npz (uint16 mm, half-res, `depth`+`idx`, 200-frame chunks)
· frames.csv · log.jsonl (processing view: ≤960 px wide, mirrored; bboxes in this view) · annot.mp4
(overlay, not for labelling) · meta.json · labels.json (A) · labels_B.json (B).
Episode = one tracked person holding one command gesture ≥3 frames (label_episodes.episodes()).
Labels: r = for robot, n = not for robot, x = bad detection. Take names D<day>_<S1..S6|W1..W3>_r<rep>,
calibration takes D<day>_CAL (person at 1/2/3/4 m: front/side/back 5 s each).
`dev_data/dev1/` = old-format webcam dev take (raw_format 1, mirrored+resized raw).

## Files
- `engage/run_live.py` record/replay · `label_episodes.py` · `evaluate.py` · `check_dataset.py` ·
  `check_orbbec.py` · `setup.sh`
- `paper/` LNCS LaTeX (anonymous) · `site/` project website builder (Vercel project go2-engagement)
- Related team repos (not included here): `KaveeshwaraBandara/lidar-cam-fusion-lab` (LiDAR–camera fusion + Kalman tracker with per-person pos_std, real Go2 EDU bag `social_fusion`; bag itself is not in git, get it from Kaveeshwara) and `KaveeshwaraBandara/go2-social-nav` (ROS 2/Gazebo sim, HuNavSim, gesture teleop). Clone next to this repo if needed (`~/lidar-cam-fusion-lab`).
- `NEXT_PHASE.md` timeline + research tasks · `docs/` older plans

## Honest novelty
Muñoz-Salinas et al. 2005 (Type-1 fuzzy interest from distance/angle/face) ≈ our Layer 1; Yun et al. 2012
fuzzy engagement for attention selection; quadruped gesture HRI (HyperPalm 2022, arXiv 2408.17066)
assumes one user; arXiv 2607.09734 review on the single-individual assumption. Claim = integration:
per-person command gating in a crowd + IT2 FOU driven by live measured uncertainty (depth, facing, gaze)
+ real-quadruped comparison against gesture-only / nearest / gaze-only / learned / Type-1.
Never claim fuzzy engagement itself as new.
