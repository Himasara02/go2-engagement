# Evaluation protocol

## Parameters frozen before data collection

All fuzzy-system and decision parameters were fixed on one webcam development take
(`dev_data/dev1`: 143.2 s, 2311 frames, 1 person, labelled 16 r / 2 n / 9 x clips)
before any robot data was recorded:

| Parameter | Value | Note |
|---|---|---|
| θ_E (motion commands) | 0.60 | engagement lower bound needed to act |
| θ_E (STOP) | 0.45 | safety bias: STOP accepted more easily |
| θ_c (gesture confidence) | 0.45 | lowered from 0.55 after dev1 (the only tuning step) |
| Sustain | 3 frames | decision must persist before it counts |
| FOU scale k (distance, facing, approach) | 1.0, 1.0, 1.0 | IT2 FOU half-width = k·σ; Type-1 = same system with k = 0 |
| Rule base, membership functions | 9 rules, Hall proxemics | `engage/engage/fuzzy.py` |

`engage/engage/fuzzy.py` and `engage/engage/methods.py` are byte-identical to the version of
12 Oct 2026. Recording took place on 28 Oct (day 1) and 30 Oct (day 2). The same parameters are
stored in every take's `meta.json` (`params`).

## Consequence: every recorded take is test data for the fuzzy policies

No parameter of gesture-only, nearest-person, Type-1 or IT2 was chosen with any recorded take,
so all 14 labelled scenario takes are used as test data. There is no dev/test split of the
robot data for these policies. Nothing is tuned on the test takes; any change to `fuzzy.py` or
`methods.py` would require re-running every policy on every take.

## Learned baselines: leave-one-take-out

Baselines with fitted parameters (logistic regression; any threshold chosen from data) are
trained with leave-one-take-out cross-validation: for each of the 14 takes, fit on the other
13 and predict the held-out take. Only held-out predictions are scored. The same per-frame
features (distance, approach, facing and their σ) and the same 3-frame sustain are used as for
the fuzzy policies.

## Dataset

- Main set: `takes_crop2` (people and track IDs from the IMAGE-mode run `takes_v2`; gestures
  from a square crop centred on each person's upper body). Calibration takes (`*_CAL`) and test
  takes (`test_*`) are excluded.
- 14 takes, scenarios S1–S6, all stationary (no walking takes), 17.4 min in total,
  1–4 participants per take (4 participants overall), recorded with an Orbbec RGB-D camera on
  the robot (colour 1280×720 @ 30 fps).
- Labels: annotator A, every clip: 101 r, 141 n, 11 x; 15 robot-directed commands never became
  a clip (`_missed`). One take (D2_S1_r1) has no clips.

## Scoring

- Episode = one tracked person holding one command gesture for ≥ 3 frames (gaps ≤ 4 frames
  merged), as produced by `label_episodes.episodes()`.
- An episode counts as accepted by a policy if that person is in the policy's sustained set in
  any frame of the episode.
- Bystander false-trigger rate = accepted `n` episodes / all `n` episodes.
- Command recall = accepted `r` episodes / all `r` episodes. `x` episodes (bad detections) are
  excluded and their number is reported. A second recall adds the `_missed` commands to the
  denominator (task 3).
- Policies are recomputed offline from the logged features (`evaluate.py`), so every policy
  sees exactly the same per-person features in every frame.
- Uncertainty: Wilson 95% intervals for rates; McNemar test for paired IT2 vs Type-1 decisions;
  bootstrap over takes for take-level variation (task 5).

## Limitations stated in the paper

- Thresholds were set on a single webcam dev take, not on robot data.
- Small study: 14 takes, 4 participants, one lab, stationary robot only.
- One annotator for the main labels (unless a second pass, `labels_B.json`, is added).
