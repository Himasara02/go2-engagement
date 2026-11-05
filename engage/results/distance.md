# Distance sources (task 8)

## A. Depth distance on the calibration takes (1 person at 1/2/3/4 m marks)

| Take | Mark | Frames | Person detected | Bias (median d − mark) | SD | Median σ_d | |err| ≤ σ_d | |err| ≤ 2σ_d | Facing by phase (thirds) |
|---|---|---|---|---|---|---|---|---|---|
| D1_CAL | 1 m | 17 | 93% | +0.17 m | 0.06 m | 0.19 m | 47% | 71% | 0.35 / 0.33 / 0.23 |
| D1_CAL | 2 m | 251 | 75% | +0.10 m | 0.11 m | 0.10 m | 41% | 92% | 0.93 / 0.12 / 0.35 |
| D1_CAL | 3 m | 137 | 44% | +0.20 m | 0.09 m | 0.10 m | 17% | 50% | 0.02 / 0.35 / 0.18 |
| D1_CAL | 4 m | 3 | 43% | +0.02 m | 0.21 m | 0.52 m | 100% | 100% | 0.17 / 0.18 / 0.19 |
| D1_CAL_r2 | 1 m | 359 | 83% | +0.14 m | 0.08 m | 0.11 m | 42% | 99% | 0.85 / 0.85 / 0.34 |
| D1_CAL_r2 | 2 m | 186 | 100% | +0.13 m | 0.12 m | 0.10 m | 16% | 70% | 0.97 / 0.04 / 0.32 |
| D1_CAL_r2 | 3 m | 271 | 97% | +0.12 m | 0.13 m | 0.10 m | 27% | 67% | 0.98 / 0.03 / 0.34 |
| D1_CAL_r2 | 4 m | 28 | 14% | +0.33 m | 0.16 m | 0.30 m | 43% | 71% | 0.74 / 0.14 / 0.37 |
| D1_CAL_r3 | 1 m | 335 | 83% | +0.09 m | 0.06 m | 0.12 m | 94% | 100% | 0.92 / 0.77 / 0.34 |
| D1_CAL_r3 | 2 m | 232 | 100% | +0.22 m | 0.10 m | 0.11 m | 10% | 74% | 0.96 / 0.04 / 0.35 |
| D1_CAL_r3 | 3 m | 259 | 100% | +0.30 m | 0.12 m | 0.09 m | 3% | 12% | 0.97 / 0.06 / 0.22 |
| D1_CAL_r3 | 4 m | 57 | 80% | +0.30 m | 0.02 m | 0.16 m | 28% | 53% | 0.13 / 0.22 / 0.14 |

Pooled over the three takes (median of take values):

| Mark | Bias | SD | Median σ_d | |err| ≤ σ_d | |err − bias| ≤ σ_d |
|---|---|---|---|---|---|
| 1 m | +0.14 m | 0.06 m | 0.12 m | 47% | 95% |
| 2 m | +0.13 m | 0.11 m | 0.10 m | 16% | 79% |
| 3 m | +0.20 m | 0.12 m | 0.10 m | 17% | 85% |
| 4 m | +0.30 m | 0.16 m | 0.30 m | 43% | 100% |

Notes: the mark is the nominal floor position, so the bias also includes the offset between the camera and the mark line. D1_CAL (first take) lost the person for long stretches; D1_CAL_r2/r3 are clean. Phase order within a mark is taken from time (front, then the two non-frontal phases); it should be checked against the video.

## B. IT2 with different σ_d models (14 scenario takes, logged depth distance)

Monocular *distance values* need pose landmarks, which are not in the logs (re-run on raw.mp4 from the 31 Oct data). This table changes only σ_d.

| σ_d model | Median σ_d | Bystander false triggers | Recall |
|---|---|---|---|
| depth σ_d (logged) | 0.12 m | 4% (5/141) | 61% (62/101) |
| monocular σ_d = 0.15 + 0.08·d | 0.32 m | 4% (5/141) | 61% (62/101) |
| depth-hole σ_d = 0.45 | 0.45 m | 3% (4/141) | 61% (62/101) |
| Type-1 (no FOU) | 0.00 m | 9% (12/141) | 68% (69/101) |
