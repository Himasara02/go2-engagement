# Extra baselines (standing, leave-one-recording-out)

Episodes r 1776, n 2185; LOOK keypoints missing on 0 person-frames.

| Policy | False triggers % | Recall % |
|---|---|---|
| look | 67.3 | 70.6 |
| look_fitted | 57.8 | 66.9 |
| seq | 6.8 | 66.2 |
| crisp | 14.5 | 69.4 |
| forest | 6.4 | 64.8 |
| it2_gaze | 7.7 | 65.4 |

McNemar vs IT2 + gaze (episodes where only one policy accepts):

- look: false triggers 1302 vs 0 (p=0); recall 91 vs 0 (p=8.1e-28)
- look_fitted: false triggers 1100 vs 6 (p=5.8e-318); recall 82 vs 55 (p=0.026)
- seq: false triggers 20 vs 40 (p=0.013); recall 36 vs 23 (p=0.12)
- crisp: false triggers 151 vs 3 (p=5.3e-41); recall 70 vs 0 (p=1.7e-21)
- forest: false triggers 12 vs 41 (p=8.2e-05); recall 19 vs 31 (p=0.12)
