# Extra baselines (walking, fitted on standing)

Episodes r 945, n 1472; LOOK keypoints missing on 0 person-frames.

| Policy | False triggers % | Recall % |
|---|---|---|
| look | 64.9 | 62.8 |
| look_fitted | 32.3 | 30.9 |
| seq | 4.4 | 51.5 |
| crisp | 6.7 | 34.8 |
| forest | 1.8 | 22.3 |
| it2_gaze | 3.5 | 49.1 |

McNemar vs IT2 + gaze (episodes where only one policy accepts):

- look: false triggers 904 vs 0 (p=1.5e-272); recall 130 vs 1 (p=9.7e-38)
- look_fitted: false triggers 451 vs 28 (p=2.2e-99); recall 73 vs 245 (p=7.1e-23)
- seq: false triggers 25 vs 12 (p=0.047); recall 82 vs 59 (p=0.064)
- crisp: false triggers 60 vs 14 (p=6.2e-08); recall 32 vs 167 (p=3.1e-23)
- forest: false triggers 3 vs 29 (p=2.6e-06); recall 10 vs 263 (p=7.3e-65)
