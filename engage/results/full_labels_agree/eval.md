# Full dataset evaluation (60 standing recordings)

Episodes: 1659 r / 1980 n; person-frames 36795; missing frames {'no person / cues / depth': 220}.

| Policy | False triggers | Recall | Recall at FT ≤ 2 % |
|---|---|---|---|
| Gesture only | 67.4% (1334/1980) [65–69] | 70.5% (1170/1659) [68–73] |  |
| Nearest person | 21.9% (433/1980) [20–24] | 20.7% (344/1659) [19–23] |  |
| Distance only | 64.9% (1286/1980) [63–67] | 68.6% (1138/1659) [66–71] |  |
| Facing only | 4.8% (96/1980) [4–6] | 67.8% (1124/1659) [65–70] | 46% |
| Gaze only | 6.6% (130/1980) [6–8] | 68.1% (1129/1659) [66–70] | 50% |
| Logistic regression | 4.4% (88/1980) [4–5] | 68.1% (1129/1659) [66–70] |  |
| Type-1 fuzzy | 10.1% (199/1980) [9–11] | 69.0% (1145/1659) [67–71] | 13% |
| Type-1 + gaze | 9.1% (181/1980) [8–10] | 69.8% (1158/1659) [68–72] | 13% |
| Type-1, mono distance | 8.5% (169/1980) [7–10] | 67.9% (1127/1659) [66–70] |  |
| IT2 fuzzy | 6.0% (118/1980) [5–7] | 67.2% (1115/1659) [65–69] | 21% |
| IT2 + gaze | 4.7% (94/1980) [4–6] | 66.7% (1107/1659) [64–69] | 60% |
| IT2, mono distance | 4.5% (90/1980) [4–6] | 60.5% (1004/1659) [58–63] |  |

Median LR coefficients: distance -0.36, approach -0.49, facing +4.63, distance_sigma -0.55, facing_sigma -0.05

## IT2 fuzzy vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1216 | 0 | 0 / 55 | 8.33e-16 | -66.1 to -56.7 | -4.7 to -2.1 |
| Nearest person | 103 / 418 | 1.01e-44 | 780 / 9 | 3.88e-216 | -21.4 to -10.4 | +37.5 to +55.6 |
| Distance only | 3 / 1171 | 0 | 31 / 54 | 0.116 | -64.3 to -53.0 | -4.2 to +3.2 |
| Facing only | 31 / 9 | 0.00544 | 13 / 22 | 0.435 | +0.2 to +2.1 | -1.7 to +0.5 |
| Gaze only | 33 / 45 | 0.435 | 22 / 36 | 0.435 | -2.2 to +1.0 | -2.3 to +0.5 |
| Logistic regression | 38 / 8 | 8.43e-05 | 9 / 23 | 0.12 | +0.7 to +2.6 | -1.9 to +0.2 |
| Type-1 fuzzy | 0 / 81 | 1.49e-23 | 0 / 30 | 2.24e-08 | -6.0 to -2.5 | -2.7 to -1.0 |
| Type-1 + gaze | 0 / 63 | 3.47e-18 | 0 / 43 | 3.18e-12 | -4.7 to -2.0 | -3.8 to -1.5 |
| Type-1, mono distance | 4 / 55 | 2.21e-11 | 22 / 34 | 0.435 | -4.2 to -1.3 | -1.7 to +0.2 |
| IT2 + gaze | 27 / 3 | 8.43e-05 | 13 / 5 | 0.435 | +0.5 to +2.1 | -0.1 to +1.1 |
| IT2, mono distance | 32 / 4 | 2.14e-05 | 126 / 15 | 8.45e-22 | +0.5 to +2.4 | +4.6 to +8.8 |

## IT2 + gaze vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1240 | 0 | 0 / 63 | 3.04e-18 | -67.0 to -58.2 | -5.3 to -2.5 |
| Nearest person | 84 / 423 | 3.88e-54 | 778 / 15 | 1.62e-206 | -22.2 to -12.1 | +37.0 to +55.2 |
| Distance only | 3 / 1195 | 0 | 31 / 62 | 0.0137 | -65.2 to -54.6 | -4.8 to +2.8 |
| Facing only | 11 / 13 | 1 | 6 / 23 | 0.0162 | -0.8 to +0.6 | -2.1 to -0.2 |
| Gaze only | 13 / 49 | 5.3e-05 | 14 / 36 | 0.0162 | -3.2 to -0.7 | -2.6 to -0.2 |
| Logistic regression | 18 / 12 | 1 | 5 / 27 | 0.00102 | -0.3 to +1.0 | -2.4 to -0.4 |
| Type-1 fuzzy | 0 / 105 | 8.87e-31 | 2 / 40 | 4.93e-09 | -7.8 to -3.2 | -3.4 to -1.3 |
| Type-1 + gaze | 0 / 87 | 2.2e-25 | 0 / 51 | 1.15e-14 | -6.4 to -2.7 | -4.4 to -1.9 |
| Type-1, mono distance | 2 / 77 | 1.57e-19 | 23 / 43 | 0.0933 | -5.9 to -2.1 | -2.3 to -0.2 |
| IT2 fuzzy | 3 / 27 | 8.43e-05 | 5 / 13 | 0.385 | -2.1 to -0.5 | -1.1 to +0.1 |
| IT2, mono distance | 17 / 13 | 1 | 118 / 15 | 8.15e-20 | -0.7 to +1.0 | +4.2 to +8.2 |

## Splits (FT / recall %)

| Split | Bin | Type-1 | IT2 | Type-1+gaze | IT2+gaze | Facing only |
|---|---|---|---|---|---|---|
| facing σ | low | 2 / 69 | 2 / 69 | 2 / 70 | 2 / 69 | 2 / 69 |
| facing σ | high | 16 / 69 | 9 / 65 | 14 / 70 | 7 / 64 | 7 / 65 |
| distance σ | low | 3 / 72 | 2 / 70 | 3 / 72 | 2 / 69 | 2 / 70 |
| distance σ | high | 19 / 67 | 11 / 65 | 18 / 69 | 9 / 65 | 9 / 66 |
| gaze σ | low | 36 / 69 | 24 / 69 | 35 / 70 | 21 / 69 | 19 / 70 |
| gaze σ | high | 5 / 65 | 3 / 49 | 4 / 63 | 2 / 43 | 2 / 48 |
| distance | < 2 m | 1 / 63 | 0 / 61 | 1 / 64 | 0 / 58 | 0 / 60 |
| distance | 2–3 m | 18 / 69 | 12 / 67 | 17 / 70 | 9 / 67 | 9 / 68 |
| distance | ≥ 3 m | 17 / 85 | 8 / 85 | 15 / 85 | 7 / 85 | 11 / 82 |

## By scenario / label source / day (FT / recall %)

| Group | Gesture only | Facing only | Gaze only | Type-1 | IT2 | IT2+gaze |
|---|---|---|---|---|---|---|
| scenario S1 | nan / 68 | nan / 66 | nan / 67 | nan / 68 | nan / 68 | nan / 67 |
| scenario S2 | 66 / 67 | 4 / 66 | 7 / 66 | 11 / 66 | 7 / 66 | 5 / 65 |
| scenario S3 | 68 / 78 | 4 / 75 | 5 / 73 | 6 / 77 | 4 / 76 | 4 / 75 |
| scenario S4 | 68 / 81 | 2 / 77 | 4 / 79 | 8 / 81 | 3 / 77 | 2 / 77 |
| scenario S5 | 100 / 59 | 0 / 58 | 0 / 57 | 0 / 58 | 0 / 57 | 0 / 57 |
| scenario S6 | 68 / 72 | 17 / 67 | 17 / 68 | 28 / 65 | 17 / 59 | 12 / 60 |
| source labels.json | 67 / 73 | 6 / 69 | 9 / 71 | 14 / 72 | 8 / 69 | 6 / 69 |
| source labels_cmd.json | 68 / 69 | 3 / 66 | 4 / 65 | 6 / 66 | 4 / 65 | 3 / 65 |
| day D1 | 72 / 81 | 0 / 81 | 0 / 81 | 0 / 81 | 0 / 81 | 0 / 81 |
| day D2 | 62 / 73 | 4 / 65 | 6 / 70 | 17 / 72 | 8 / 63 | 4 / 63 |
| day D3 | 69 / 67 | 9 / 65 | 13 / 66 | 16 / 66 | 11 / 65 | 10 / 65 |
| day D4 | 67 / 71 | 3 / 69 | 4 / 68 | 6 / 69 | 3 / 68 | 2 / 68 |

## FOU scale k (FT / recall %)

k=0: 10 / 69 · k=0.5: 8 / 68 · k=1: 6 / 67 · k=2: 4 / 64

## Ablations (FT / recall %)

- no approach: Type-1 11 / 70, IT2 8 / 69
- facing only: Type-1 11 / 70, IT2 7 / 69
- distance only: Type-1 0 / 11, IT2 0 / 4

## Noise (mean FT / recall %, 20 seeds)

- facing s=0: type1 10.1 / 69.0, it2 6.0 / 67.2, it2_fixed_sigma 6.0 / 67.2, facing_only 4.8 / 67.8
- facing s=0.1: type1 8.7 / 68.6, it2 4.5 / 64.5, it2_fixed_sigma 5.3 / 66.2, facing_only 4.2 / 62.0
- facing s=0.2: type1 7.5 / 65.0, it2 2.4 / 44.4, it2_fixed_sigma 4.7 / 60.2, facing_only 3.3 / 49.1
- facing s=0.3: type1 6.8 / 58.2, it2 1.3 / 27.1, it2_fixed_sigma 4.3 / 52.5, facing_only 3.1 / 41.4
- distance s=0: type1 10.1 / 69.0, it2 6.0 / 67.2, it2_fixed_sigma 6.0 / 67.2, facing_only 4.8 / 67.8
- distance s=0.3: type1 10.0 / 69.0, it2 5.7 / 66.9, it2_fixed_sigma 5.9 / 67.1, facing_only 4.8 / 67.8
- distance s=0.6: type1 9.8 / 68.9, it2 5.2 / 66.0, it2_fixed_sigma 5.8 / 66.9, facing_only 4.8 / 67.8
- distance s=1.0: type1 9.5 / 68.8, it2 4.7 / 64.7, it2_fixed_sigma 5.7 / 66.5, facing_only 4.8 / 67.8

Episode-mean depth distance (m) p5/25/50/75/95: r: 1.92, 2.39, 2.61, 2.84, 2.93; n: 1.54, 1.82, 2.01, 2.36, 3.04
