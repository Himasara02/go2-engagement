# Full dataset evaluation (60 standing recordings)

Episodes: 1776 r / 2185 n; person-frames 39433; missing frames {'no person / cues / depth': 266}.

| Policy | False triggers | Recall | Recall at FT ≤ 2 % |
|---|---|---|---|
| Gesture only | 67.6% (1476/2185) [66–69] | 70.6% (1253/1776) [68–73] |  |
| Nearest person | 21.7% (474/2185) [20–23] | 20.0% (356/1776) [18–22] |  |
| Distance only | 65.2% (1425/2185) [63–67] | 68.6% (1219/1776) [66–71] |  |
| Facing only | 8.0% (175/2185) [7–9] | 66.6% (1182/1776) [64–69] | 21% |
| Gaze only | 9.6% (210/2185) [8–11] | 65.4% (1162/1776) [63–68] | 28% |
| Logistic regression | 7.8% (171/2185) [7–9] | 67.1% (1192/1776) [65–69] |  |
| Type-1 fuzzy | 13.3% (291/2185) [12–15] | 68.2% (1211/1776) [66–70] | 11% |
| Type-1 + gaze | 12.5% (273/2185) [11–14] | 68.8% (1222/1776) [67–71] | 13% |
| Type-1, mono distance | 11.6% (254/2185) [10–13] | 66.9% (1189/1776) [65–69] |  |
| IT2 fuzzy | 8.8% (193/2185) [8–10] | 66.0% (1172/1776) [64–68] | 20% |
| IT2 + gaze | 7.7% (168/2185) [7–9] | 65.4% (1162/1776) [63–68] | 20% |
| IT2, mono distance | 7.1% (156/2185) [6–8] | 59.3% (1053/1776) [57–62] |  |

Median LR coefficients: distance -0.21, approach -0.31, facing +3.68, distance_sigma -0.45, facing_sigma -0.05

## IT2 fuzzy vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1283 | 0 | 0 / 81 | 1.41e-23 | -63.5 to -53.7 | -6.4 to -3.0 |
| Nearest person | 158 / 439 | 2.83e-30 | 831 / 15 | 4.77e-222 | -18.6 to -6.8 | +37.0 to +54.9 |
| Distance only | 4 / 1236 | 0 | 32 / 79 | 9.45e-05 | -62.0 to -50.0 | -5.7 to +1.8 |
| Facing only | 35 / 17 | 0.105 | 14 / 24 | 0.322 | -0.1 to +1.9 | -1.7 to +0.5 |
| Gaze only | 41 / 58 | 0.322 | 47 / 37 | 0.326 | -2.4 to +0.8 | -1.6 to +3.1 |
| Logistic regression | 38 / 16 | 0.0269 | 6 / 26 | 0.00428 | +0.3 to +1.9 | -2.1 to -0.3 |
| Type-1 fuzzy | 0 / 98 | 1.14e-28 | 0 / 39 | 4.37e-11 | -6.3 to -2.9 | -3.4 to -1.2 |
| Type-1 + gaze | 0 / 80 | 2.48e-23 | 0 / 50 | 2.49e-14 | -5.2 to -2.4 | -4.1 to -1.7 |
| Type-1, mono distance | 7 / 68 | 1.52e-12 | 24 / 41 | 0.207 | -4.3 to -1.6 | -2.1 to +0.1 |
| IT2 + gaze | 29 / 4 | 9.84e-05 | 15 / 5 | 0.207 | +0.5 to +2.0 | +0.0 to +1.2 |
| IT2, mono distance | 46 / 9 | 4.77e-06 | 135 / 16 | 1.94e-23 | +0.7 to +2.7 | +4.7 to +8.7 |

## IT2 + gaze vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1308 | 0 | 0 / 91 | 1.29e-26 | -64.3 to -55.2 | -7.0 to -3.5 |
| Nearest person | 138 / 444 | 3.75e-37 | 827 / 21 | 1.05e-212 | -19.4 to -8.4 | +36.5 to +54.4 |
| Distance only | 4 / 1261 | 0 | 31 / 88 | 1.84e-06 | -62.8 to -51.6 | -6.3 to +1.3 |
| Facing only | 13 / 20 | 0.889 | 6 / 26 | 0.00375 | -1.2 to +0.5 | -2.2 to -0.2 |
| Gaze only | 19 / 61 | 2.46e-05 | 35 / 35 | 1 | -3.3 to -0.7 | -1.9 to +2.2 |
| Logistic regression | 21 / 24 | 1 | 4 / 34 | 6.04e-06 | -0.9 to +0.7 | -2.6 to -0.8 |
| Type-1 fuzzy | 0 / 123 | 3.39e-36 | 2 / 51 | 3.82e-12 | -7.9 to -3.7 | -4.1 to -1.7 |
| Type-1 + gaze | 0 / 105 | 8.38e-31 | 0 / 60 | 2.26e-17 | -6.8 to -3.2 | -4.8 to -2.2 |
| Type-1, mono distance | 5 / 91 | 2.28e-20 | 25 / 52 | 0.0168 | -5.8 to -2.3 | -2.8 to -0.4 |
| IT2 fuzzy | 4 / 29 | 8.74e-05 | 5 / 15 | 0.207 | -2.0 to -0.5 | -1.2 to +0.0 |
| IT2, mono distance | 29 / 17 | 0.415 | 126 / 17 | 1.42e-20 | -0.1 to +1.2 | +4.2 to +8.1 |

## Splits (FT / recall %)

| Split | Bin | Type-1 | IT2 | Type-1+gaze | IT2+gaze | Facing only |
|---|---|---|---|---|---|---|
| facing σ | low | 6 / 69 | 6 / 69 | 6 / 70 | 5 / 69 | 5 / 70 |
| facing σ | high | 19 / 67 | 11 / 62 | 17 / 67 | 9 / 60 | 10 / 62 |
| distance σ | low | 6 / 71 | 5 / 69 | 5 / 71 | 4 / 69 | 4 / 69 |
| distance σ | high | 23 / 66 | 14 / 64 | 22 / 67 | 12 / 63 | 13 / 65 |
| gaze σ | low | 43 / 70 | 32 / 69 | 42 / 70 | 30 / 69 | 30 / 70 |
| gaze σ | high | 6 / 58 | 3 / 43 | 6 / 56 | 2 / 39 | 3 / 43 |
| distance | < 2 m | 2 / 61 | 1 / 60 | 2 / 62 | 1 / 57 | 1 / 59 |
| distance | 2–3 m | 23 / 68 | 17 / 66 | 22 / 69 | 14 / 66 | 14 / 67 |
| distance | ≥ 3 m | 19 / 80 | 9 / 78 | 16 / 78 | 7 / 76 | 11 / 76 |

## By scenario / label source / day (FT / recall %)

| Group | Gesture only | Facing only | Gaze only | Type-1 | IT2 | IT2+gaze |
|---|---|---|---|---|---|---|
| scenario S1 | nan / 68 | nan / 66 | nan / 66 | nan / 68 | nan / 68 | nan / 67 |
| scenario S2 | 67 / 67 | 4 / 66 | 8 / 66 | 11 / 66 | 7 / 66 | 5 / 65 |
| scenario S3 | 67 / 78 | 4 / 75 | 5 / 73 | 6 / 77 | 4 / 76 | 4 / 75 |
| scenario S4 | 68 / 81 | 5 / 71 | 7 / 72 | 11 / 77 | 6 / 71 | 5 / 71 |
| scenario S5 | 100 / 59 | 12 / 58 | 12 / 57 | 50 / 58 | 25 / 58 | 25 / 57 |
| scenario S6 | 69 / 70 | 31 / 63 | 31 / 58 | 39 / 62 | 29 / 56 | 26 / 56 |
| source labels.json | 67 / 72 | 8 / 67 | 10 / 68 | 15 / 71 | 9 / 67 | 8 / 67 |
| source labels_cmd.json | 68 / 69 | 9 / 66 | 9 / 63 | 11 / 66 | 8 / 65 | 8 / 64 |
| day D1 | 74 / 80 | 13 / 80 | 13 / 80 | 13 / 80 | 13 / 80 | 13 / 80 |
| day D2 | 63 / 73 | 5 / 61 | 7 / 66 | 18 / 70 | 8 / 58 | 5 / 58 |
| day D3 | 69 / 67 | 10 / 65 | 13 / 65 | 17 / 66 | 11 / 65 | 10 / 65 |
| day D4 | 68 / 71 | 8 / 68 | 8 / 65 | 11 / 69 | 8 / 67 | 7 / 66 |

## FOU scale k (FT / recall %)

k=0: 13 / 68 · k=0.5: 11 / 67 · k=1: 9 / 66 · k=2: 6 / 62

## Ablations (FT / recall %)

- no approach: Type-1 15 / 69, IT2 11 / 68
- facing only: Type-1 14 / 69, IT2 10 / 68
- distance only: Type-1 1 / 11, IT2 1 / 4

## Noise (mean FT / recall %, 20 seeds)

- facing s=0: type1 13.3 / 68.2, it2 8.8 / 66.0, it2_fixed_sigma 8.8 / 66.0, facing_only 8.0 / 66.6
- facing s=0.1: type1 11.9 / 67.6, it2 7.2 / 63.3, it2_fixed_sigma 8.1 / 64.9, facing_only 7.0 / 60.6
- facing s=0.2: type1 10.6 / 63.9, it2 4.4 / 43.6, it2_fixed_sigma 7.3 / 58.9, facing_only 5.7 / 48.0
- facing s=0.3: type1 9.7 / 57.2, it2 2.5 / 26.3, it2_fixed_sigma 6.6 / 51.4, facing_only 5.1 / 40.5
- distance s=0: type1 13.3 / 68.2, it2 8.8 / 66.0, it2_fixed_sigma 8.8 / 66.0, facing_only 8.0 / 66.6
- distance s=0.3: type1 13.3 / 68.1, it2 8.5 / 65.7, it2_fixed_sigma 8.8 / 65.9, facing_only 8.0 / 66.6
- distance s=0.6: type1 13.0 / 68.0, it2 8.0 / 64.8, it2_fixed_sigma 8.6 / 65.7, facing_only 8.0 / 66.6
- distance s=1.0: type1 12.8 / 67.9, it2 7.3 / 63.5, it2_fixed_sigma 8.5 / 65.3, facing_only 8.0 / 66.6

Episode-mean depth distance (m) p5/25/50/75/95: r: 1.91, 2.39, 2.61, 2.84, 2.93; n: 1.54, 1.85, 2.05, 2.43, 3.04
