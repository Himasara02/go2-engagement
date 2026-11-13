# Full dataset evaluation (60 standing recordings)

Episodes: 945 r / 1472 n; person-frames 41984; missing frames {'no person / cues / depth': 125}.

| Policy | False triggers | Recall | Recall at FT ≤ 2 % |
|---|---|---|---|
| Gesture only | 65.1% (958/1472) [63–67] | 62.9% (594/945) [60–66] |  |
| Nearest person | 34.2% (504/1472) [32–37] | 20.2% (191/945) [18–23] |  |
| Distance only | 57.9% (853/1472) [55–60] | 29.3% (277/945) [26–32] |  |
| Facing only | 4.9% (72/1472) [4–6] | 58.9% (557/945) [56–62] | 33% |
| Gaze only | 6.6% (97/1472) [5–8] | 58.4% (552/945) [55–62] | 30% |
| Logistic regression | 5.0% (73/1472) [4–6] | 55.7% (526/945) [52–59] |  |
| Type-1 fuzzy | 6.2% (92/1472) [5–8] | 51.9% (490/945) [49–55] | 24% |
| Type-1 + gaze | 6.5% (96/1472) [5–8] | 55.4% (524/945) [52–59] | 25% |
| Type-1, mono distance | 6.0% (88/1472) [5–7] | 53.0% (501/945) [50–56] |  |
| IT2 fuzzy | 4.3% (64/1472) [3–6] | 48.4% (457/945) [45–52] | 23% |
| IT2 + gaze | 3.5% (52/1472) [3–5] | 49.1% (464/945) [46–52] | 25% |
| IT2, mono distance | 4.1% (60/1472) [3–5] | 45.7% (432/945) [43–49] |  |

Median LR coefficients: distance -0.21, approach -0.31, facing +3.69, distance_sigma -0.45, facing_sigma -0.05

## IT2 fuzzy vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 894 | 3.33e-268 | 0 / 137 | 2.07e-40 | -64.5 to -55.8 | -17.3 to -11.5 |
| Nearest person | 40 / 480 | 1.46e-95 | 295 / 29 | 2.45e-55 | -36.3 to -25.0 | +19.5 to +35.4 |
| Distance only | 22 / 811 | 9.1e-207 | 227 / 47 | 3.1e-28 | -57.4 to -48.9 | +8.9 to +27.2 |
| Facing only | 10 / 18 | 0.74 | 4 / 104 | 5.49e-25 | -1.3 to +0.0 | -13.7 to -6.8 |
| Gaze only | 13 / 46 | 0.000173 | 11 / 106 | 1.74e-19 | -3.9 to -0.7 | -13.6 to -6.0 |
| Logistic regression | 15 / 24 | 0.74 | 35 / 104 | 4.24e-08 | -1.3 to -0.2 | -10.3 to -3.7 |
| Type-1 fuzzy | 0 / 28 | 7.45e-08 | 0 / 33 | 3.03e-09 | -2.9 to -0.9 | -4.9 to -2.3 |
| Type-1 + gaze | 0 / 32 | 5.59e-09 | 0 / 67 | 1.9e-19 | -3.2 to -1.2 | -9.1 to -5.1 |
| Type-1, mono distance | 7 / 31 | 0.000813 | 37 / 81 | 0.000503 | -2.4 to -0.9 | -7.3 to -2.1 |
| IT2 + gaze | 13 / 1 | 0.011 | 12 / 19 | 0.74 | +0.3 to +1.3 | -2.4 to +0.5 |
| IT2, mono distance | 17 / 13 | 0.74 | 83 / 58 | 0.214 | -0.3 to +0.7 | +0.3 to +5.0 |

## IT2 + gaze vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 906 | 8.13e-272 | 0 / 130 | 2.64e-38 | -65.2 to -56.4 | -16.7 to -10.7 |
| Nearest person | 30 / 482 | 9.56e-105 | 304 / 31 | 3.31e-56 | -37.0 to -26.0 | +20.4 to +36.3 |
| Distance only | 16 / 817 | 1.66e-216 | 234 / 47 | 8.78e-30 | -58.0 to -49.6 | +9.7 to +27.9 |
| Facing only | 2 / 22 | 0.000251 | 4 / 97 | 5.37e-23 | -2.3 to -0.4 | -13.1 to -6.0 |
| Gaze only | 6 / 51 | 6.24e-09 | 7 / 95 | 1.18e-19 | -4.6 to -1.3 | -12.9 to -5.3 |
| Logistic regression | 6 / 27 | 0.00194 | 34 / 96 | 4.45e-07 | -2.1 to -0.8 | -9.2 to -3.3 |
| Type-1 fuzzy | 1 / 41 | 2.35e-10 | 0 / 26 | 2.98e-07 | -4.2 to -1.3 | -4.2 to -1.5 |
| Type-1 + gaze | 0 / 44 | 1.48e-12 | 0 / 60 | 2.43e-17 | -4.5 to -1.6 | -8.1 to -4.5 |
| Type-1, mono distance | 6 / 42 | 8.07e-07 | 35 / 72 | 0.00223 | -3.4 to -1.4 | -6.3 to -1.4 |
| IT2 fuzzy | 1 / 13 | 0.00732 | 19 / 12 | 0.401 | -1.3 to -0.3 | -0.5 to +2.4 |
| IT2, mono distance | 11 / 19 | 0.401 | 85 / 53 | 0.0243 | -1.2 to +0.1 | +1.1 to +6.0 |

## Splits (FT / recall %)

| Split | Bin | Type-1 | IT2 | Type-1+gaze | IT2+gaze | Facing only |
|---|---|---|---|---|---|---|
| facing σ | low | 2 / 44 | 2 / 40 | 3 / 47 | 1 / 43 | 3 / 53 |
| facing σ | high | 11 / 59 | 7 / 56 | 11 / 63 | 6 / 55 | 7 / 64 |
| distance σ | low | 5 / 51 | 4 / 46 | 6 / 54 | 3 / 47 | 4 / 56 |
| distance σ | high | 8 / 53 | 5 / 50 | 8 / 57 | 4 / 50 | 6 / 62 |
| gaze σ | low | 13 / 55 | 9 / 52 | 12 / 58 | 8 / 53 | 9 / 61 |
| gaze σ | high | 3 / 43 | 2 / 38 | 3 / 49 | 1 / 37 | 3 / 54 |
| distance | < 2 m | 4 / 29 | 1 / 14 | 4 / 29 | 1 / 0 | 1 / 14 |
| distance | 2–3 m | 4 / 53 | 3 / 49 | 4 / 55 | 3 / 50 | 3 / 53 |
| distance | ≥ 3 m | 14 / 52 | 10 / 48 | 14 / 56 | 8 / 49 | 13 / 62 |

## By scenario / label source / day (FT / recall %)

| Group | Gesture only | Facing only | Gaze only | Type-1 | IT2 | IT2+gaze |
|---|---|---|---|---|---|---|
| scenario W1 | 67 / 67 | 0 / 63 | 7 / 63 | 0 / 55 | 0 / 51 | 0 / 53 |
| scenario W2 | 65 / 62 | 4 / 55 | 5 / 56 | 5 / 45 | 3 / 41 | 2 / 39 |
| scenario W3 | 58 / 61 | 30 / 60 | 15 / 57 | 30 / 51 | 24 / 48 | 24 / 48 |
| scenario W4 | 79 / 53 | 4 / 47 | 5 / 49 | 11 / 51 | 3 / 50 | 1 / 51 |
| scenario W5 | 57 / 50 | 8 / 42 | 11 / 42 | 9 / 44 | 8 / 39 | 8 / 39 |
| scenario W6 | 71 / 83 | 9 / 80 | 9 / 81 | 7 / 73 | 7 / 67 | 7 / 73 |
| source labels.json | 65 / 62 | 8 / 56 | 10 / 57 | 9 / 56 | 7 / 52 | 6 / 54 |
| source labels_cmd.json | 65 / 63 | 4 / 60 | 6 / 59 | 5 / 50 | 3 / 47 | 3 / 47 |
| day D5 | 65 / 63 | 5 / 59 | 7 / 58 | 6 / 52 | 4 / 48 | 4 / 49 |

## FOU scale k (FT / recall %)

k=0: 6 / 52 · k=0.5: 5 / 50 · k=1: 4 / 48 · k=2: 3 / 37

## Ablations (FT / recall %)

- no approach: Type-1 9 / 60, IT2 7 / 60
- facing only: Type-1 8 / 60, IT2 6 / 60
- distance only: Type-1 1 / 2, IT2 0 / 1

## Noise (mean FT / recall %, 20 seeds)

- facing s=0: type1 6.2 / 51.9, it2 4.3 / 48.4, it2_fixed_sigma 4.3 / 48.4, facing_only 4.9 / 58.9
- facing s=0.1: type1 5.8 / 51.5, it2 3.6 / 45.2, it2_fixed_sigma 4.1 / 47.4, facing_only 4.2 / 54.4
- facing s=0.2: type1 5.7 / 48.6, it2 2.4 / 31.8, it2_fixed_sigma 3.9 / 42.6, facing_only 3.8 / 45.3
- facing s=0.3: type1 6.2 / 44.2, it2 1.7 / 21.9, it2_fixed_sigma 4.0 / 37.9, facing_only 3.8 / 40.2
- distance s=0: type1 6.2 / 51.9, it2 4.3 / 48.4, it2_fixed_sigma 4.3 / 48.4, facing_only 4.9 / 58.9
- distance s=0.3: type1 6.2 / 51.8, it2 4.3 / 48.0, it2_fixed_sigma 4.4 / 48.3, facing_only 4.9 / 58.9
- distance s=0.6: type1 6.2 / 51.6, it2 4.0 / 47.5, it2_fixed_sigma 4.3 / 48.2, facing_only 4.9 / 58.9
- distance s=1.0: type1 6.2 / 51.5, it2 3.6 / 46.9, it2_fixed_sigma 4.3 / 48.2, facing_only 4.9 / 58.9

Episode-mean depth distance (m) p5/25/50/75/95: r: 2.32, 2.89, 3.47, 4.21, 5.09; n: 1.55, 2.11, 2.54, 2.95, 4.27
