# Full dataset evaluation (60 standing recordings)

Episodes: 1847 r / 2093 n; person-frames 38808; missing frames {'no person / cues / depth': 246}.

| Policy | False triggers | Recall | Recall at FT ≤ 2 % |
|---|---|---|---|
| Gesture only | 67.5% (1413/2093) [65–69] | 70.3% (1299/1847) [68–72] |  |
| Nearest person | 21.2% (444/2093) [20–23] | 20.4% (376/1847) [19–22] |  |
| Distance only | 65.1% (1362/2093) [63–67] | 68.2% (1260/1847) [66–70] |  |
| Facing only | 7.5% (158/2093) [6–9] | 65.1% (1202/1847) [63–67] | 20% |
| Gaze only | 8.9% (186/2093) [8–10] | 65.7% (1214/1847) [64–68] | 27% |
| Logistic regression | 6.8% (143/2093) [6–8] | 64.8% (1197/1847) [63–67] |  |
| Type-1 fuzzy | 12.5% (261/2093) [11–14] | 66.6% (1231/1847) [64–69] | 11% |
| Type-1 + gaze | 11.5% (241/2093) [10–13] | 67.4% (1245/1847) [65–70] | 13% |
| Type-1, mono distance | 10.8% (227/2093) [10–12] | 65.3% (1207/1847) [63–67] |  |
| IT2 fuzzy | 8.2% (171/2093) [7–9] | 64.2% (1185/1847) [62–66] | 20% |
| IT2 + gaze | 6.9% (145/2093) [6–8] | 63.7% (1176/1847) [61–66] | 20% |
| IT2, mono distance | 6.5% (135/2093) [5–8] | 57.7% (1065/1847) [55–60] |  |

Median LR coefficients: distance -0.29, approach -0.39, facing +3.23, distance_sigma -0.32, facing_sigma -0.03

## IT2 fuzzy vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1242 | 0 | 0 / 114 | 1.83e-33 | -64.2 to -54.5 | -8.2 to -4.5 |
| Nearest person | 151 / 424 | 1.03e-29 | 833 / 24 | 1.23e-210 | -18.7 to -7.6 | +35.7 to +51.9 |
| Distance only | 5 / 1196 | 0 | 35 / 110 | 3.51e-09 | -62.3 to -50.9 | -7.2 to +0.4 |
| Facing only | 26 / 13 | 0.213 | 13 / 30 | 0.0766 | -0.2 to +1.5 | -2.1 to +0.2 |
| Gaze only | 34 / 49 | 0.235 | 21 / 50 | 0.00537 | -2.3 to +0.8 | -3.0 to -0.2 |
| Logistic regression | 40 / 12 | 0.00102 | 17 / 29 | 0.235 | +0.5 to +2.4 | -1.7 to +0.3 |
| Type-1 fuzzy | 0 / 90 | 2.75e-26 | 0 / 46 | 3.69e-13 | -6.3 to -2.7 | -3.5 to -1.6 |
| Type-1 + gaze | 0 / 70 | 2.54e-20 | 0 / 60 | 2.43e-17 | -4.9 to -2.1 | -4.6 to -2.1 |
| Type-1, mono distance | 6 / 62 | 9.82e-12 | 25 / 47 | 0.0766 | -4.3 to -1.4 | -2.2 to -0.2 |
| IT2 + gaze | 29 / 3 | 2.3e-05 | 15 / 6 | 0.235 | +0.5 to +2.1 | -0.1 to +1.1 |
| IT2, mono distance | 41 / 5 | 4.41e-07 | 140 / 20 | 3.67e-22 | +0.7 to +2.8 | +4.5 to +8.5 |

## IT2 + gaze vs others (McNemar, Holm; bootstrap 95 % CI of the difference, points)

| vs | n: only ref / only other | p | r: only ref / only other | p | Δ FT | Δ recall |
|---|---|---|---|---|---|---|
| Gesture only | 0 / 1268 | 0 | 0 / 123 | 3.39e-36 | -65.1 to -56.1 | -8.8 to -4.9 |
| Nearest person | 130 / 429 | 5.33e-37 | 830 / 30 | 1.32e-202 | -19.5 to -9.2 | +35.3 to +51.3 |
| Distance only | 3 / 1220 | 0 | 35 / 119 | 7.25e-11 | -63.2 to -52.6 | -7.8 to -0.1 |
| Facing only | 8 / 21 | 0.0965 | 4 / 30 | 4.32e-05 | -1.5 to +0.2 | -2.5 to -0.5 |
| Gaze only | 13 / 54 | 4.47e-06 | 11 / 49 | 6.81e-06 | -3.3 to -0.8 | -3.3 to -1.0 |
| Logistic regression | 18 / 16 | 0.864 | 11 / 32 | 0.00957 | -0.5 to +0.8 | -2.1 to -0.3 |
| Type-1 fuzzy | 0 / 116 | 4.09e-34 | 2 / 57 | 7.37e-14 | -8.0 to -3.5 | -4.2 to -1.9 |
| Type-1 + gaze | 0 / 96 | 4.04e-28 | 0 / 69 | 4.74e-20 | -6.6 to -2.9 | -5.1 to -2.5 |
| Type-1, mono distance | 4 / 86 | 5.62e-20 | 26 / 57 | 0.00527 | -6.0 to -2.3 | -2.8 to -0.7 |
| IT2 fuzzy | 3 / 29 | 2.04e-05 | 6 / 15 | 0.235 | -2.1 to -0.5 | -1.1 to +0.1 |
| IT2, mono distance | 25 / 15 | 0.308 | 130 / 19 | 2.38e-20 | -0.4 to +1.3 | +4.2 to +7.8 |

## Splits (FT / recall %)

| Split | Bin | Type-1 | IT2 | Type-1+gaze | IT2+gaze | Facing only |
|---|---|---|---|---|---|---|
| facing σ | low | 5 / 68 | 5 / 68 | 5 / 69 | 5 / 68 | 5 / 68 |
| facing σ | high | 18 / 65 | 11 / 59 | 16 / 66 | 9 / 58 | 10 / 60 |
| distance σ | low | 5 / 69 | 4 / 67 | 5 / 69 | 4 / 66 | 4 / 67 |
| distance σ | high | 22 / 65 | 13 / 62 | 21 / 67 | 11 / 62 | 12 / 64 |
| gaze σ | low | 43 / 69 | 32 / 68 | 42 / 70 | 30 / 68 | 31 / 69 |
| gaze σ | high | 6 / 53 | 3 / 39 | 5 / 52 | 2 / 35 | 2 / 39 |
| distance | < 2 m | 2 / 54 | 1 / 51 | 1 / 54 | 1 / 49 | 1 / 52 |
| distance | 2–3 m | 22 / 67 | 15 / 65 | 21 / 68 | 13 / 65 | 14 / 66 |
| distance | ≥ 3 m | 19 / 80 | 9 / 76 | 16 / 78 | 7 / 76 | 12 / 73 |

## By scenario / label source / day (FT / recall %)

| Group | Gesture only | Facing only | Gaze only | Type-1 | IT2 | IT2+gaze |
|---|---|---|---|---|---|---|
| scenario S1 | nan / 68 | nan / 66 | nan / 67 | nan / 68 | nan / 68 | nan / 67 |
| scenario S2 | 66 / 67 | 7 / 61 | 10 / 61 | 13 / 63 | 9 / 61 | 8 / 60 |
| scenario S3 | 68 / 76 | 6 / 70 | 6 / 70 | 8 / 73 | 6 / 71 | 5 / 70 |
| scenario S4 | 69 / 79 | 5 / 68 | 7 / 70 | 11 / 73 | 5 / 68 | 4 / 68 |
| scenario S5 | 100 / 59 | 50 / 58 | 50 / 58 | 50 / 58 | 50 / 57 | 50 / 57 |
| scenario S6 | 66 / 72 | 20 / 66 | 19 / 68 | 30 / 65 | 18 / 59 | 14 / 59 |
| source labels.json | 67 / 72 | 9 / 65 | 11 / 67 | 16 / 68 | 10 / 65 | 8 / 64 |
| source labels_cmd.json | 68 / 69 | 6 / 65 | 7 / 65 | 9 / 65 | 6 / 64 | 5 / 63 |
| day D1 | 68 / 81 | 0 / 79 | 0 / 79 | 0 / 79 | 0 / 79 | 0 / 79 |
| day D2 | 63 / 73 | 7 / 57 | 9 / 62 | 20 / 65 | 9 / 55 | 6 / 54 |
| day D3 | 69 / 67 | 12 / 62 | 15 / 62 | 18 / 63 | 13 / 61 | 12 / 61 |
| day D4 | 68 / 71 | 6 / 67 | 6 / 67 | 8 / 68 | 6 / 66 | 5 / 66 |

## FOU scale k (FT / recall %)

k=0: 12 / 67 · k=0.5: 10 / 66 · k=1: 8 / 64 · k=2: 6 / 60

## Ablations (FT / recall %)

- no approach: Type-1 13 / 68, IT2 10 / 66
- facing only: Type-1 13 / 68, IT2 9 / 66
- distance only: Type-1 1 / 11, IT2 0 / 4

## Noise (mean FT / recall %, 20 seeds)

- facing s=0: type1 12.5 / 66.6, it2 8.2 / 64.2, it2_fixed_sigma 8.2 / 64.2, facing_only 7.5 / 65.1
- facing s=0.1: type1 11.0 / 66.0, it2 6.6 / 61.4, it2_fixed_sigma 7.6 / 63.0, facing_only 6.7 / 60.9
- facing s=0.2: type1 9.8 / 62.5, it2 3.9 / 42.3, it2_fixed_sigma 6.6 / 57.3, facing_only 5.3 / 49.4
- facing s=0.3: type1 8.8 / 55.9, it2 2.1 / 25.9, it2_fixed_sigma 5.9 / 50.2, facing_only 4.6 / 41.8
- distance s=0: type1 12.5 / 66.6, it2 8.2 / 64.2, it2_fixed_sigma 8.2 / 64.2, facing_only 7.5 / 65.1
- distance s=0.3: type1 12.4 / 66.7, it2 7.9 / 63.8, it2_fixed_sigma 8.1 / 64.1, facing_only 7.5 / 65.1
- distance s=0.6: type1 12.1 / 66.6, it2 7.4 / 62.8, it2_fixed_sigma 8.0 / 63.8, facing_only 7.5 / 65.1
- distance s=1.0: type1 11.9 / 66.5, it2 6.9 / 61.5, it2_fixed_sigma 7.8 / 63.4, facing_only 7.5 / 65.1

Episode-mean depth distance (m) p5/25/50/75/95: r: 1.88, 2.38, 2.60, 2.84, 2.93; n: 1.54, 1.85, 2.05, 2.41, 3.04
