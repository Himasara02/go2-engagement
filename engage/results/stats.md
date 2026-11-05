# Statistics (task 5)

Episodes: 141 bystander (n), 101 robot-directed (r), 14 takes. Bootstrap: 10000 resamples of takes, seed 0.

## Rates with confidence intervals

| Policy | False triggers | Wilson 95% | Bootstrap 95% (takes) | Recall | Wilson 95% | Bootstrap 95% (takes) |
|---|---|---|---|---|---|---|
| Gesture only | 60% (84/141) | 51–67% | 45–68% | 74% (75/101) | 65–82% | 66–80% |
| Nearest person | 27% (38/141) | 20–35% | 14–36% | 30% (30/101) | 22–39% | 5–54% |
| Facing only | 6% (8/141) | 3–11% | 2–12% | 65% (66/101) | 56–74% | 49–76% |
| Distance only | 56% (79/141) | 48–64% | 39–67% | 58% (59/101) | 49–68% | 35–75% |
| Logistic regression | 4% (5/141) | 2–8% | 0–7% | 62% (63/101) | 53–71% | 45–72% |
| Type-1 fuzzy | 9% (12/141) | 5–14% | 3–18% | 68% (69/101) | 59–77% | 57–75% |
| IT2 fuzzy | 4% (5/141) | 2–8% | 1–7% | 61% (62/101) | 52–70% | 43–72% |

## Paired comparisons (exact McNemar) and bootstrap differences (IT2 − other)

Primary test: IT2 vs Type-1. Holm correction over the 6 comparisons within each label.

| IT2 vs | Bystander eps: only IT2 / only other | p | p (Holm) | Robot eps: only IT2 / only other | p | p (Holm) | Δ false triggers (boot.) | Δ recall (boot.) |
|---|---|---|---|---|---|---|---|---|
| Gesture only | 0 / 79 | 3.31e-24 | 1.99e-23 | 0 / 13 | 0.000244 | 0.00122 | -64 to -42 pp | -29 to -4 pp |
| Nearest person | 4 / 37 | 1.03e-07 | 4.1e-07 | 33 / 1 | 4.07e-09 | 2.44e-08 | -33 to -10 pp | +8 to +56 pp |
| Facing only | 0 / 3 | 0.25 | 0.5 | 0 / 4 | 0.125 | 0.375 | -7 to +0 pp | -9 to -1 pp |
| Distance only | 1 / 75 | 2.04e-21 | 1.02e-20 | 12 / 9 | 0.664 | 1 | -63 to -35 pp | -22 to +29 pp |
| Logistic regression | 1 / 1 | 1 | 1 | 0 / 1 | 1 | 1 | -1 to +3 pp | -4 to +0 pp |
| Type-1 fuzzy | 0 / 7 | 0.0156 | 0.0469 | 0 / 7 | 0.0156 | 0.0625 | -14 to +0 pp | -19 to -1 pp |
