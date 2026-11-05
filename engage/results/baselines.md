# Baselines (task 4)

Same gesture gate (command gesture, conf ≥ 0.45) and 3-frame sustain for all. Learned thresholds/models: leave-one-take-out, Youden J on the 13 training takes. Brackets: Wilson 95% CI.

| Policy | Bystander false triggers | Recall (clips) | Recall (incl. missed) | Youden J |
|---|---|---|---|---|
| Gesture only | 60% (84/141) [51–67] | 74% (75/101) [65–82] | 65% (75/116) | 0.15 |
| Nearest person | 27% (38/141) [20–35] | 30% (30/101) [22–39] | 26% (30/116) | 0.03 |
| Facing only | 6% (8/141) [3–11] | 65% (66/101) [56–74] | 57% (66/116) | 0.60 |
| Distance only | 56% (79/141) [48–64] | 58% (59/101) [49–68] | 51% (59/116) | 0.02 |
| Logistic regression | 4% (5/141) [2–8] | 62% (63/101) [53–71] | 54% (63/116) | 0.59 |
| Type-1 fuzzy | 9% (12/141) [5–14] | 68% (69/101) [59–77] | 59% (69/116) | 0.60 |
| IT2 fuzzy | 4% (5/141) [2–8] | 61% (62/101) [52–70] | 53% (62/116) | 0.58 |

- Facing only: threshold per fold median 0.60 (range 0.60 to 0.78)
- Distance only: threshold per fold median -2.90 (range -3.00 to -2.70)
- Logistic regression: threshold per fold median 0.65 (range 0.47 to 0.80)
- Logistic regression median coefficients (standardised): distance -0.62, approach -0.43, facing +3.50, distance_sigma -0.69, facing_sigma +0.35
- Distance threshold is shown as negative distance (accept if distance ≤ −τ).
