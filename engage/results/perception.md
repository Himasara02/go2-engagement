# Perception: full-frame hands vs per-person centred crops (task 7)

Same people, track IDs and frames in both; only the hand stage differs. Full-frame values identical to the takes_v2 logs for 13 takes (takes_v2 log not available: D2_S1_r1).

Command gesture = class in {call, fist, mute, one, palm, stop, stop_inverted, two_up} with confidence ≥ 0.4.

| Take | Command person-frames (full → crop) | Episodes (full → crop) | Commander command frames (full → crop) |
|---|---|---|---|
| D1_S1_r1 | 1 → 123 | 0 → 5 | 1 → 116 |
| D1_S1_r2 | 301 → 231 | 5 → 10 | 301 → 231 |
| D1_S1_r3 | 217 → 434 | 13 → 22 | 217 → 434 |
| D1_S2_r1 | 51 → 372 | 5 → 38 | 25 → 174 |
| D2_S1_r1 | 0 → 0 | 0 → 0 | – (none chosen) |
| D2_S2_r1 | 2 → 254 | 0 → 32 | 0 → 124 |
| D2_S3_r1 | 73 → 520 | 8 → 65 | 1 → 82 |
| D2_S3_r2 | 0 → 262 | 0 → 21 | – (none chosen) |
| D2_S4_r1 | 0 → 81 | 0 → 7 | 0 → 12 |
| D2_S4_r2 | 5 → 35 | 0 → 3 | 2 → 24 |
| D2_S5_r1 | 4 → 67 | 0 → 3 | 2 → 30 |
| D2_S5_r2 | 32 → 78 | 3 → 6 | 27 → 74 |
| D2_S6_r1 | 10 → 173 | 0 → 19 | 0 → 17 |
| D2_S6_r2 | 16 → 211 | 0 → 22 | 1 → 19 |
| **Total** | **712 → 2841** | **34 → 253** | **577 → 1337** |

## Gesture classes of labelled crop2 episodes

| Label | Classes |
|---|---|
| r | fist 31, one 29, call 19, palm 12, mute 10 |
| n | call 93, mute 27, fist 18, stop_inverted 1, one 1, palm 1 |
| x | call 5, fist 3, mute 2, stop_inverted 1 |

All person-frames with conf ≥ 0.4 (crop2), by class: dislike 6439, call 931, fist 821, mute 470, like 420, one 346, palm 258, three2 88, ok 41, peace_inverted 26, stop_inverted 13, rock 4, stop 2.

Note: there is no per-frame ground truth of the intended gesture class, so this is a class distribution, not a confusion matrix. Known issue (collector's handover): `call` is often read as `mute` (both map to backward), `one`/`two_up` are rare, relaxed hands mostly read as `dislike` (not a command). This affects which command would run, not whether a gesture is for the robot.
