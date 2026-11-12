# Runtime per frame (AMD Ryzen 7 5825U with Radeon Graphics, one process, 150 frames of D4_S3_r5, 3.0 people/frame)

| Stage | Median ms |
|---|---|
| decode | 1.2 |
| people_strips | 164.3 |
| people_whole_frame | 41.1 |
| hands_per_person | 27.5 |
| gaze_per_person | 9.3 |
| gate_per_person | 0.1 |

Per person (hands + gaze + gate): 36.9 ms. Frame total: 277 ms with strip people detection (3.6 fps), 154 ms with one whole-frame pass (6.5 fps).
