# LOOK diagnostic (13 Nov 2026)

LOOK scores on our keypoints (standing, every evaluated person-frame): r median 0.994, n median 0.992,
> 0.5 on 100 % (r) and 99.8 % (n) of frames; correlation with our facing score 0.07.
Probe on D4_S3_r5 (500 person-frames): as is 0.993; head-keypoint confidences set to 0 -> 0.054;
all confidences 0.3 -> 0.197; left/right swapped (back view geometry) -> 0.993.
=> the pretrained model decides almost only from head-keypoint confidence. LOOK was trained on OpenPifPaf
confidences, which drop when the face turns away; MediaPipe gives visibility ~1.0 for the head points
(mean 1.00 for nose, eyes, ears) whatever the head direction. So with our pose model LOOK cannot work;
a fair test needs OpenPifPaf keypoints (not done).
