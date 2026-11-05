# Fix 31 Oct, part 2: detect every person on every frame

**Mihara's diagnosis is right.** MediaPipe's pose model in VIDEO mode only looks for new people when it loses track. After locking on to the first person it just follows them, so the others never get a box. The 6-person limit wasn't the cause.

**Fix:** run detection from scratch on every frame (IMAGE mode) for poses and hands. Our own tracker already keeps each person's #id across frames. It's a little slower (expect ~8–10 fps); that's fine.

## Steps

1. Paste this into Claude Code (keeps the Windows changes):

   > Apply FIX_31OCT_v2 from this update without undoing our Windows changes:
   > (1) engage/engage/perception.py: add a `mode` argument to Perception (default from env PERCEPTION_MODE, else "image"); use `mp_vision.RunningMode.IMAGE` for both PoseLandmarker and HandLandmarker when mode is "image", and call `self.pose.detect(img)` / `self.hands.detect(img)` instead of `detect_for_video` (keep the VIDEO path when mode is "video"). Store `self.mode`.
   > (2) run_live.py: save `"perception_mode": perc.mode` in meta.json.
   > (3) reprocess.py: also reprocess takes whose meta has no `perception_mode == "image"`, and write perception_mode into meta.
   > (4) check_dataset.py: flag raw_format-2 takes whose perception_mode isn't "image".
   > (5) add count_people.py from this update.
   > Compare with the files in this update; don't change anything else.

2. **Check one take first (about 2 minutes):**
   ```
   python count_people.py takes/D1_S2_r1
   ```
   Expected: `video` mode is stuck at 1 person; `image` mode shows mostly 3 people (the number in view). Open `takes/D1_S2_r1/_people_check.mp4`: everyone should have a green box most of the time.

3. **If image mode looks right:** `python reprocess.py` (all takes), then label again (A and B), then `python check_dataset.py`.

4. **If image mode still misses people** (for example under 70% of frames show everyone), don't reprocess. Send Himasara the count_people output and the `_people_check.mp4`. Plan B is a stronger person detector (YOLO) with the pose model run on each person separately.

Recording can continue in the meantime: new takes use the fixed code, and the saved video can always be reprocessed.
