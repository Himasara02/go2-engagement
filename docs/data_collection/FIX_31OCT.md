# Fix 31 Oct: not every person gets a box

**Cause:** the code tracked at most **4 people**. Takes with 5 people always miss one, and that person's gestures never reach the labelling tool, so they can't be fixed during labelling.

**Fix:** the limit is now 6. Every take is re-detected from the saved video and depth, so **nothing has to be recorded again**. Only the labelling has to be redone.

## Steps (on the recording laptop)

1. **Stop labelling** for now.
2. Put the new files in place. The laptop's code was adapted for Windows, so **don't overwrite files**; let Claude Code apply the change. Open Claude Code in the project folder and paste:

   > Apply the 31 Oct fix from FIX_31OCT.md without undoing our Windows changes:
   > (1) in engage/engage/perception.py set the Perception default `max_people=6` and store `self.max_people = max_people`;
   > (2) in engage/run_live.py add `"max_people": getattr(perc, "max_people", None)` to meta.json;
   > (3) add the new file engage/reprocess.py from this update (adapt paths for Windows if needed);
   > (4) in engage/label_episodes.py draw the other people's boxes thicker (2 px, light blue) with their #id;
   > (5) in engage/check_dataset.py flag raw_format-2 takes whose meta max_people is missing or below 6.
   > Don't change anything else in the method.

3. Re-detect all recorded takes (about 1–3 minutes per take):
   ```
   python reprocess.py
   ```
   Old labels are kept in each take's `labels_old/` folder.
4. **Label every take again** (A and B), then run `python check_dataset.py` until it says no problems.
5. In the remaining sessions: **at most 5 people** in the picture, everyone **fully inside the frame** (not cut at the edges).

## Which gestures?

Not just any hand movement. Only these 5 count as commands:

| Gesture | How | Command |
|---|---|---|
| **Palm** | Open hand, fingers spread, palm facing the target | STOP |
| **Fist** | Closed fist facing the target | Forward |
| **Call** | Thumb and little finger out, like a phone | Backward |
| **One** | Index finger up | Turn left |
| **Two-up** | Index and middle finger up, held together | Turn right |

- **Commander and bystanders use the same 5 gestures.** Only the target differs: the dog for the commander, another person for a bystander.
- Hold each one **about 2 s** at chest height, with the hand clearly visible to the camera, then drop it.
- Other hand shapes (thumbs up, OK, peace sign, waving) aren't commands; the system ignores them. Don't use "finger on the lips" (it's read as Backward).
