# ICAISC paper plan (simple version)

**Submit on Thursday 12 November.** The deadline is Sunday 15 November, the day after your last exam, so don't count on it.

## The idea in one line
In a crowd, the robot should only obey people who are actually looking at it and talking to it, and ignore people waving at each other.

## What we have to prove
1. Normal gesture control obeys anyone, so bystanders can accidentally control the robot.
2. Our system ignores bystanders but still obeys the real commander.
3. The Type-2 version works better than Type-1 when the robot can't see people clearly (walking, far away, someone blocking the view, people turning).

## What goes in the paper (full version)
- **Gaze**: is the person's *eyes* on the robot, from the robot's own camera (MediaPipe iris landmarks). Its uncertainty grows with distance, so IT2 trusts gaze up close and relies on head/body far away.
- **Head + body facing**, distance and approach speed (already built).
- **Real-depth check**: our camera distance compared against LiDAR/RealSense on the team's Go2 ROS bag, so the uncertainty numbers are measured, not guessed.
- **5 methods compared** on the same data: gesture-only, nearest person, Type-1, IT2, and a simple learned model (logistic regression).
- **Extra tests**: two independent labellers (agreement score), statistical test IT2 vs Type-1, a threshold sweep curve, a noise test, and with/without gaze.

**Key trick:** every recording saves the raw video. So gaze and any later improvement can be **re-run on the videos afterwards**. Recording does not have to wait for the code to be finished.

## Your exams
Economics 29 Oct · PS2 30 Oct · Robotics 2 Nov · Embedded 5 Nov · Renewable 9 Nov · PS3 11 Nov · PE2 14 Nov (all mornings)

**Rule:** all the robot work and results are done **before 29 Oct**. During exams you only write, in short slots, and **the day before every exam stays free**.

## Part 1: before exams (16–28 Oct) — data and results

| Days | What to do |
|---|---|
| **Fri 16 – Sun 18 Oct** | Connect the laptop to the dog's camera. Make the consent form. Book **8+ people** for 21–24 Oct. Open the Overleaf LNCS template. |
| **Mon 19 – Tue 20 Oct** | Claude Code: save raw video at **full camera resolution** (needed for gaze), add the gaze module, the second-labeller mode and the extra tests, and run the ROS-bag distance check. Record 2–3 practice videos, set the settings, then **freeze them**. Start Method and Related Work in Overleaf. |
| **Wed 21 – Sat 24 Oct** | **Recording days.** 3 sessions in 2 places, **10 people** if possible, about 40 short videos (dog standing and walking). Include close (1–2 m) takes so gaze has something to see. Bystanders keep gesturing to each other. Two people label each day's videos that evening. Goal: **300+ labelled gestures**. |
| **Sun 25 Oct** | Claude Code re-runs everything on the videos (with gaze) and makes all the tables and graphs. |
| **Mon 26 Oct** | Spare day: one extra recording session if there aren't enough gestures. Check the results make sense. |
| **Tue 27 – Wed 28 Oct** | Study for Economics and PS2. No paper work. |

## Part 2: during exams — writing only

| Day | What to do (about 3 hours) |
|---|---|
| **Sat 31 Oct** | Write the Experiments and Results sections. |
| **Tue 3 Nov** | Write the Introduction, Discussion, Conclusion and Abstract. Full draft done. |
| **Fri 6 – Sat 7 Nov** | Teammates and a lecturer read it. Fix their comments. |
| **Thu 12 Nov** | Remove names, logos, thanks and links. Final check. **Submit.** |
| Sun 15 Nov | Deadline, emergency backup only. |

Free (study) days: 28 Oct, 29 Oct, 1 Nov, 4 Nov, 8 Nov, 10 Nov, 13 Nov.

## We submit no matter what
If we end up with fewer gestures than planned, we still submit. We report the numbers honestly, with confidence intervals, and call it a pilot study on a real robot. Don't try to record during exams.

## Share the work
Teammates should own pieces so it doesn't all fall on you:

- one runs the recording sessions,
- two do the labelling,
- one writes Related Work,
- one checks the references and anonymising.

If they have the same exams, split the Part 2 writing slots between you.

## If something goes wrong
| Problem | What to do |
|---|---|
| Laptop can't connect to the dog | Tape a phone or webcam on the dog and record with that |
| Not enough bystander gestures | Do one more session and tell bystanders to gesture more |
| Type-2 is no better than Type-1 | Say so honestly; show the cases where it does help (far, walking, occluded) |
| Gaze too noisy | It still counts: report it, and IT2 automatically leans on head/body when gaze is unreliable |
| Not enough people | Ask Rysera students and staff (6 minimum) |

## Remember
- The videos are the most important part. Without them there's no paper.
- If the paper is accepted, someone must present it **in person in Poland in June 2027**. Ask WSO2 about funding early.
