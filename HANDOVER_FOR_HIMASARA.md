# Himasara: repo, handover and next steps

## 1. Put the project in your GitHub repo (10 min, once)

On your laptop:
```bash
unzip ~/Downloads/go2-engagement.zip -d ~
cd ~/go2-engagement
git init -b main
git add -A
git commit -m "Go2 engagement project: code, paper, site, data-collection guide"
```
Create a **private** repo on github.com (e.g. `go2-engagement`, no README), then:
```bash
git remote add origin https://github.com/<your-username>/go2-engagement.git
git push -u origin main
```
(If you already have a repo you want this in, copy the folder's contents into it instead and commit.)

Add the boy: repo → **Settings → Collaborators → Add people** → his GitHub username. Then send him the repo link and **`HANDOVER_DATA_COLLECTION.pdf`**.

Also print `forms_to_print.pdf` (consent forms ×10, participant sheet, take log ×3) or send it to him to print.

## 2. What he does

He follows `HANDOVER_DATA_COLLECTION.md` from setup to handover:
- **Day 0:** setup, camera test, system test.
- **Days 1–3:** recording (about 12 takes per day + 1 calibration take) and labelling every evening, by two people.
- **Day 4:** `check_dataset.py` until it shows no problems, then the handover.

He works on a branch called `data-collection`. The data itself **never goes into git** (it's about 10–15 GB); `.gitignore` already blocks it.

Things you need to give him:
- the repo link,
- the Orbbec camera, its USB-3 cable and a way to mount it on the dog,
- the laptop (40 GB free),
- a 64 GB+ SSD,
- the name of a second person to be **annotator B**,
- a Google Drive folder link (backup).

Suggested dates: setup **Sun 25 Oct**, recording **Mon 26 – Wed 28**, handover **Thu 29 evening**.

## 3. What you get back

- **SSD (and Drive backup):** a folder `go2_dataset_<date>/` containing `takes/` (one folder per take: `raw.mp4`, `depth/`, `frames.csv`, `log.jsonl`, `meta.json`, `annot.mp4`, `labels.json`, `labels_B.json`), plus `dataset_report.md` and photos of the take log.
- **Paper:** consent forms and the participant sheet. Keep these privately; they're the only place names appear.
- **Git:** his `data-collection` branch, if Claude Code changed any code for him. On GitHub open **Pull requests → New → data-collection → main**, read the changes, and merge only if nothing in `engage/engage/fuzzy.py`, `methods.py` or `perception.py` changed the method.

Quick check when you receive it:
```bash
cd ~/go2-engagement/engage
cp -r /media/$USER/<SSD>/go2_dataset_<date>/takes ./takes
source .venv/bin/activate        # first time: bash setup.sh
python check_dataset.py          # should say "None. Ready to hand over."
```

## 4. Continue with Claude Code (31 Oct onwards)

```bash
cd ~/go2-engagement
git pull                         # after merging his branch, if any
claude
```
First message to Claude Code:

> Read CLAUDE.md and NEXT_PHASE.md. The dataset is now in engage/takes/ (Orbbec colour + depth, two annotators). Do the "after recording" tasks 5–12 in order: offline replay with depth, gaze module, dev/test split (D1 first 3 takes = dev), six policies, statistics with Cohen's kappa, the analyses and ablations, the distance-source comparison, then paper-ready figures and LaTeX tables into paper/. Show me the main table before writing anything into the paper.

Then fill the paper's `\RES{}` numbers and `\TODO{}` parts from the tables, following your writing slots (31 Oct: Method / Experiments / Results; 3 Nov: Discussion, Conclusion, Abstract; 6–7 Nov: review; **12 Nov: submit**).

## 5. If the data comes late or short

- Data arrives after 31 Oct: use 31 Oct for Claude Code's processing and 3 Nov for writing Results. Review moves to 6–8 Nov.
- Fewer than 300 labelled gestures: still submit, as an honest pilot study with confidence intervals. Don't record during exams.
