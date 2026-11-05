# Start here: analysis + paper (5–12 Nov 2026)

## 1. Folder setup (once)

Put these side by side, e.g. in `D:\go2\`:

```
D:\go2\
  go2-engagement\                              <- this folder (code, paper, handover\)
  handover_himasara_2026-10-31_data.zip        <- Mihara's 31 Oct data (5.1 GB), don't unzip
```

Open a terminal in `D:\go2\go2-engagement` and run `claude`. Paste the prompt in section 2.

## 2. First prompt for Claude Code (paste as is)

> Read CLAUDE.md, START_HERE.md and NEXT_PHASE.md. I'm Himasara. Work through the task list in
> START_HERE.md section 3, in order. Before task 1, run `git init` and commit the code (no data).
> Commit after each task. After each task, show me the key numbers in a short table and wait for
> my OK before going on.

## 3. Task list (for Claude Code)

| # | Task | Done when |
|---|---|---|
| 0 | **Setup.** Python venv in `engage/` (Windows: `setup.ps1`, Linux: `setup.sh`). Rebuild the dataset: follow `handover/handover_2026-11-03_update/GUIDE_FOR_CLAUDE.md` with `--old ../handover_himasara_2026-10-31_data.zip --out ../dataset_2026-11-03`. | Prints `EXACT REPLICA OK` |
| 1 | **Reproduce.** `python evaluate.py ../dataset_2026-11-03/engage/takes_crop2/D*_S*` | Matches: gesture-only FT 60% (84/141) recall 74% (75/101); nearest 27% / 30%; Type-1 9% / 68%; IT2 4% (5/141) / 61% (62/101) |
| 2 | **Protocol.** The fuzzy parameters were frozen on a webcam dev take before data collection, so every take counts as test data for the fuzzy policies. Learned baselines use leave-one-take-out cross-validation. Write this into `results/protocol.md`. | File written |
| 3 | **Missed commands** (NEXT_PHASE 8b). Add each take's `_missed` count to the recall denominator for all policies; also report perception recall. `x` clips excluded and counted. | Table with and without missed |
| 4 | **Baselines.** Facing-only (threshold facing), distance-only, and logistic regression on the same per-frame features (distance, approach, facing, + their σ), leave-one-take-out, with the same 3-frame sustain. Gaze-only only if task 9 is done. | All policies in one table |
| 5 | **Statistics.** Wilson 95% CIs; McNemar IT2 vs Type-1 on paired episodes (both n and r); bootstrap CIs resampling takes. | p-values + CIs in results |
| 6 | **Analysis.** θ_E sweep: false-trigger vs recall curve for Type-1 and IT2 (IT2 should dominate). Results split by σ (low/high), by distance bin, by scenario S1–S6. Ablations: facing only / distance only / no approach; FOU scale k ∈ {0, 0.5, 1, 2} (k=0 ≡ Type-1). Noise injection on distance/facing. | Figures + tables in `results/` |
| 7 | **Perception comparison.** Full-frame hands (`takes_v2`) vs per-person crops (`takes_crop2`): command-gesture person-frames and episodes per take; commander gesture frames. Classifier confusion note (call→mute, rare one/two_up). | One small table |
| 8 | **Distance sources.** On the `D1_CAL*` takes (person at 1/2/3/4 m): depth distance vs monocular shoulder-width distance, error and σ. Then IT2 with monocular σ_d vs depth σ_d. | Table |
| 9 | (Optional, if time) **Gaze** (NEXT_PHASE task 6) on per-person crops, as a 4th input; rerun everything. If not done, remove every gaze claim from the paper. | — |
| 10 | **Figures/tables for the paper.** Vector PDF, 12.2 cm wide, into `paper/figures/`; LaTeX tables into `paper/tables/`. Pipeline figure too. | Files exist |
| 11 | **Fill the paper.** Replace every `\RES{}` and `\TODO{}` in `paper/` with results/text. Fix claims to match what was done: no walking results (no W takes), gaze only if task 9 done, hands on per-person crops (describe the crop method from the GUIDE), Orbbec camera model/resolution/fps from `meta.json`, participants/takes/minutes counts. One annotator unless `labels_B.json` exists (then Cohen's κ). | `grep -r "RES{\|TODO{" paper` finds nothing |
| 12 | **Final check.** Compile (if LaTeX is installed) or check syntax; ≤ 12 pages LNCS; anonymous (no names, Rysera, WSO2, university, links, and no acknowledgements); every `% VERIFY` reference checked on Google Scholar; zip `paper/` as `paper_overleaf.zip`. | Zip ready |

## 4. Overleaf

Claude Code can't log in to Overleaf, so pass the paper as a zip:

1. Claude Code edits `paper/` here and makes `paper_overleaf.zip` (task 12).
2. Overleaf → New Project → **Upload Project** → that zip. It compiles as-is (llncs is built in).
3. **Make your final edits in Overleaf.** From then on, Overleaf is the master copy.
4. If you want Claude Code to change something after that: Overleaf → Menu → Download → **Source**,
   unzip it over `paper/`, then ask Claude Code, then upload again. Never edit both at once.

(Overleaf Git/GitHub sync can skip the zips, but it needs a paid Overleaf plan.)

## 5. What only Himasara does

- Get a second person to label `takes_crop2` as annotator B (1–2 h, `label_episodes.py <take> --annotator B`), so the paper can report agreement. Optional but strong.
- Decide: record walking takes, or cut walking from the paper (default: cut).
- Read and fix the text in Overleaf; submit by **Thu 12 Nov** (deadline Sun 15 Nov).
