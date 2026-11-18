# Run sheet: Experiment 1 (in-band interference) and Experiment 2 (acoustic-label control arm)

Written 29 October 2026 for Zafar Ahmad. Both experiments are pre-registered (`EXP1_PREREG.md`, SHA-256
c0f0269d...; `EXP2_PREREG.md`, SHA-256 f0b9fe2b...) and logged in `extension_study/prereg/CHANGES_AFTER_FREEZE.md`
and `extension_study/runs/ledger.csv`. Each CirCor scoring run happens once. Nothing is changed after a result is seen.
The notebooks refuse to classify in a dry or limited run, and they refuse a GPU other than an A100 for the full run
(arm E must reproduce Run 1 bit for bit).

Standing rules for every Colab run: Runtime > Change runtime type > **A100 GPU**, **High-RAM**. Leave every flag at
its built value (`DRY_RUN = False`, `LIMIT_RECORDINGS = 0`, `SMOKE = False`). Runtime > Run all. If the session
disconnects, reopen the notebook, set `RESUME_DIR` in the manual-override cell to the run folder it had created, and
run all again (every stage is cached). Do not edit a cell; if something fails, send me the error output and I will
rebuild the notebook from its builder (the builders are the source of truth; the `.ipynb` files are never hand-edited).

## Experiment 1: added in-band interference at test time (run this first)

**Done, 29 October 2026.** Run `runs/exp1/20261029_1331` (A100 40 GB, 2.8 h); gate PASSED; scored once with `exp1_analysis.py`; H1 rejected (T under Gaussian noise at 0 dB above E: +3.70 [+0.75, +6.71]). Analysis output and write-up: `extension_study/runs/exp1_analysis/` (`exp1_results.json`, `EXP1_RESULTS.md`; copy of the note in this folder). Logged in the ledger and the change log; in the manuscript as Methods, Results (Table 6), Discussion, and Supplement Section S-Noise. Still on Drive, not downloaded: `cache/`, `diagnostics.csv`, `cache_hashes.json`, `folds/` of that run (optional).

Notebook: `PhD Level Research/extension_study/notebooks/Experiment1_InBandNoise.ipynb` (built by `build_exp1.py`;
notebook source SHA-256 recorded in the ledger).

What it needs on Drive (all already there from Runs 1 and 2):

- the CirCor `training_data` folder and `training_data.csv` under `MyDrive/JBHI-Revision/` (found by the Drive walk,
  as in Run 1);
- the frozen T and W weights in `MyDrive/JBHI-Revision/Extension_Study/checkpoints/20261018_0908/`.

ESC-50: the notebook downloads the archive (about 600 MB) from GitHub itself and checks the 120 clips it uses by hash.
If the download fails in Colab, upload the unpacked `ESC-50-master` folder to Drive and set `MANUAL_ESC50_DIR` to it
in the manual-override cell; nothing else changes.

Steps:

1. Upload the notebook to Colab, A100 High-RAM, Run all. Expect 3 to 6 hours: 610 murmur-present recordings, five
   clean controls (C, E, S, T, W), six interference conditions (band-limited Gaussian, bursts, ESC-50 crying, breathing
   and coughing; each at 10 and 0 dB in-band SNR), segmentation-only arms S, T, W and both-stage arms E, C, S, T, W per
   condition, 53 arms in all, each through the 464 features, the five-seed MLP and the random forest on the embedded
   child-grouped folds.
2. The integrity gate (Stage 8) must print PASSED: the five clean controls reproduce Runs 1 and 2 exactly, the frozen
   weights match the freeze record, and arm B's construction equals the shared module. A FAILED gate means the run is
   not used; send me the output.
3. Download `MyDrive/JBHI-Revision/Extension_Study/runs/exp1/<timestamp>/` **without** the `cache/` folder (results.json,
   oof_predictions.npz, exp1_boundaries.json, exp1_noise_manifest.json, diagnostics.csv, run_manifest.json,
   run_flags.json) to `PhD Level Research/extension_study/runs/exp1/<timestamp>/`.
4. Locally, from `PhD Level Research/extension_study`:

       python notebooks/exp1_analysis.py "E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz" "runs/run2/20261018_1358/oof_predictions.npz" "runs/exp1/<timestamp>" "runs/exp1_analysis"

   This writes `exp1_results.json`: H1 (T_seg under band-limited Gaussian at 0 dB above E, one-sided 0.025), the
   retention of each arm's gain under each condition, the extractor's own loss (E_both against E), and the
   reproduction checks. I then log the run and the result in the ledger and write the results note.

## Experiment 2: the acoustic-label control arm T_ac (three steps; step 1 can run in parallel with Experiment 1)

**Steps 1 and 2 done, 30 October 2026.** Training run `20261029_1824_ac` (A100 80 GB; three seeds, about 19 min each; best validation F1 0.936 to 0.941; Gate A kappa 0.95 to 0.99 on four datasets and 0.69 to 0.72 on EPHNOGRAM, descriptive). Archived in `extension_study/runs/run3_freeze/20261029_1824_ac/` and logged. `Extension_Run3.ipynb` built from the freeze record (`extension_study/notebooks/Extension_Run3.ipynb`, source SHA-256 40212c2b...) and logged; a local label-free interface test ran stages 0 to 6 on 24 recordings. **Next: step 3 below.**

Step 1, training (paired data only, no CirCor):

1. Upload to Drive, into `MyDrive/JBHI-Revision/Extension_Study/corpus/labels_acoustic_v1/`, the two files from
   `PhD Level Research/derived/labels_acoustic_v1/`: `events_paired_v3.npz` (12 MB) and `laws.json`. The training pack
   `corpus/paired_v3/` is already on Drive from T's training.
2. Upload `extension_study/notebooks/Extension_Train_AC.ipynb` to Colab, A100 High-RAM, Run all. Expect about 1.5 hours
   (three seeds, about 30 minutes each). It writes `MyDrive/JBHI-Revision/Extension_Study/checkpoints/<stamp>_ac/`
   with three `.pt` files, `freeze.json` and `paired_test.json` (Gate A, descriptive, read beside T's
   `checkpoints/20261018_0908/paired_test.json`).
3. Download `freeze.json` and `paired_test.json` to `PhD Level Research/extension_study/runs/run3_freeze/<stamp>_ac/`
   and tell me the stamp.

Step 2, building the scoring notebook (local, me): `python notebooks/build_run3.py <stamp>_ac` writes
`Extension_Run3.ipynb` with the freeze record embedded and every hash checked; I log the freeze and the notebook.

Step 3, scoring on CirCor (one run):

1. Upload `Extension_Run3.ipynb` to Colab, A100 High-RAM, Run all. Expect about 1.5 hours, as Run 2. Arms: E
   (reproduction control), T_ac, T_ac_annot.
2. Stage 8 must print PASSED (arm E's features and predictions equal Run 1's; the frozen weights match the record).
3. Download `MyDrive/JBHI-Revision/Extension_Study/runs/run3/<timestamp>/` without `cache/` to
   `PhD Level Research/extension_study/runs/run3/<timestamp>/`.
4. Locally:

       python notebooks/exp2_analysis.py "E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz" "runs/run2/20261018_1358/oof_predictions.npz" "runs/run3/<timestamp>" "runs/run3/<timestamp>/analysis"

   This writes `exp2_results.json`: H1 (T minus T_ac, two-sided 95% interval; three readings fixed in the plan),
   T_ac against E, S and S_pub, its recovery fraction at matched coverage beside T's 0.63, the random forest,
   detection, and the label-gate figures (acoustic labels 99.68% S1 and 99.66% S2 inside the hand-corrected sound,
   against 99.98% and 99.88% for the ECG-derived labels).

## After Run 3 arrives (prepared and dry-tested on a stand-in, 30 October 2026)

**Done, 30 October 2026.** Run 3 `runs/run3/20261029_1955` (A100 80 GB, 43 min of stage time); gate PASSED; scored once: T minus T_ac -0.89 [-3.06, +1.31], reading 'no difference shown at this precision'; T_ac composite 50.8, T_ac minus E +7.1 [+4.3, +9.8], T_ac minus S +3.8 [+1.3, +6.5], recovery 0.85 [0.59, 1.14] (T 0.63), detection 80.4. Logged; `runs/exp2_analysis/EXP2_RESULTS.md`; manuscript, supplement (Section S-Control), tables, letter, and bundle updated; title reworded to 'Learnt cardiac-phase gating ...' as the plan commits (authors to confirm the wording).

From `PhD Level Research/extension_study`, with `<stamp>` the Run 3 folder name:

1. Copy the downloaded folder to `runs/run3/<stamp>/` (without `cache/`).
2. `python notebooks/exp2_analysis.py "E:/NUST MS Thesis/PhD Level Research/Results/A100 Results/extracted/runs/run1/20261016_1516/oof_predictions.npz" "runs/run2/20261018_1358/oof_predictions.npz" "runs/run3/<stamp>" "runs/exp2_analysis"` (one scoring; refuses a wrong plan hash or an arm E that does not reproduce Run 1).
3. `python notebooks/write_exp2_results.py runs/exp2_analysis/exp2_results.json runs/run3/<stamp> runs/exp2_analysis/EXP2_RESULTS.md --copy-to-next-study`
4. `python notebooks/log_exp2_run.py <stamp>` (ledger and change log, append-only; `--dry` prints the rows first).
5. Package: add `runs/run3/<stamp>` and `runs/exp2_analysis` to `00_admin/make_package.py` ITEMS, run `make_package.py`, then `python 00_admin/patch_manuscript_exp2.py --results 06_results/exp2_analysis/exp2_results.json --pkg . --date "<day month>"` (wording follows the reading fixed in EXP2_PREREG.md), `make_tables.py`, `make_supp_control.py`, `make_supp_tables.py`, `check_manuscript.py` (main text must stay under 5,000 words; the dry run gave 4,988 and an abstract of 247), the banned-term scan, `build_all.py`, `03_letters/assemble_letter.py` (update the counts sentence by hand), refresh the reviewer bundle (change log, ledger, EXP2_RESULTS.md, README), log the rebuilt files.
6. Title: if the reading is 'no difference' or 'T_ac above T', the plan says the title and abstract are reworded; the abstract is handled by the patch, the title is the authors' decision (appears in main.tex, the supplement title page, the TRIPOD checklist, the cover letter, and the response letter).

## What the manuscript does with the results

- Experiment 1 answers the in-band noise question the Introduction raises as rationale; today the Discussion says the
  noise question is untested. A positive H1 lets that sentence become a result; a negative one is reported as such.
- Experiment 2 answers whether "electrocardiogram-supervised" names a mechanism (T above T_ac) or a training recipe (no
  difference shown, or T_ac above T); the title and abstract are reworded accordingly, as the plan states.
