# R1_Gating_Campaign.ipynb: how to run it

This folder holds five files:

| file | role |
|---|---|
| `build_gating_campaign.py` | builder for the gating notebook. Reads the verbatim code blocks out of the analysis source and the diagnostic notebooks at build time, assembles the cells, checks that every code cell compiles, scans for em dashes, embeds the provenance hashes and a SHA-256 of the notebook's own cell sources, and writes the notebook. Run `python build_gating_campaign.py` to regenerate it; the `.ipynb` is never edited by hand. |
| `R1_Gating_Campaign.ipynb` | the Colab notebook produced by the builder. 48 cells (30 code, 18 markdown), runs top to bottom. |
| `build_cnn_baseline.py` | builder for the CNN baseline notebook (second half of this file). |
| `R1_CNN_Baseline.ipynb` | the CNN baseline notebook. |
| `NOTEBOOK_README.md` | this file. |

The review documents `FIX_LIST.md` and `REVIEWER_SNIPPETS.md` sit beside them.

The builder resolves its sources relative to its own location: `../../../Synthetic-Gating-Experiment/`
(the `.py` export plus the four diagnostic notebooks) and
`../../../../Milestone-3/New Pipeline (Critic)/ABLATION_GATING_VALIDATION.ipynb` (the published
equal-time control). Override with the environment variables `GATING_SRC_DIR` and
`GATING_ABLATION_NB` if the folders move.

## Runtime to choose

Colab Pro, **High-RAM** runtime, GPU. A100 or L4 preferred; a T4 works with the generator batch
automatically dropped to 16. The 8 kHz audio (about 2.2 GB) is held in RAM only for the chunks
whose generator cache is missing; after that the resident set is the 4 kHz audio (1.1 GB), the
float16 synthetic ECG (1.1 GB) and the feature matrices (small). The worker pool forks processes
on Linux (so on Colab) and uses every core; on a platform without the fork start method (Windows)
the same code runs serially through an executor with the same interface, and the manifest records
`pool.kind = sequential`. `torch.use_deterministic_algorithms(True, warn_only=True)` is on together
with the cuDNN and TF32 settings.

## Drive layout the notebook expects

Nothing is hard-coded; Stage 0 walks `/content/drive/MyDrive` once (depth 9) for both targets and
prints what it found. What it is looking for:

* **Dataset.** The `training_data` folder with the largest number of paired `.wav` + `.tsv` files.
  The complete copy (3163 pairs) is at
  `MyDrive/JBHI-Revision/October Thesis Dataset/CirCor DigiScope/the-circor-digiscope-phonocardiogram-dataset-1.0.3/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data`.
  A 3155-pair copy under `MyDrive/October Thesis Dataset/CirCor DigiScope/...` and a stale 14-file
  folder also exist; both lose on the pair count and are skipped. `training_data.csv` must be the
  sibling of the chosen folder. A searched run refuses to start if the best copy has fewer than
  2500 pairs; that guard applies to the search only, a `MANUAL_TRAINING_DATA` path bypasses it (the
  pair count is still printed, so check it).
* **Generator.** Every `trail4_best.pt` on Drive is hashed (SHA-256) and printed. Known copies:
  `MyDrive/October Thesis Dataset/trail4_checkpoints/trail4_best.pt` and
  `MyDrive/October Thesis Dataset/Michigan-External Testing/models/trail4_best.pt`. The
  `trail4_checkpoints` copy is used when present; a loud warning is printed if the copies differ.
* **Output.** `MyDrive/JBHI-Revision/R1_Gating_Campaign/<YYYYMMDD_HHMM>/` (suffix `_dry` for a dry
  run, `_limitN` for `LIMIT_RECORDINGS = N`) with `folds/`, `cache/`, `cache/cv/`, `figures/`, and
  the files listed below.

### Manual override

The cell headed **MANUAL OVERRIDE** (right after the environment cell) has three variables:
`MANUAL_TRAINING_DATA`, `MANUAL_CHECKPOINT`, `MANUAL_OUTPUT_ROOT`. Leave them `None` to search;
set any of them to a path string to pin that item and skip its search. Resolved paths are printed
and asserted either way. A pinned `MANUAL_CHECKPOINT` must exist and be a torch archive (a zip
file); otherwise Stage 0 stops with a message naming the path.

## Flags (first code cell)

| flag | default | meaning |
|---|---|---|
| `DRY_RUN` | `False` | `True` runs a class-aware subset: about 60 murmur-present recordings drawn round-robin over the nine grading / timing / shape (task, class) buckets, one recording per patient while distinct patients last, plus about 60 murmur-absent recordings from further patients; `N_SPLITS = 5`, `N_BOOT = 100`, own `_dry` run folder, per-class counts of the subset printed in Stage 1. About 60 min on a 16-core machine without a GPU (the generator alone takes about 45 min there, because torch runs single-threaded under the thread pin that protects the feature workers), a few minutes on an L4 after the Drive search. The numbers are for exercising the code only: the three-class tasks sit on about 6 to 25 recordings per class, their folds are reduced to the smallest class when that is below 5, and their CIs are wide or undefined. A task whose subset has fewer than `N_SPLITS` patients or a class with one member is skipped and listed under `sample.skipped_tasks`. |
| `LIMIT_RECORDINGS` | `0` | 0 = all recordings; N > 0 = an evenly spaced subset of N in its own `_limitN` folder. Any subset disables the validation gate. |
| `N_BOOT` | `2000` | patient-level bootstrap replicates. |
| `RESUME_DIR` | `None` | path of an existing run folder. Its `run_flags.json` (dry_run, limit_recordings, seed, n_splits, chunk, n_min_cycles, dry_run_subset) must equal the current flags, otherwise Stage 0 refuses; a `_dry` folder cannot be resumed with `DRY_RUN = False` and the reverse. Every stage then reuses that folder's caches and recomputes only what is missing or stale. |
| `FORCE_FIGURES` | `False` | draw figures although the validation gate failed. Diagnostic use only. |
| `SEED`, `N_SPLITS`, `GATE_TOL`, `CHUNK`, `N_MIN_CYCLES` | 42, 10, 3.0, 400, 2 | seed, folds, gate tolerance in macro-F1 points, recordings per cache file, minimum cycles per arm for a recording to stay in the kept set. |
| `DRY_RUN_SUBSET` | `dict(present=60, absent=60)` | size of the dry-run subset. |

## What each stage does and writes

| stage | what | writes (under the run folder) | full run, A100 / L4 |
|---|---|---|---|
| 0 | one Drive walk for the dataset and every checkpoint copy, SHA-256 of each copy, run folder, `run_flags.json` check | run folder, `run_flags.json` | 1-3 min for the walk plus 5-10 s per checkpoint copy hashed |
| 1 | index WAV + TSV + label (sorted by file name: the reference run's order on every platform), exclusion list, dry-run subset | `index.json` | < 1 min |
| 2 | decode audio, resample once to 8 kHz, cut the analysed portion, native 4 kHz array | `cache/audio_NNN.npz` (8 chunks of 400, about 3.3 GB) | 4-8 min (Drive I/O) |
| 3 | batched generator inference, per-window polarity on the real samples, float16 cache tied to the checkpoint's SHA-256 | `cache/ecg_NNN.npz` (about 1.1 GB) | 6-12 min on A100, 10-20 min on L4 |
| 4 | six boundary arms in the worker pool, rate-fidelity diagnostics, kept set | `cache/cycles_arms_ab.pkl`, `cache/cycles_arms_cdee2.pkl`, `cache/offsets_ms.pkl`, `diagnostics.csv`, `arm_failures.json` | 2-5 min |
| 5 | 464 features per arm, worker pool, resumable part file and atomic `.npz`, both tied to the cycles hash | `cache/features_<arm>.npz` (6 files), `cache/feat_part_arms.npz` while running | about 25-40 min on 12 Colab vCPUs |
| 6 | folds per task | `folds/<task>.npz` | seconds |
| 7 | MLP + RF per arm and task, metrics, bootstrap (vectorised, float64), paired tests | `cache/cv/<arm>__<task>.npz`, `results.json`, `oof_predictions.npz` | 20-35 min |
| 8 | validation gate: sample check, then arms C and E refitted with the reference recipe (8 full-batch fits per fold) | `cache/cv/<arm>__gate__<task>.npz`, `results.json` (`validation_gate`); on a sample mismatch also `cache/features_<arm>__refset.npz`, `folds/<task>__refset.npz`, `cache/cv/<arm>__gate_refset__<task>.npz` | 5-10 min; not run in a dry run |
| 9 | ablations: window length, timing budget, redundancy, scaling, subgroups | `cache/features_window_*.npz`, `cache/features_jitter_*.npz`, `cache/cv/*`, `cache/redundancy_corr_pcg_seg.npy`, `results.json` (`ablations`), `oof_predictions.npz` (ablation predictions added) | 60-100 min (the jitter features alone take about as long as Stage 5) |
| 10 | figures F1-F8 from the written outputs, each passing a layout check before it is saved; saved to the run folder, not shown inline; skipped (with a banner) when the gate failed | `figures/*.pdf`, `figures/*.png` | 1-2 min |
| 11 | manifest | `run_manifest.json` | seconds (cache hashes are taken at write time; only files written by an earlier notebook version are hashed here) |

Expect about 2.5 to 4.5 hours end to end for the full 3006-recording run; the CPU stages (features,
jitter features) dominate, so the GPU class matters little. A crashed session loses at most one chunk of work: set `RESUME_DIR` to the run folder and run
the notebook again from the top. Two things are recomputed on every resume because they are not
cached: the bootstraps (only the CV predictions are cached; about 10-15 min for every arm and task
at `N_BOOT = 2000`) and the diagnostics table (seconds). Stage seconds are accumulated across
sessions in `stage_seconds.json` and reported both per session and cumulatively in the manifest.

### Output files

* `results.json`: everything numeric, with a `schema` describing each section. Rates are in
  percent, AUROC / AUPRC in [0, 1], every pooled metric carries a 95 % patient-bootstrap CI over
  the valid resamples (`n_boot_valid`; `[null, null]` when fewer than 10), paired differences carry
  the CI, P(diff > 0) over the finite draws and the fold-wise paired t-test. The per-arm metrics are
  under `arms.<task>.<arm>.<clf>` and the paired differences under `paired.<task>.<X-Y>.<clf>` for
  the pairs A-E, B-E, C-E, D-E, B-A, C-A, C-B, C-D, A-D, A-E2, B-E2, C-E2 and E2-E. C-D (annotated
  phase boundaries against fixed RR fractions on the same annotated cycle starts) is the comparison
  R3-M4 asks for; A-D (synthetic against annotated anchors under the same fixed fractions) isolates
  anchor timing. `run.tasks` lists the
  tasks that ran, `sample.kept_names` the kept set, `sample.skipped_tasks` the tasks a reduced run
  skipped. Offset signs: `median_offset_ms` = arm cycle start minus annotated S1 onset (negative =
  arm precedes S1); `arm_mean_syn_ac_offset_ms` = acoustic S1 onset minus synthetic R-peak.
  `ablations.redundancy.per_arm` holds, for every arm, the statistics set of the supplement's
  feature-redundancy section (pairs above 0.8 / 0.9 / 0.95 / 0.99 with features involved, clusters,
  largest cluster and independent groups; coincident pairs at |r| >= 0.9999; constant features;
  median and mean |r|; PCA components for 90 / 95 / 99 % of the variance), so the notebook's numbers
  and the supplement's can be compared directly.
* `oof_predictions.npz`: per task `y`, `names`, `patient_ids`, `fold_id`; per task, arm and
  classifier the pooled out-of-fold `pred` and `proba`; the same for the ablations under the tags
  `window_04s`, `window_08s`, `window_16s`, `window_full` (detection, MLP and RF), `jitter_010` to
  `jitter_160`, `pcg_seg_reduced` and `pcg_seg_fullscaler` (every task, MLP).
* `diagnostics.csv`: one row per recording and arm (rate, RR-CV, jump rate, insertion and deletion
  rate at 150 ms, median signed offset, over-detection ratio, detector path, join de-duplications,
  close pairs dropped after realignment, fusion statistics including `arm_n_rescued`, annotation
  gaps for arm D, `arm_annotated_coverage_fraction` and `arm_annotated_span_fraction` for arms C
  and E2).
* `folds/<task>.npz`: `names`, `patient_ids`, `y`, `fold_id`, `n_splits`, `train_NN` / `test_NN`
  index arrays into that task's labelled set, and the kept-set names. The CNN notebook must read
  these. `folds/<task>__refset.npz` appears only when the gate evaluated the reference-rule set.
* `arm_failures.json` (worker exceptions per recording and arm group), `run_flags.json`,
  `stage_seconds.json`, `cache_hashes.json` (git-style blob hash of every cache file, taken when it
  was written).
* `figures/`: `F1_forest_macro_f1`, `F2_paired_vs_equal_time`, `F3_timing_budget`,
  `F4_offset_histograms`, `F5_subgroups`, `F6_window_length`, `F7_confusion_matrices`,
  `F8_redundancy_heatmap`, each as PDF (vector) and PNG (600 dpi). Before a figure is saved a layout
  check draws it and stops the cell, naming the texts, if any text lies outside the canvas or two
  texts overlap, so a clipped or collided figure is never written. F6 and F8 are single-column
  (88 mm) figures with the same 8 / 9 / 10 pt fonts as the double-column ones; they only look
  larger when both are scaled to the same screen width.
* `run_manifest.json`: versions, GPU, seeds, run flags, dataset and checkpoint hashes, worker-pool
  kind, `resumed_from` and the session list, per-stage seconds (this session and cumulative), the
  cache-file hashes, the configuration hash (`cfg_hash`), the arm-constant hash (`arm_cfg_hash`),
  the cycles hash of every feature cache, the notebook source hash at build time and (in Colab) of
  the live notebook, `arm_failures`, `cache_rebuilds`, the gate result, the verbatim-block
  provenance, and `equal_time_usage_by_results_entry` naming which equal-time definition each
  results entry and figure used.

### Re-running Stage 10 on a finished folder

The figure cells read only `results.json`, `cache/offsets_ms.pkl` and
`cache/redundancy_corr_pcg_seg.npy`, and take `VALIDATION_OK` from `results.json`. Set
`RESUME_DIR`, run the flag cell, the environment cell, the override cell and Stage 0, then the
three figure cells (cells 41 to 43).

## The two equal-time controls

* **Arm E, `equal_time`** is the mechanism study's control (`equal_time_cycles`, verbatim from the
  analysis source): fixed 75 bpm pseudo-cycles cut into four equal quarters, no information from the
  recording. This is the arm the validation gate compares against the reference values.
* **Arm E2, `equal_time_matched`** reproduces the published ablation's control exactly
  (`get_nongated_cycles`, verbatim from `ABLATION_GATING_VALIDATION.ipynb`): the analysed portion is
  cut into N equal-duration pseudo-cycles where **N is the number of annotated cycles of arm C
  inside the analysed portion** (the portion-restricted count, and the analysed length, not the
  whole-recording values), each cut into four equal 25 % phases, with seconds converted to samples
  by `int(t * sr)` as that pipeline did. It inherits the annotated **count**, not the annotated
  rate: wherever annotation covers only part of the portion the pseudo-cycles run slow (in the local
  dry run the median implied rate was about 56 bpm against 99 bpm annotated). This is the published
  definition and is left as it is; Stage 4 asserts per recording that the E2 count equals the arm C
  count and records `annotated_coverage_fraction` beside it. E2 appears in every metrics table, in
  the paired differences (A-E2, B-E2, C-E2, E2-E) and in the forest plot.

## Validation gate

Two checks, both required. **Sample:** per task, n and the number of patients must equal the
reference run (detection 3006 / 874, grading 612 / 178, timing 608 / 177, shape 605 / 176); on a
mismatch the recordings dropped per arm are printed. **Numbers:** arms C and E are refitted with the
**reference recipe** (`fit_mlp_reference`: the reference study's own model class `ReferenceMLP`
with PyTorch default initialisation, as in cell 30 of the mechanism notebook, not the V5 class with
its Kaiming step; focal loss gamma 2 with class weights, one full-batch step per epoch, 300 epochs,
validation every 5 epochs, patience 5 checks, constant seed, inner `GroupShuffleSplit` 0.15; TF32
on during these fits only, as in the reference run, and restored afterwards) and compared with
C: 79.2 / 58.3 / 49.0 / 46.6 and E: 69.6 / 51.0 / 39.5 / 39.0 (detection / grading / timing /
shape); the primary recipe's value is printed beside each cell for orientation.

**The rule.** A cell is within when its deviation is at most `GATE_TOL` = 3.0 points, or when the
reference value lies inside the 95 % patient-bootstrap interval of the gate value. The second clause
is the design brief's "mismatch larger than bootstrap noise": with the correct model class the
reference recipe still moves by 4 to 5 points from one seed to the next on single cells, so a fixed
3-point band alone would fail a faithful reproduction about as often as not. Each cell's value,
deviation, interval and verdict are printed and stored (`ci`, `within_tolerance`,
`reference_inside_ci`, `within`). The gate passes when the sample matches and every cell is
within. Otherwise `VALIDATION_OK = False`: the figure cells print the failure and are skipped
(unless `FORCE_FIGURES` is set), and the provenance and manifest cells still run, so the manifest
records the failure.

If the kept set differs because arms A or B dropped recordings (arms C, D, E and E2 keep exactly
the 3006 reference recordings), the two gate arms are also evaluated on the reference-rule set (at
least two whole-recording TSV cycles and at least 1 s of audio) with folds drawn on that set and the
same rule; when that set matches the reference sample and every cell is within, the gate passes
with the headline "PASSED ON THE REFERENCE-RULE SET, NOT ON THE KEPT SET", names the arms that
dropped recordings, and records `decided_on = "reference-rule set"`. Every number of such a run
must be reported with the kept-set n, not with 3006. Under `DRY_RUN` or `LIMIT_RECORDINGS` the cell
prints "not applicable" and runs no reference fits.

## Design notes worth knowing before reading results

* The analysed portion rule (whole 8 s windows plus a trailing window of at least 2 s) trims up to
  2 s from the end of some recordings compared with the reference run, which used every sample.
  Arm C annotations past the end of the analysed portion are dropped; the count is in
  `diagnostics.csv` (`arm_n_beyond_portion`).
* The 4 kHz array is the native 4 kHz samples of the analysed portion (every CirCor file is 4 kHz),
  max-abs normalised over that portion; on an untrimmed recording it is sample-identical to the
  reference loader. A file at another rate would be resampled once from the 8 kHz portion.
* Arm A computes the 5 to 30 Hz envelope per generator window and concatenates the envelopes, so
  the step between independently z-scored, independently polarity-flipped windows never enters
  the filter; the prominence ladder and the 300 ms distance are then applied once, globally. After
  realignment to the raw maximum, two peaks closer than 300 ms lose the smaller one and the count is
  recorded (`arm_n_close_pairs_dropped`). The verbatim `_detect_sos` stays in the notebook for
  provenance and is not called.
* Arm B's 250 ms acoustic-peak spacing thins the candidate list that drives insertions only. A
  second, unspaced list of envelope maxima at 50 % of the p95 level or more (`rescue_height`)
  rescues a synthetic candidate whose S1 was suppressed by a louder S2 (`arm_n_rescued`). In the
  reviewer's perfect-anchor test this raised the support fraction from 0.82 to 0.98 and lowered
  false fallbacks from 11.7 % to 1.0 %, while wrong anchors stay at 0.39. Inside the systolic S2
  search window the same prominence is used with no spacing rule, because S2 sits 150 to 330 ms
  after S1 at paediatric heart rates. The 50 % crossing places the S1 onset about 17 ms and the S2
  onset 18 to 22 ms after the annotation; S2 refinement is mildly murmur-dependent (23 % vs 10 % of
  cycles more than 40 ms late). When fewer than half the candidates are supported the supported
  acoustic onsets are kept and only the removals and insertions are skipped (`fusion_fallback`).
* Arm D applies the fixed fractions inside each annotated cycle's own extent and never spans an
  interior annotation gap; the gaps are counted (`arm_n_annotation_gaps`).
* The window-length ablation renormalises each truncated excerpt (max-abs) before extracting the
  boundary-free features, as the loader normalises a whole recording.
* The bootstrap is vectorised in float64 (one recording-weight matrix per task) and draws the same
  patient resamples as the verbatim `boot_ci`; resamples whose truth holds one class are masked as
  the oracle skips them, and `n_boot_valid` is reported. Stage 7 asserts agreement with `boot_ci`
  and `boot_diff_ci` to 1e-6 on detection and on one three-class task, and asserts the vectorised
  weighted AUROC / AUPRC equal to sklearn on the point estimate and three replicates before any CI
  is reported. Balanced accuracy is asserted against sklearn as macro-F1 is.
* The primary MLP recipe is the article's V5 configuration with two stated departures from the V5
  loop: inverse-frequency class weights in the loss instead of a `WeightedRandomSampler`, and early
  stopping on inner macro-F1 instead of accuracy; gradient norms are clipped at 1.0. The gate uses
  the reference recipe described above. Per-class counts of every inner validation split are in
  `fit_info`.
* The redundancy screen (|r| > 0.9) is label-free and fitted once on all kept recordings; clusters
  are the connected components of the pair graph, as in the supplement.
* Subgroups use heart-rate tertiles from arm C and, as a second stratifier, from arm A
  (`syn_fixed`), plus the CirCor age band; a stratum needs at least 10 patients for an estimate.
* Every feature matrix, CV result and fold file stores the recording names it was built on, and the
  feature and CV caches also store content hashes (cycles hash; feature-matrix hash, configuration
  hash, recipe); the arm caches store the arm-constant hash. Anything that does not match is
  rebuilt, never reused. All cache writes are atomic (temporary file, then `os.replace`); an
  unreadable file is removed, rebuilt and listed under `cache_rebuilds` in the manifest.
* The generator cache stores the SHA-256 of the checkpoint that produced it and the arm caches carry
  the same hash, so a run resumed with another checkpoint regenerates the synthetic ECG and every
  boundary, feature and CV result that depends on it. The feature part file carries the cycles hash
  as the final feature file does.
* A figure that fails its layout check is reported, the other figures are still drawn, and the
  error is raised again at the very end of the notebook, after the manifest (which lists it under
  `figure_errors`) is on disk.
* Worker exceptions never masquerade as short recordings: the recording is dropped with the reason
  "worker error" and the exception is in `arm_failures.json`.

## Local dry run (30 Sep to 1 Oct, Windows, CPU only, sequential pool)

`DRY_RUN = True` on a fresh folder ran every stage to completion in 63.3 min: generator 46.3 min
(377 windows on the CPU), features 6.1 min, timing budget 6.3 min, classifiers and evaluation
2.5 min, everything else under a minute each. The subset was 120 recordings from 120 patients (60
murmur-present with grading 26 / 12 / 22, timing 30 / 18 / 12, shape 30 / 14 / 16; 60 murmur-absent);
all 120 were kept, no task was skipped, and every task had 5 folds. Arm A used the primary detector
path in 120 of 120 recordings, with 90 peaks within 100 ms of a window join, no join
de-duplication needed, and 71 close pairs dropped after realignment. Arm B fell back in 20.0 % of
recordings with a median supported fraction of 0.74 (757 rescues, 684 insertions, 691 removals);
the build without the rescue rule gave 57.5 % and 0.43 on its 40-recording subset. Arm D met 5
interior annotation gaps in 3 % of recordings. E2's cycle count equalled arm C's in every recording;
its median implied rate was 55 bpm against 104 bpm annotated, with a median annotated coverage of
0.56 (87 of 120 recordings below 0.8). The gate cell printed the sample check and "not applicable".
Both bootstrap oracle checks (detection, grading) and both sklearn equivalence checks passed, no
cache was rebuilt and no worker failed.

Further checks were run on that folder or on copies of it, none of them a second full run: resumed
passes of the final notebook (every stage from the caches; figures and manifest rewritten); a
resume with three deliberately truncated cache files (a feature matrix, a CV result, a fold file),
which were detected, rebuilt bit-identically and listed under `cache_rebuilds`; and the whole
notebook with `VALIDATION_APPLICABLE` forced on in the gate cell, which exercised the reference
recipe, the interval rule, the reference-rule set, the skipped figure cells after a failed gate and
the manifest that records the failure.

**The gate on the full set, locally.** The gate code of the notebook (cells 31 and 34, on a CPU)
was also run on all 3006 recordings for arms C and E, with features computed through the notebook's
own loader, annotation parser, equal-time control and extractor. The folds it draws are identical
to the reference study's stored folds on all four tasks and the sample matches (3006 / 874,
612 / 178, 608 / 177, 605 / 176). Reference-recipe macro-F1 and deviation, detection / grading /
timing / shape: arm C 78.2 (-1.0) / 56.8 (-1.5) / 46.0 (-3.0) / 46.0 (-0.6); arm E 68.9 (-0.7) / 50.8 (-0.2) / 41.2 (+1.7) / 40.1 (+1.1). Worst deviation 3.0 points; 7 of 8 cells are within the
3-point tolerance, 1 more are within because the reference lies inside their 95 % interval, 0
are outside; the gate passes on the kept set under the rule above. A CUDA run draws different dropout masks than a CPU
run, so the Colab cells will differ from these by seed-level noise. The forked pool, the GPU path
and the Drive search can only be exercised on Colab.


---

# CNN baseline notebook (`R1_CNN_Baseline.ipynb`, reviewer comment R1-m5)

| file | role |
|---|---|
| `build_cnn_baseline.py` | builder. Reads the ORIGINAL Milestone-3 notebooks at build time (`MILESTONE3_DATA_PREPARATION.ipynb`, `MILESTONE3_DATA_AUGMENTATION_V5.ipynb`, `MILESTONE3_MODEL_TRAINING.ipynb`, plus `MILESTONE4_TEMPORAL_V4.ipynb` for the patient-level label coder that fed V5), extracts 12 code blocks verbatim by def/class name through `ast`, verifies 25 procedural snippets it re-uses against the originals, validates the notebook with nbformat, py_compiles every code cell, scans every cell for em / en dashes and every markdown cell for the banned-word list, and only then writes the file (a failed check leaves the notebook on disk untouched). Override the source paths with `M3_DATAPREP_NB`, `M3_AUG_NB`, `M3_TRAIN_NB`, `M4_TEMPORAL_NB`. Run `python build_cnn_baseline.py` to regenerate. |
| `R1_CNN_Baseline.ipynb` | the Colab notebook. 35 cells (18 code, 17 markdown), runs top to bottom, resumable. |

## What it does

Retrains the article's PCG-only CNN+attention baseline (MFCC 40 + delta + delta-delta, 120 x 101 map per 4 s chunk; `MurmurClassifier` 455,042 parameters, `MultiLabelClassifier` 521,737 parameters; AdamW 1e-3 / 1e-4, batch 64, up to 100 epochs, `ReduceLROnPlateau(max, 0.5, 5)`, early stopping patience 10 on the validation AUC, best state restored) under the ten patient-grouped folds exported by `R1_Gating_Campaign.ipynb`, with a patient-disjoint inner validation split, and scores it at RECORDING level (mean of chunk softmax probabilities per recording, argmax). Chunk-level metrics are reported alongside. Every pooled metric carries a patient-level bootstrap 95% CI (2000 resamples); per-fold accuracy and macro-F1 give the SD the reviewer asked for. The multilabel model is trained once per characterisation task on that task's folds (so its predictions for the task are always out of fold) and only the task's head is scored.

## Runtime and flags

Colab Pro, High-RAM, GPU (A100 preferred; L4 about 2 to 3 times slower). Full run: an estimate of about 3 to 4 hours on an A100 (chunk cache 10 to 20 min of Drive I/O, murmur folds about 90 min, characterisation folds 60 to 90 min, continuity section about 30 min). That estimate was made with mixed precision on and TF32 at its default; the article run is full fp32 and may take longer. Mixed precision is off by default (`USE_AMP = False`; the original trained in fp32), and `seed_everything` switches TF32 off for matmul and cuDNN, as the gating notebook does; the manifest records `environment.amp`, `matmul_tf32` and `cudnn_tf32`. If `USE_AMP` is switched on, the manifest lists it under `deviations` as well. Drive space: about 4 GB for the CNN output folder (chunk cache, augmented-feature caches of about 200 MB per murmur fold, models, predictions).

Run order for the article:

1. Run `R1_Gating_Campaign.ipynb` to completion (full run, no `_dry` or `_limitN` suffix on its folder).
2. Open `R1_CNN_Baseline.ipynb` with a GPU, High-RAM runtime; leave every flag at its default.
3. Runtime > Run all. The paths cell prints the gating folder it bound; copy that path into `RUN_DIR` after the first session, so a newer gating run made between sessions cannot be bound instead.
4. After a disconnect: Runtime > Run all again. The chunk cache and every finished fold are resumed; Sections 10, 11 and 13 (scoring, continuity, merge) are recomputed on every run, and Section 11 retrains its four models (about 30 min) after any resume. Running with `RUN_CONTINUITY = False` rewrites `results.json` without the `continuity` block, so keep it `True` for the last pass.

The MANUAL OVERRIDE cell (Section 1) holds every flag:

| flag | default | meaning |
|---|---|---|
| `DRY_RUN` | `False` | 40 recordings (20 Present with all three characterisation labels, 20 Absent, whole patients), 3 epochs, the gating folds merged into 2 super-folds, AMP off, 200 bootstrap resamples, joint balancing capped at 300 iterations (verbatim limit 10,000 otherwise). About 3 to 5 min on a CPU. On a small recording set the original 70/15/15 split of Section 11 can leave a part empty; that continuity run is then skipped and listed under `continuity.skipped`. Outputs go to `cnn_baseline_dryrun/`, never to the full run's folder; a gating `_dry` run folder is preferred when one exists. |
| `LIMIT_RECORDINGS` | `None` | recording subset with full epochs and all folds (smoke test only); outputs go to `cnn_baseline_limit<N>/`; may sit on a gating `_limitN` folder. |
| `DRIVE_ROOT` | `'/content/drive/MyDrive'` | where Drive is mounted; the dataset, the gating runs (`JBHI-Revision/R1_Gating_Campaign/`) and the original split files are resolved under it. |
| `LABEL_MODE` | `'patient'` | `'patient'` = coding of the gated arms (verbatim `get_recording_labels` from the M4 feature notebook: every recording of a Murmur = Present patient is Present, Unknown dropped, grading I/II/III = 0/1/2, timing Holosystolic/Early/Mid = 0/1/2, shape Plateau/Decrescendo/Diamond = 0/1/2, everything else -1). `'location'` = the original M3 coding (verbatim `is_murmur_audible_at_location` and `get_murmur_characteristics`: Present only at the annotated murmur locations, higher grades / Late-systolic / Crescendo merged into class 2). Both distributions are printed in Section 4 whatever the mode. A `'location'` run writes to its own `cnn_baseline..._location/` folder. |
| `RUN_DIR` | `None` | pin a gating run folder; `None` picks one under `MyDrive/JBHI-Revision/R1_Gating_Campaign/` that has a `folds/` subfolder with fold files. The gating notebook names its folders `<stamp>` (full run), `<stamp>_dry` (dry run) and `<stamp>_limitN` (`LIMIT_RECORDINGS = N`): a full CNN run takes the newest full folder and skips every `_dry` and `_limitN` folder, printing each skip; a `DRY_RUN` prefers the newest `_dry` folder and otherwise takes the newest other one. The paths cell prints which folder was bound and why. A full run stops when it is pinned to a `_dry` or `_limitN` folder, when the gating folder's own `run_flags.json` holds `dry_run` or `limit_recordings`, or when its fold files hold fewer than 2,500 detection or 500 characterisation recordings. With `RUN_DIR = None` the newest full gating run is bound on every session, so pin `RUN_DIR` for the article run after the first session. |
| `FOLD_SOURCE` | `'gating'` | `'self'` draws `StratifiedGroupKFold(10, shuffle, seed 42)` folds in the notebook (smoke tests only; outputs go to `cnn_baseline_selffolds/`). A `DRY_RUN` with no gating run on Drive falls back to `'self'` and records a deviation. |
| `RUN_CONTINUITY` | `True` | Section 11 (original 70/15/15 split). |
| `USE_AMP`, `CACHE_AUG_FEATURES`, `SEED`, `N_BOOT`, `ALLOW_LABEL_MISMATCH`, `GATED_KEY_MAP` | `False`, `True`, 42, 2000, `False`, `{}` | mixed precision (off for the article run; `True` is recorded as a deviation); cache augmented features per fold on Drive (about 200 MB per murmur fold); seeds; bootstrap replicates; whether a label disagreement with the fold file stops the run; manual array map for the merge cell. |

Environment variables `R1_DRY_RUN`, `R1_DRIVE_ROOT`, `R1_RUN_DIR`, `R1_FOLD_SOURCE`, `R1_LABEL_MODE`, `R1_LIMIT_RECORDINGS`, `R1_RUN_CONTINUITY`, `R1_ORIGINAL_SPLITS_JSON`, `R1_ORIGINAL_CHUNK_META` override the same flags (used by the local self-test; harmless on Colab).

## What it needs from the gating run

* The four fold files exactly as `build_gating_campaign.py` writes them: `folds/detection.npz`, `folds/grading.npz`, `folds/timing.npz`, `folds/shape.npz`. The gating notebook calls the murmur task `detection`; the CNN notebook keeps `murmur` internally and maps the names through `GATING_TASK = {'murmur': 'detection', ...}` wherever it reads gating files (fold file names, `oof_predictions.npz` arrays, `results.json['arms']`). Each file holds `names`, `patient_ids`, `y`, `fold_id`, `n_splits`, `keep_names`, `mask_over_keep`, `seed` and the `train_NN` / `test_NN` index arrays into `names` (integer indices; a boolean mask over `names` and plain name arrays are also accepted, nothing is guessed from the values). The loader also accepts `recording_names` / `rec_names`, `train_idx_k` / `test_idx_k`, `fold_k_train`, object arrays of folds, or a `.json` with the same content. The join is by recording name, never by position. Assertions: every test recording exists in the CNN index and has a valid label; no patient in both train and test of any fold; test folds pairwise disjoint; if the fold file carries `y`, it must equal the CNN labels (0 disagreements expected in `patient` mode; in `location` mode the disagreement is expected and is recorded as a deviation); a full run needs at least 2,500 detection and 500 characterisation recordings per fold file.
* For the merge cell (Section 13): `<RUN_DIR>/oof_predictions.npz` with `<gating task>__y`, `__names`, `__patient_ids`, `__fold_id`, `<gating task>__<arm>__<clf>__pred` / `__proba` and the top-level `arms` and `tasks` arrays, and `<RUN_DIR>/results.json` (`run.arm_letters` for labels; `arms.<gating task>.<arm>.<clf>.macro_f1_ci` for the arm CIs in figure B2). The cell prints the discovered (arm, task, classifier) table, aligns by recording name, checks that the gating label (`<gating task>__y`) equals the baseline label on every aligned recording (the run stops on a disagreement and prints one summary line with the number of arm x task groups checked; under `LABEL_MODE = 'location'` the arm is skipped and a deviation recorded), and writes `paired_vs_gated.json` with, per arm and task: macro-F1, accuracy and balanced accuracy of the arm and of the CNN on the aligned recordings with their 95% CIs on those recordings (`arm_ci95_aligned`, `baseline_ci95_aligned`), the paired patient-bootstrap difference (same patient resamples for both), its 95% CI and P(diff > 0), the fold-wise paired t-test on the shared folds, the label-check note and, for macro-F1, which source figure B2 used for the arm's point and interval (the gating `results.json` when every arm recording is aligned, otherwise the aligned-set bootstrap). If the gating outputs are missing the cell reports `skipped` and the rest of the notebook is unaffected. `GATED_KEY_MAP = {'<arm label>': {'<task>': {'probs': <array>, 'rec': <array>, 'y': <array>, 'clf': 'mlp'}}}` overrides discovery.

## Stages and outputs (all under `<RUN_DIR>/cnn_baseline<suffix>/`)

The suffix is empty for the article run and otherwise `_selffolds`, `_dryrun`, `_limit<N>`, `_location` (`LABEL_MODE = 'location'`) or their combination, so a dry run, a subset run or a location-label run never shares caches or results with the article run. `run_flags.json` (dry_run, limit_recordings, fold_source, seed, epochs, label_mode, use_amp) is written on a fresh run; a resume into a folder made under different flags is refused, so a label-mode change or an AMP toggle can neither resume into nor overwrite the article folder.

| section | what | writes |
|---|---|---|
| 2 | dataset resolution (the expected complete copy is tried first; otherwise every `training_data` folder under Drive is counted and the maximum paired count wins), gating run folder (`_dry` / `_limitN` rule above, gating `run_flags.json` check), output folder and run flags | `run_flags.json` |
| 3 | verbatim preprocessing, MFCC, augmentation; Butterworth pole check (max pole radius 0.9864 at 4 kHz, so the verbatim `ba` design is used; `sos` if any radius is 1 or more, or per recording if the `ba` output is not finite) | |
| 4 | recording index and both label codings | |
| 5 | fold loading, join by name, integrity assertions, DRY_RUN subset | |
| 6 | chunk cache: 4 kHz, 25 to 400 Hz zero-phase, per-recording z-score, 4 s chunks with a circularly padded trailing remainder of at least 2 s, MFCC per chunk; maps stored standardised in float16 with float32 mean/std (max round-trip error 3.9e-03 map-std units as printed by the notebook, recorded in the manifest), z-scored audio kept for every chunk of a Murmur = Present patient; per-recording chunk counts cross-checked against the original `chunk_metadata.csv` when it is on Drive | `cache/chunks_<hash>_*` |
| 7 | models (verbatim), training utilities | |
| 8 | balancing: murmur Present chunks oversampled with `augment_audio` until Present = Absent (verbatim rule); multilabel iterative joint balancing towards 33.3% per class across grading, timing and shape (vectorised port with the verbatim selection rule, asserted equal to the pandas loop) | `cache/aug_<mode>_<task>_fold<k>.npz` |
| 9 | ten folds per task, inner split `GroupShuffleSplit(0.15, seed 42)` by patient (next seed tried if a class is missing; seed recorded; if no seed in 42..61 holds every class, the best available split is used and a deviation is recorded with fold, task and seeds tried), online augmentation, training, chunk-level test predictions. Each `aug_*` cache stores the SHA-1 of its sorted train chunk ids and the chunk-cache hash; each `preds_*` file stores the fold's test recordings, train recordings and epoch budget. A fold is resumed only when all of these equal the current plan; otherwise it is recomputed and the reason printed and listed under `cache_events` in the manifest. The deviations noted while a fold was trained are stored in its `train_*.json` and restored on resume, so a resumed run writes the same deviation list as an uninterrupted one | `cache/preds_*`, `cache/train_*.json`, `models/<mode>_<task>_fold<k>.pt` |
| 10 | scoring on the gating notebook's definitions: macro-F1 over the fixed label set of the task (absent or never-predicted class counts as 0, `f1_score(labels=range(C), zero_division=0)`, cross-checked against sklearn), AUROC undefined when a class is absent, pooled OOF at recording and chunk level, per fold with population SD, bootstrap CIs, majority baseline, confusion matrices; the mirrored `arms` block | `results.json`, `oof_predictions.npz` |
| 11 | continuity on the original 70/15/15 split: C1 original protocol (location labels, chunk-level 80/20 early-stopping re-split of pooled train+val, test split at chunk and recording level under both label codings) and C2 corrected protocol on the same split (patient labels, early stopping on the original validation split). The split is read from `Processed-Data/patient_splits.json` when present and recreated with the verbatim `train_test_split` calls otherwise; both are compared when both exist (identical on the local check). | `results.json` (`continuity`) |
| 12 | figures B1 (recording-level macro-F1 with CI), B3 (chunk vs recording level), B4 (confusion matrices, 2 x 2 grid). Font family set explicitly to DejaVu Sans, as in the gating notebook, so local and Colab output match | `figures/*.pdf`, `figures/*.png` (600 dpi) |
| 13 | merge with the gated arms, figure B2 forest plot. Every (arm, task, classifier) group found in the gating `oof_predictions.npz` is compared and written to the JSON, including the ablation groups the gating notebook stores there (`jitter_*`, `window_*`, `pcg_seg_fullscaler`, `pcg_seg_reduced`). The figure shows only the arms listed in the file's top-level `arms` array (A, B, C, D, E, E2; MLP and RF), in that order and at the same height in every panel, with one shared x range for the difference row | `paired_vs_gated.json` (`arms`, `b2`), `figures/B2_*` |
| 14 | manifest: versions, GPU, seeds, filter design, label mode and distributions, chunk-cache facts, fold-file hashes and integrity counts, run flags, cache events, stage seconds, deviations, SHA-256 of every verbatim block | `run_manifest.json` |

`results.json` layout, native part: `tasks.<task>.recording_level.pooled` (accuracy, macro_f1, balanced_accuracy, per_class_recall, per_class_specificity, sensitivity/specificity for detection, auroc, auprc_macro, auprc_positive, confusion_matrix, `ci95` including `auroc_n_resamples_defined`), `.per_fold` (with `n_classes_present`), `.fold_summary` (`fold_macro_f1` list, `fold_macro_f1_mean`, `fold_macro_f1_sd` = population SD as in the gating notebook, `fold_macro_f1_sd_sample` = sample SD, the same four for accuracy, balanced_accuracy and auroc, `fold_auroc_n_undefined`), `.majority_baseline`; `chunk_level` mirrors it; `training` per fold (sizes, augmentation counts, joint-balance iterations, epochs run, best epoch, inner-val seed, seconds). All rates in the native part are fractions in [0, 1]. Task names in the native part are the CNN's (`murmur`, `grading`, `timing`, `shape`).

`results.json` layout, mirrored part: `arms.<gating task>.cnn_pcg_only` repeats the recording-level result in the layout of `arms.<task>.<arm>.<clf>` of the gating `results.json`, so one table script reads both files with the arm name `cnn_pcg_only`:

| gating `arms.<task>.<arm>.<clf>` entry | CNN `arms.<gating task>.cnn_pcg_only` entry | units |
|---|---|---|
| `n`, `n_patients` | same | counts |
| `majority.{majority_class, accuracy, macro_f1, class_counts}` | same (one level up, next to the metrics) | percent |
| `accuracy`, `accuracy_ci`, `macro_f1`, `macro_f1_ci`, `balanced_accuracy`, `balanced_accuracy_ci` | same | percent |
| `sensitivity`, `sensitivity_ci`, `specificity`, `specificity_ci` (per class) | same | percent |
| `auroc`, `auroc_ci`, `auprc`, `auprc_ci`, `score_note` | same | [0, 1] |
| `confusion_matrix` | same (rows = truth) | counts |
| `fold_macro_f1`, `fold_macro_f1_mean`, `fold_macro_f1_sd`, `fold_accuracy`, `fold_accuracy_mean`, `fold_accuracy_sd` | same; `_sd` is the population SD in both files | percent |
| (no equivalent) | `chunk_level` (the same entries at chunk level), `level`, `label_mode`, `n_folds` | |

The gating task names apply in the mirrored part (`detection` = the CNN's `murmur`). `oof_predictions.npz` arrays: `<task>__rec_names`, `__patient_ids`, `__y_true`, `__probs`, `__pred`, `__fold`, `__n_chunks`, and `__chunk_rec`, `__chunk_ids`, `__chunk_y`, `__chunk_probs`, `__chunk_fold` (CNN task names).

## Deviations from the original code (all recorded in `run_manifest.json`)

1. Early stopping. The original guard `if early_stopping(val_auc): if early_stopping.early_stop: break` could never fire (the call returns True only on improvement), which is why both original stages ran all 100 epochs. The notebook stops when patience (10) is exhausted, as the Methods state.
2. Best state. `best_model_state = model.state_dict().copy()` in the original was a shallow copy that tracked the live weights, so the saved "best" weights were the last epoch's. The notebook restores a detached copy of the best-AUC state.
3. Early-stopping set. Patient-disjoint inner split instead of the chunk-level 80/20 re-split of pooled train+val chunks (the leaky protocol is reproduced only in Section 11 C1 for the continuity comparison).
4. Online augmentation on cached maps. The gain part is exact: gain g becomes an offset of `sqrt(n_mels) * 20 * log10(g)` on MFCC coefficient 0 in every frame, deltas unchanged (the notebook asserts this at runtime to 1e-3). The shift part is an approximation: a circular shift of s samples becomes a circular roll of `round(s / 160)` frames. Measured against an audio-domain roll followed by re-extraction: a whole-frame shift (s a multiple of 160 samples) differs in 6 to 8 frames at each edge of the map (librosa's centred reflect padding and the delta interpolation); a sub-frame shift, the general case, differs in every frame by a small amount (rms about 0.02 to 0.07 standardised units, up to 3.3 at the edges); |s| below 80 samples, about 10% of the draws, rounds to zero frames and does nothing. An exact alternative (keep the z-scored audio of every training chunk, about 1.1 GB, and re-extract the MFCC of the 30% shifted samples per batch) is not implemented in this revision.
5. Feature cache round-tripped through float16 so fresh and resumed runs use identical inputs.
6. Joint balancing: pool restricted to chunks with all three labels valid before balancing (the original balanced first and dropped invalid-label chunks at load time).
7. Batching: shuffled GPU-resident tensors instead of a DataLoader; a single-sample last batch is skipped (BatchNorm).
8. Scoring at recording level is primary; the original reported chunk-level accuracy.
9. Only when `USE_AMP = True`: mixed precision (torch.autocast float16) during training and inference; the original trained in fp32. Off by default, so the item is absent from the manifest of the article run.

Run-time items join the list when they occur: `sos` filter fallback, `location`-mode label disagreement with the fold file, an inner validation split without every class (fold, task, seeds tried), an accuracy fallback for the training monitor, and, under `LABEL_MODE = 'location'`, gated arms skipped in the merge cell.

Not deviations but worth knowing: the printed class-name lists for timing and shape in the original training/test notebooks did not match the integer coding of the data preparation; the notebook uses the coding of the gated arms (`Holosystolic, Early systolic, Mid systolic`; `Plateau, Decrescendo, Diamond`), verified against the fold-file labels. Forty-three recordings with a numbered suffix (`_MV_1` and similar) are parsed by the original rule as location `1` or `2`, so the original location-aware coding labels them not audible; this is kept verbatim in `location` mode and printed.

## Fidelity checks the notebook performs on every run

Parameter counts equal the original (455,042 / 521,737); the chunker gives 2 full + 1 padded chunk for 10 s and drops a 1.9 s remainder; the feature-domain gain moves c0 by `sqrt(n_mels) * 20 * log10(1.1)` for a gain of 1.1 and leaves every other row unchanged (asserted to 1e-3); the vectorised joint-balancing planner equals the verbatim pandas loop on 40 reference picks; the numpy metric implementation equals sklearn on the gating definitions (accuracy, macro-F1 with `labels=range(C), zero_division=0`, balanced accuracy, AUROC) on the pooled predictions; the gating labels equal the baseline labels on every recording aligned in the merge cell; label counts in `patient` mode equal the V5 counts (616 Present recordings; 612 / 608 / 605 with grading / timing / shape labels); fold-file labels equal the notebook's labels; per-recording chunk counts equal the original `chunk_metadata.csv` (0 mismatches on the local check; 17,176 chunks in the original); the recreated 70/15/15 split equals the original `patient_splits.json` (identical on the local check).

## Local smoke test (done before hand-over)

Executed end to end with nbclient on a CPU (Python 3.13, torch 2.11 CPU, librosa 0.11, numpy 2.2; a numpy 2.4 kernel fails inside librosa because numba 0.61 needs numpy 2.2 or lower, so use a kernel with numpy 2.2) against the local dataset copy and two fixture gating runs written in the exact schema of `build_gating_campaign.py` (file names `folds/detection.npz` and so on, array names `detection__y`, `detection__names`, `detection__patient_ids`, `detection__fold_id`, `detection__<arm>__<clf>__pred` / `__proba`, top-level `arms` and `tasks`, `results.json['arms'][task][arm][clf]` in percent): a full-style folder `<stamp>` (3,002 / 612 / 608 / 605 kept, 10 folds) and a `<stamp>_dry` folder (40 patients, one recording each, folds reduced as the gating notebook reduces them: 10 / 3 / 2 / 3).

* Run 1 (DRY_RUN, `RUN_DIR = None`): bound the `_dry` folder and printed why; 0 cell errors in 263 s; fold integrity passed (0 label disagreements, patient-disjoint, disjoint test folds); the c0 gain assertion passed; the inner-split fallback recorded six deviations with fold, task and seeds tried (expected on 5 to 11 training recordings); sklearn cross-check passed for all four tasks; results.json (native and mirrored `arms` block), oof_predictions.npz, paired_vs_gated.json (12 arm x classifier labels, 48 groups, label check 0 disagreements everywhere), figures B1 to B4 and run_manifest.json were written to `cnn_baseline_dryrun/`; C1 multilabel was skipped and recorded because the 70/15/15 split left no Present test recording in the 36-recording subset.
* Run 2 (same flags): `run_flags.json` matched, chunk cache and all eight folds resumed with the fold plan and epoch budget verified.
* Run 3 (fixture folds re-drawn with another seed, same kept set): chunk cache reused, every `aug_*` cache rebuilt (train chunk ids differ) and every fold recomputed with the printed reason "test recordings differ"; both listed under `cache_events` in the manifest.
* Negative tests on cells 0 to 2: a full run pinned to the `_dry` folder stops with RuntimeError before writing anything; a resume into a folder whose `run_flags.json` holds other flags (seed 7, epochs 100) stops with RuntimeError naming the differing flags.
* Integration against the gating notebook's own dry-run output (`<stamp>_dry` with four fold files of 120 / 60 / 60 / 60 recordings and 5 folds, `oof_predictions.npz` with 186 arrays, `results.json`): the CNN DRY_RUN pinned to that folder ran all cells without error, fresh and resumed; 0 label disagreements with the fold files; the merge cell found 17 arms (the six declared arms with MLP and RF plus the jitter, window, scaling and redundancy groups), aligned 84 arm x task groups by recording name with 0 label disagreements, and wrote the paired statistics for all of them; a resumed run restored the six inner-split deviations of the cached folds and wrote the same 15-item deviation list as the fresh run. An incomplete gating folder (two fold files only) stops the folds cell with a FileNotFoundError naming the missing task.

The DRY_RUN metrics themselves are meaningless (36 recordings, 3 epochs) and are not reported here.
