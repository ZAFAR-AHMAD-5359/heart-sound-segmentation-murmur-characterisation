# Heart-sound segmentation and phase-based features for paediatric murmur characterisation: code, notebooks and outputs

This is the material of the study in manuscript EHJDH-D-26-00501, revision R1 (*European Heart Journal - Digital
Health*). It holds the analysis code, the notebooks as they ran, their outputs, the frozen network weights, and the
pre-registration records named in the Data Availability statement.

Read section 1 for the study, section 3 for the experiment map, and section 5 for the source of each reported number.

## 1. The study in brief

**Question.** The classifiers read 258 of their 464 features inside four cardiac phases: S1, systole, S2 and diastole.
How much of the accuracy of murmur grading, timing and shape depends on where the phase boundaries fall? Can an
automatic method trained without any annotation of the CirCor DigiScope dataset come near the expert-reviewed boundaries?

**Design.** Every method cuts the same 3,000 recordings (816 participants) of the CirCor DigiScope dataset into cycles of four
phases. Every method then feeds the same 464 features, the same classifiers and the same participant-grouped ten-fold split. Only the
source of the boundaries differs.

| Method | Boundaries | Role |
|---|---|---|
| E | Equal-time windows at 75 beats per minute, four equal quarters | Baseline |
| E2 | Equal pseudo-cycles, as many as the annotation has cycles | Descriptive |
| C | Expert-reviewed segmentation of the CirCor DigiScope dataset | Reference only |
| D | Annotated cycle starts, phases at fixed fractions of each cycle | Fixed against adaptive phases |
| A, B | R-peaks of the synthetic ECG of the companion study; fixed fractions (A) or acoustic refinement (B) | Earlier generator |
| S, S_pub | Springer segmenter; heart-rate search 40 to 220 (S) or the published 30 to 120 beats per minute (S_pub) | Automatic comparators |
| T | Timing network: S1, S2 and R-peak heads, trained with the ECG as teacher on six adult paired PCG-ECG datasets | Primary new method |
| W | The same network, trained only to rebuild the ECG | Reconstruction test |
| T_ac | As T, trained on S1 and S2 labels from the PCG alone | Control for the ECG teacher |
| CNN | No boundaries: a PCG-only convolutional network | Detection reference |

Each automatic method also has a matched-coverage version (suffix `_annot`). It keeps only the cycles that start inside
the annotated stretches, so it is scored on the same audio as C. The code and its output files call each boundary
method an "arm" (for example `arm_failures.json`).

**Endpoint.** The composite macro-F1: the mean macro-F1 over grading, timing and shape, from a five-seed ensemble of
multilayer perceptrons. The intervals come from 2,000 bootstrap draws of participants, one set shared by every method.

## 2. Rules the code enforces

- **Pre-registration first.** `2_preregistration/extension_study/PREREG_EXT.md` fixed the methods, endpoint, hypotheses
  and folds on 16 October 2026, 16:07 Pakistan time, before any method was scored on the CirCor DigiScope dataset. Its SHA-256 sits beside it.
  The dated change log `CHANGES_AFTER_FREEZE.md` records every later decision with the results known at the time.
- **No data of the CirCor DigiScope dataset in training.** T, W and T_ac learnt and were selected on paired adult data alone. The test
  `1_code/extension_study/tests/test_firewall.py` fails if a training or selection module refers to a path of the CirCor DigiScope dataset.
- **One scoring per method.** Each scoring went into the append-only ledger `2_preregistration/ledger.csv`, with a time
  stamp and the SHA-256 of the predictions.
- **Builders, not hand edits.** A Python builder writes each Colab notebook; no notebook was edited by hand. A notebook
  carries the code it runs and checks the hashes of its inputs. Its `run_manifest.json` records the package versions,
  the settings and the hashes.
- **Reproduction controls.** Each later run repeats a method of an earlier run, and a gate checks that the repeated
  predictions match before the analysis. For example, method E of Run 2 must equal method E of Run 1.
- **No typed numbers.** Scripts write every results table of the paper from the result files (section 5).

## 3. Experiment map

The steps are listed in the order they ran. Dates are in 2026, Pakistan time. Code paths are relative to
`1_code/extension_study/`, and outputs to `3_results/`. Each run folder holds the notebook as run (`*.ipynb`),
`run_manifest.json`, `results.json` and the out-of-fold predictions `oof_predictions.npz`.

| # | Date | Step and question | Code and notebook | Outputs | Analysis and its output | Paper |
|---|---|---|---|---|---|---|
| 0 | 15 Oct | **Folds.** No participant may sit in both training and test data; 70 children enrolled twice under two IDs. | `extstudy/splits.py` | `folds_child/` | | Methods |
| 1 | 16 Oct | **Run 1.** H1: do Springer boundaries (S) beat equal-time windows (E)? The run also scores C, D, E2, A, B, S_pub and the matched-coverage methods. | `notebooks/build_run1.py` writes `Extension_Run1.ipynb` | `run1_20261016_1516/` | `extstudy/stats.py`: `run1_stats/` | Tables 3, 4 |
| 2 | 17 Oct | **CNN.** A detection reference with no phase boundaries, on the same folds. | `build_cnn_run1.py` writes `Extension_Run1_CNN.ipynb` | `cnn_baseline/` | `run1_stats/cnn_stats.json` | Results |
| 3 | 15 to 18 Oct | **Labels, then training of T and W.** Teacher R-peaks on the paired ECG; S1 and S2 marks on the PCG envelope; a label gate against the PhysioNet/CinC 2016 reference annotations; the training pack; three seeds per network, selected on paired validation data and frozen. W asks whether rebuilding the ECG waveform adds anything over marking the heart sounds. | `extstudy/qrs.py`, `relabel.py`, `labels.py`, `refcheck.py`, `corpus.py`; `build_train.py` writes `Extension_Train_TW.ipynb` | `tw_training_and_freeze_20261018_0908/`: weights, `freeze.json`, `paired_test.json` | `check_export_paired.py`, `check_w_val.py` | Supplement S-TW |
| 4 | 18 Oct | **Run 2.** H2 to H4: does T beat E, is T non-inferior to S, and does T beat S? W is an estimate. | `build_run2.py` writes `Extension_Run2.ipynb` | `run2_20261018_1358/` | `merge_run2.py` with `extstudy/stats.py`: `run2_scoring/` | Tables 3, 4; Figure 3 |
| 5 | 18 to 19 Oct | **Experiment 0, the beat-error budget.** How much beat-level error (missed or added cycle starts, jitter) can expert boundaries carry before the gain over E is lost? | `extstudy/corrupt.py`; `build_exp0.py` writes `Experiment0_BeatErrorBudget.ipynb` | `exp0_20261018_1659/` | `exp0_analysis.py`: `exp0_analysis/` | Figure 4; S-Budget |
| 6 | 19 Oct | **S2 follow-up** (exploratory, not pre-registered). Why do the automatic methods keep less of the gain than the curve of Experiment 0 predicts? | `s2_phase_analysis.py`, `s2_variability_analysis.py` | `s2_followup/` | | S-Budget |
| 7 | 20 to 21 Oct | **Experiment 0b.** For the same amount of error, does failure in some recordings cost more than error spread over all of them? | `build_exp0b.py` writes `Experiment0b_FailureConcentration.ipynb` | `exp0b_prep/`, `exp0b_20261021_0913/` | `exp0b_analysis.py`, `exp0b_where_lost.py`: `exp0b_analysis/` | S-Budget |
| 8 | 21 to 22 Oct | **Experiment 0c.** Does the same failure cost more on the recordings where Springer fails than on as many recordings drawn at random? | `exp0c_prep.py`, `extstudy/placement.py`; `build_exp0c.py` writes `Experiment0c_FailurePlacement.ipynb` | `exp0c_prep/`, `exp0c_20261022_0916/` | `exp0c_analysis.py`: `exp0c_analysis/` | S-Budget |
| 9 | 22 Oct | **The remaining Run 2 estimates.** Heart-rate bands, infants, the sensitivity analysis without fallback recordings, and T against the CNN. | `run2_estimation.py` | `run2_estimation/` | | Table 5; S-Subgroups |
| 10 | 23 Oct | **Secondary metrics** (Reviewer 3): balanced accuracy, class-wise sensitivity, AUROC and AUPRC. | `secondary_metrics.py` | `secondary_metrics/` | | S-Metrics |
| 11 | 23 and 29 Oct | **Revision checks.** Contrasts with S_pub; contrasts with C at matched coverage; results without the pregnant participants; end-to-end characterisation with detection misses counted as failures. Then, from the independent code review: bootstrap-seed sensitivity, detection misses by class, fallback counts. | `revision_checks.py`, `revision_checks2.py` | `revision_checks/`, `revision_checks2/` | | Table 4; Supplement |
| 12 | 29 Oct | **Experiment 1, in-band interference.** How much of the gain survives noise confined to 20 to 500 Hz and not locked to the cardiac cycle? Gaussian noise, bursts, and ESC-50 crying, breathing and coughing at 10 and 0 dB. | `extstudy/noise.py`; `build_exp1.py` writes `Experiment1_InBandNoise.ipynb` | `exp1_20261029_1331/` | `exp1_analysis.py`: `exp1_analysis/` | Table 6; S-Noise |
| 13 | 29 Oct | **Experiment 2, training of T_ac.** Is the ECG teacher needed? The design of T, trained on S1 and S2 labels from the PCG alone. | `extstudy/labels_acoustic.py`; `build_train_ac.py` writes `Extension_Train_AC.ipynb` | `ac_training_and_freeze_20261029_1824/` | | S-Control |
| 14 | 29 to 30 Oct | **Run 3.** T_ac scored once on the CirCor DigiScope dataset, as T was. | `build_run3.py` writes `Extension_Run3.ipynb` | `run3_20261029_1955/` | `exp2_analysis.py`, `write_exp2_results.py`: `exp2_analysis/` | Tables 3, 4; S-Control |

The plans of Experiments 0 to 2 are in `2_preregistration/error_budget_experiments/`. Each plan was hashed before its
boundary sets, corrupted audio or labels existed. Experiments 0b, 0c, 1 and 2 were designed after the main results
were known, and each plan states what was known when it was written. Results notes sit beside the outputs, for
example `run1_stats/RUN1_RESULTS.md`, `run2_scoring/RUN2_RESULTS.md` and `exp1_analysis/EXP1_RESULTS.md`.

## 4. Folder map

```
README.md                        this map
FILE_INDEX.csv                   every file with its size and SHA-256, and whether the journal zip holds it
1_code/
  extension_study/               the study code
    extstudy/                    Python package: folds, teacher R-peaks, labels, training pack, network, losses,
                                 augmentation, training, inference, beat decoder, cycle export, Springer port,
                                 corruption and interference generators, pre-registered statistics (stats.py)
    notebooks/                   build_*.py (one builder per Colab notebook), the notebooks as built, and every
                                 local analysis script of the map above
    tests/                       pytest suite, one file per module (firewall test for the CirCor DigiScope dataset included)
    archive/                     the label code of versions 1 and 2, which failed their checks (kept as the record)
    README.md, CODE_DESIGN.md    how to run; module plan, data flow and the reasons for each rule
  pcgecg/                        readers for the six paired PCG-ECG datasets, the data audit, the PCG envelope
  r1_base_notebooks/             the first-revision campaign notebooks (file names R1_Gating_Campaign and
                                 R1_CNN_Baseline) and their builders; build_run1.py and build_cnn_run1.py check them
                                 cell by cell and patch them into the Run 1 and CNN notebooks. The campaign's own
                                 results are superseded (Supplement S-Changes).
  paper_tables_and_figures/      make_tables.py, make_supp_tables.py, make_supp_control.py and the figure scripts
2_preregistration/
  extension_study/               PREREG_EXT.md and its hash, RUN1_LOCK.json, CHANGES_AFTER_FREEZE.md
  error_budget_experiments/      the plans of Experiments 0, 0b, 0c, 1 and 2 with their hashes, their results notes,
                                 the run sheet of Experiments 1 and 2, the S2 follow-up note
  ledger.csv                     the append-only run ledger
3_results/                       one folder per run or analysis, named by step and time stamp (section 3)
weights/generator.pt             the synthetic-ECG generator of methods A and B (repository only)
first_submission_code/           the code released with the first submission (repository only; section 8)
```

## 5. Where each reported number comes from

| In the paper | Result files in `3_results/` | Script in `1_code/paper_tables_and_figures/` |
|---|---|---|
| Tables 3, 4 and 5; Supplement table of secondary metrics | `run2_scoring/stats_results.json`, `run2_scoring/estimation_contrasts.json`, `run2_estimation/run2_estimation.json`, `revision_checks/revision_checks.json`, `run1_stats/cnn_stats.json`, `exp2_analysis/exp2_results.json`, `secondary_metrics/secondary_metrics.json` | `make_tables.py` |
| Table 6 and the S-Noise tables | `exp1_analysis/exp1_results.json` | `make_supp_tables.py` |
| Supplement tables of S-TW, S-Subgroups and S-Budget; end-to-end, pregnancy and detection-miss tables | `tw_training_and_freeze_20261018_0908/`, `run2_estimation/`, `exp0_analysis/`, `exp0b_analysis/`, `exp0c_analysis/`, `revision_checks/`, `revision_checks2/` | `make_supp_tables.py` |
| Supplement S-Control | `exp2_analysis/`, `ac_training_and_freeze_20261029_1824/`, `tw_training_and_freeze_20261018_0908/` | `make_supp_control.py` |
| Figure 3 | `run2_scoring/stats_results.json` | `fig3_results.py` |
| Figure 4 | `exp0_analysis/`, `exp0b_analysis/`, `run2_scoring/stats_results.json`, `s2_followup/` | `fig_beat_error_budget.py` |
| Figure 5 (spectral overlap) | computed from curated recordings of the CirCor DigiScope dataset, not stored here | `fig2_psd_compute.py`, `fig2_psd.py` |

The scripts were written for the revision package, where this folder's `3_results/` is named `06_results/`. To rerun
one, point the results path at the top of the script (`RES`, `SRC` or `RUNS`) to `3_results/`.

To check a reported value without rerunning anything, load the out-of-fold predictions and score them again:
`run2_scoring/oof_predictions.npz` holds the merged predictions of Runs 1 and 2, and `extstudy/stats.py` scores them.

## 6. How to rerun

**Data.** The datasets are public and are not copied here: CirCor DigiScope 1.0.3, EPHNOGRAM, SensSmartTech and
PhysioNet/CinC 2016 on PhysioNet; the HFrEF set, FOSTER and BSSLAB on Zenodo, OSF and the King's College London
repository; ESC-50 on GitHub (Experiment 1). The manuscript gives the full references.

**Environment.** Python 3.13 with numpy, scipy, pandas, scikit-learn, PyTorch, matplotlib, PyWavelets, wfdb, soundfile,
nbformat and pytest; `pcgecg` also imports neurokit2. Each `run_manifest.json` records the versions of its run. Run 2 on
an NVIDIA A100, for example: PyTorch 2.11.0 with CUDA 13.0, numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1, pandas 2.2.3
and PyWavelets 1.8.0.

**Paths.** `1_code/extension_study/extstudy/config.py` holds every path, for a local Windows layout (`LOCAL`) and for
Google Drive on Colab (`DRIVE`). Set `PHD_LOCAL` to the folder of the raw and derived data, `LOCAL['pcgecg_code']` to
`1_code`, `LOCAL['r1_builder']` and the builder paths at the top of `notebooks/build_run1.py` and `build_cnn_run1.py`
to `1_code/r1_base_notebooks/`, `LOCAL['generator']` to `weights/generator.pt`, and `_CIRCOR_LOCAL` to the CirCor DigiScope
`training_data` folder.

**Order.**

1. Data audit and harmonised arrays: `pcgecg` on a local CPU (`1_code/pcgecg/README.md`, about 10 minutes).
2. Folds, teacher R-peaks, labels, label gate and training pack: the `extstudy` modules of step 3 of the map, on a
   local CPU (about 1 hour; `1_code/extension_study/README.md` lists the commands). Upload the pack to Drive.
3. Colab notebooks, in the order of the map: Run 1, CNN, the training of T and W, Run 2, Experiments 0, 0b and 0c,
   Experiment 1, the training of T_ac, Run 3. Each builder writes its notebook; each notebook writes a run folder.
   Run 1 takes about 4 to 5 hours on an A100, Run 2 about 1 hour, and each network training 25 to 40 minutes.
4. The local analysis scripts of the map, on the downloaded run folders (minutes on a CPU).
5. The tables and figures (section 5).

**Layout of the outputs.** The code, the tests and the usage lines of the analysis scripts refer to the authors' layout,
in which the outputs sat in `1_code/extension_study/runs/`. This folder renames them as below; copy or link them back
under `runs/` to rerun a script or a test. The ledger was `runs/ledger.csv`, and the pre-registration folder was
`1_code/extension_study/prereg/`.

| Authors' path under `runs/` | Folder in `3_results/` |
|---|---|
| `folds_child`, `run1_stats`, `run2_scoring`, `run2_estimation`, `secondary_metrics`, `revision_checks`, `revision_checks2`, `exp0_analysis`, `exp0b_prep`, `exp0b_analysis`, `exp0c_prep`, `exp0c_analysis`, `exp1_analysis`, `exp2_analysis` | same name |
| `run1/20261016_1516`, `run2/20261018_1358`, `run3/20261029_1955` | `run1_20261016_1516`, `run2_20261018_1358`, `run3_20261029_1955` |
| `exp0/20261018_1659`, `exp0b/20261021_0913`, `exp0c/20261022_0916`, `exp1/20261029_1331` | `exp0_20261018_1659`, `exp0b_20261021_0913`, `exp0c_20261022_0916`, `exp1_20261029_1331` |
| `run2_freeze/20261018_0908`, `run3_freeze/20261029_1824_ac` | `tw_training_and_freeze_20261018_0908`, `ac_training_and_freeze_20261029_1824` |
| `s2_analysis` | `s2_followup` |

**Tests.** `cd 1_code/extension_study` and `python -m pytest tests -q`. A test that needs absent data skips itself. On
18 November 2026 the suite ran from an unpacked copy of the journal zip, with `runs/` rebuilt as above and the authors'
data paths in `config.py` and the tests: 146 passed, 1 skipped (it needs a Run 1 cycle cache, not stored here), none
failed.

## 7. Integrity records

- `2_preregistration/extension_study/PREREG_EXT.sha256`: the hash of the pre-registration at freezing. Check it with
  `sha256sum PREREG_EXT.md`. The plans of the later experiments have their own `.sha256` files.
- `2_preregistration/extension_study/RUN1_LOCK.json`: the hashes of the Run 1 builder and notebook. The later builders
  read the Run 1 notebook only when its hash matches.
- `2_preregistration/ledger.csv`: every run, scoring and built file with its SHA-256, in time order.
- `run_manifest.json` in each run folder, and `SHA256SUMS` in some: the hashes of the inputs and outputs of that run.
- `FILE_INDEX.csv`: the SHA-256 of every file in this folder.

The repository stores every file byte for byte (`.gitattributes`), so the recorded hashes hold after a clone on any
system.

## 8. What is not here, and why

- **Raw and derived data.** The datasets are public under their own licences. The derived arrays (harmonised records,
  teacher beats, labels, the training pack of about 0.4 GB) are rebuilt by the code; the pre-registration records the
  hashes of the label laws and the pack manifest.
- **Caches and full run archives.** The audio and feature caches (about 3 GB per run) and the archived run folders stay
  with the authors. So does the first-revision campaign run that the integrity gate of Run 1 compared against (about
  7 GB).
- **Run 20261016_1232.** A first attempt at Run 1 on an L4 GPU failed its integrity gate: the R1 feature matrices of
  methods A and B did not reproduce. Its outputs were sealed and not scored; the ledger lists their hashes. Run
  20261016_1516 is the confirmatory Run 1.
- **Smoke tests**, interface tests and planning notes of a follow-up study.
- **Large binary files, in the repository only.** The frozen weights (`*.pt`: three seeds each of T, W and T_ac, and
  `weights/generator.pt`), the Run 2 feature matrices (`run2_20261018_1358/features_*.npz`) and the per-recording network
  traces (`run2_20261018_1358/tw_traces.pkl`). The journal zip leaves them out for size; `FILE_INDEX.csv` lists them
  with their hashes.
- **The first-submission code**, in the repository only. `first_submission_code/` holds the code released with the
  first submission (commit 751a376, 22 August 2026), unchanged. The study code of the revision is `1_code/`. The
  generator checkpoint in `weights/` is the one that methods A and B ran with: Run 1 recorded its SHA-256 (284a5b43...).

## 9. Licences

The Springer segmenter in `1_code/extension_study/extstudy/springer/` is a Python port of the PhysioNet/CinC 2016 sample
entry and carries its GPL-3 licence (`LICENSE_sample2016_GPL3.txt`). The datasets keep their own licences. The rest of the
code carries no licence file at present; for reuse, please contact the corresponding author of the manuscript.
