# Run sheet for the two Colab notebooks (EHJ-DH revision R1)

Read this page first. `NOTEBOOK_README.md` has the full detail.

**Status, 1 Oct 2026: both notebooks are final and ready for Colab.** The validation gate of the gating notebook was rebuilt and then tested on this PC on all 3,006 recordings: the sample matched the reference run and all eight cells passed (worst deviation 3.0 points).

## What the two notebooks answer

| Notebook | Reviewer items | Question |
|---|---|---|
| `R1_Gating_Campaign.ipynb` | R3-M1, R3-M5, R1-M1, R1-M2, R3-M4, R1-m8, R3-m3 | Under patient-grouped folds with in-fold scaling, how do six sources of cardiac-cycle boundaries compare on the same 464 features and classifiers? Arms: A synthetic R-peaks with fixed fractions, B synthetic R-peaks fused with acoustic S1 onsets, C PCG segmentation (CirCor annotation files), D PCG-segmentation cycle starts with fixed fractions, E equal-time at 75 bpm, E2 equal-time count-matched. |
| `R1_CNN_Baseline.ipynb` | R1-m5, R3-M1 | The phonocardiogram-only CNN baseline rerun on the same folds, with fold standard deviations, recording-level scores, and paired statistics against every arm. |

## Before you start

1. Upload both `.ipynb` files to Google Drive, open them in Colab Pro, and choose a GPU runtime with High-RAM (A100 first choice, then L4; a T4 works but is slow).
2. Confirm these two things exist on your Drive. The notebooks search for them, print what they found, and stop with a clear message if something is missing.
   * The complete CirCor training set (3,163 `.wav` files with their `.tsv` and `.hea` files). Expected at `MyDrive/JBHI-Revision/October Thesis Dataset/CirCor DigiScope/the-circor-digiscope-phonocardiogram-dataset-1.0.3/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data`.
   * The generator checkpoint `trail4_best.pt`. Expected at `MyDrive/October Thesis Dataset/trail4_checkpoints/trail4_best.pt`.
3. If either sits somewhere else, set `MANUAL_TRAINING_DATA` or `MANUAL_CHECKPOINT` in the override cell (the third code cell). Do not move files.

## Order and time

| Step | Notebook | Flags | Time on A100 | What to check |
|---|---|---|---|---|
| 1 | Gating | `DRY_RUN = True` | about 10 min | Runs every stage on 120 recordings. All cells finish; eight figures appear. |
| 2 | Gating | `DRY_RUN = False`, `RESUME_DIR = None` | 2.5 to 4.5 h | Stage 0 prints the dataset path, 3,163 pairs, and the checkpoint hash. Stage 8 prints the validation gate result. |
| 3 | CNN | `DRY_RUN = True` | about 10 min | It binds the gating dry-run folder and reaches the manifest cell. |
| 4 | CNN | `DRY_RUN = False` | 3 to 4 h | It binds the full gating run folder (not a `_dry` one), prints 0 label disagreements, and reaches the manifest cell. |

Steps 1 and 3 are optional but cheap; they catch a path or package problem before the long runs.

For step 4, set `RUN_DIR` in the CNN notebook to the full gating run folder from step 2 (the path printed at the start of that run). With `RUN_DIR = None` the notebook picks the newest gating run, which is fine the first time but could bind a different folder if you make another gating run later.

Figures are saved into the run folder and are not shown inline.

Free Drive space needed: about 6 GB (about 2 GB for the gating run with its caches, about 4 GB for the CNN run).

## If the session disconnects

Nothing is lost. In the gating notebook set `RESUME_DIR` to the run folder that was printed at the start (for example `.../R1_Gating_Campaign/20261003_0915`) and run all cells again; finished stages load from cache. The CNN notebook resumes on its own when `RUN_DIR` points at the same gating run folder: choose Run all again. Finished folds load from cache; the scoring, continuity, and merge sections are recomputed each time (the continuity section retrains four models, about 30 min).

## The validation gate (gating notebook, Stage 8)

The gate refits arms C and E with the recipe of the earlier clean experiment and compares them with its reference values (macro-F1 79.2 / 58.3 / 49.0 / 46.6 for C and 69.6 / 51.0 / 39.5 / 39.0 for E). A cell agrees when it is within 3 points of the reference, or when the reference lies inside its 95% bootstrap interval. Three outcomes are possible.

* **PASSED**: the code reproduces the earlier experiment. Use the results.
* **PASSED ON THE REFERENCE-RULE SET**: arms A or B dropped a few recordings, so the kept set is not the earlier sample; the reproduction holds on the earlier sample itself. Use the results and report the kept-set size that the message prints.
* **FAILED**: the notebook prints a banner, skips the figures, and still writes the manifest. Send me the gate cell output and `run_manifest.json`. Do not use the numbers.

A failure on Colab does not have to mean a fault. The gate refits small networks, and a GPU draws different random dropout masks than the PC test did, so one detection cell can land outside the rule by chance. In that case do not rerun: every result is already saved in the run folder, and the figures can be drawn afterwards from the cache in a few minutes once I have looked at the gate output.

## What to send back

From the run folder on Drive (`MyDrive/JBHI-Revision/R1_Gating_Campaign/<date_time>/`):

* `results.json`, `run_manifest.json`, `diagnostics.csv`, `oof_predictions.npz`
* the `figures` folder (F1 to F8, PDF and PNG)
* the `cnn_baseline` folder: `results.json`, `paired_vs_gated.json`, `run_manifest.json`, `figures` (B1 to B4)

A zip of the run folder without its `cache` subfolders is the easiest way.

## Things that are expected, not errors

* Arm E2 shows a low implied heart rate. It inherits the annotated cycle count, and the annotation covers only part of many recordings; the notebook prints the coverage.
* A few recordings are dropped per arm (too few cycles). The kept set is the intersection across arms and is printed.
* Small strata in the subgroup figure show "n<10" instead of a bar.
* The dry-run numbers mean nothing; they exist to test the plumbing.

## What was verified locally before hand-over

* Both notebooks are generated from builder scripts and match them cell for cell; every code cell compiles; no dashes or banned words in the text cells.
* The gating notebook ran every stage on a 120-recording subset on this PC (63 min on CPU) and resumed correctly after cache files were deliberately damaged.
* The CNN notebook ran its dry run on a fixture and on the real gating dry-run output, fresh and resumed, with 0 label disagreements across 84 arm-task groups.
* Not testable on this PC: GPU paths, the forked worker pool, the Drive search, and a passing validation gate. Watch Stage 0 and Stage 8 on the first Colab run.
