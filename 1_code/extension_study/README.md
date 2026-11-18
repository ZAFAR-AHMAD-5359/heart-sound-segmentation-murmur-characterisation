# Extension study: code

Code for `../08_EXTENSION_DESIGN.md`; the module plan is in `CODE_DESIGN.md`.

## Status, 15 October 2026

| Piece | State |
|---|---|
| `extstudy/config.py` | done: local and Drive paths, constants, freeze dates, excluded duplicates, fixed leads, teacher R-peak rule |
| `extstudy/splits.py` | done: CirCor children (Patient ID with Additional ID), child-grouped folds on the R1 sets, paired-data split table |
| `runs/folds_child/` | done: detection 3,000 recordings in 816 children; grading 160, timing 159, shape 158 children; 0 linked pairs split; hashes in `folds_manifest.json` |
| `pcgecg` data fixes | done: training-a NaN samples interpolated (45 records); community-test ejection fraction looked up by subject; SensSmartTech and EPHNOGRAM now read from `Datasets/Dataset-5` and `Dataset-6` (byte-identical to the first audit's copies) |
| `extstudy/robust_beats.py`, `qrs.py` | done: Panel H's frozen detector plus the teacher rule (section 6 of `CODE_DESIGN.md`) |
| `extstudy/relabel.py` | done: all 3,404 paired records relabelled; 3,007 usable, 53.1 labelled hours, 273,795 beats (first audit: 2,433, 43.6 h, 218,444); S-versus-R lag gap at most 0.5 ms everywhere; outputs in `../derived/beats_v2/`, summary in `../07_DATA_AUDIT.md` section 9 |
| `extstudy/labels.py`, `label_check.py`, `refcheck.py` | labels v3 adopted for training (18 Oct; 69,176 training beats, pack `paired_v3`). v1 and v2 failed Claude's visual checks; the label gate logged before the comparison rests on the hand-corrected PhysioNet/CinC 2016 training-a annotations: v3 S1 99.98%, S2 99.88% of marks on the annotated sound (5,805 beats). Claude's verdicts did not track that reference (kappa 0.09), so its v3 visual check (187/200) is reported and does not gate. Trail: `prereg/CHANGES_AFTER_FREEZE.md` |
| `extstudy/frontend.py` | done: band-pass 25 to 400 Hz at 1 kHz, 24-bin log spectrogram at 250 Hz, recording medians; numpy and torch versions agree |
| `extstudy/corpus.py` | done: pack `../derived/corpus/paired_v1/` (2,908 records, 57.5 h, 412 MB, SHA-256 manifest); upload this folder to `MyDrive/JBHI-Revision/Extension_Study/corpus/paired_v1/` |
| `extstudy/decoder.py` | beat decoder shared by T and W done (amplitude emission, two-pass Viterbi); S2 search and cycle export next |
| tests | 76 passing (`python -m pytest tests -q`, about 2 minutes) |
| `notebooks/build_run1.py`, `build_cnn_run1.py` | done: `Extension_Run1.ipynb` and `Extension_Run1_CNN.ipynb`, patched from the R1 builders (each checked to reproduce its R1 notebook first): child-grouped embedded folds, pinned 3,000 recordings, Springer S_wpk and S_pub, matched-coverage copies, five-seed MLP ensembles, integrity gate; a dry run stops before classification |
| Run 1 results | done: `runs/run1_stats/RUN1_RESULTS.md` (H1 rejected at 0.0125; CNN scored) |
| `extstudy/model.py`, `losses.py`, `augment.py`, `train.py` | done: U-Net (872,387 parameters), focal and waveform losses, interval warping with exact heart-sound windows, sampler with dataset shares, validation and selection, deterministic parallel batch stream |
| ECG targets for W | done: `ecg_250_int16.npy` beside the pack (pack files unchanged; logged) |
| `notebooks/build_train.py` | done: `Extension_Train_TW.ipynb` (Colab, paired data only); real training blocked until the 200-beat label check passes |
| `extstudy/infer.py`, `cycles.py`, `notebooks/build_run2.py` | done: frozen T and W (18 Oct, run 20261018_0908) applied label-free; cycles from T's heads and from W through arm B; `Extension_Run2.ipynb` reproduces Run 1 (cycle hashes checked locally) and scores T, W, T_annot, W_annot once; `merge_run2.py` then the frozen stats.py (H2 to H4). Ready to run on an A100 |
| `extstudy/corrupt.py`, `notebooks/build_exp0.py` | done: Experiment 0 (beat-error budget; plan in the next-study folder); `Experiment0_BeatErrorBudget.ipynb` and `exp0_analysis.py`. Ready to run on an A100 after Run 2 |
| everything else | Run 2 CirCor notebook next |

## Run

```
cd "E:\NUST MS Thesis\PhD Level Research\extension_study"
python -m pytest tests -q          # all tests
python -m extstudy.splits          # regenerate the child-grouped fold files and their manifest
python -m extstudy.relabel         # teacher R-peaks for every paired record (about 11 minutes on 14 workers)
python -m extstudy.labels          # S1/S2 labels and heart-rate laws (about 5 minutes)
python -m extstudy.label_check     # 24 pages of 10 panels: 200 labelled beats plus 40 blinded controls
python -m extstudy.corpus          # training pack with hash manifest (about 1 minute)
python notebooks/build_run1.py     # Extension_Run1.ipynb (Colab)
python notebooks/build_cnn_run1.py # Extension_Run1_CNN.ipynb (Colab, after Run 1 has written its folds)
```

## Follow-up experiments (pre-registered 29 October 2026; run sheet `Next Study - Beat-Accurate Synthetic ECG/RUN_SHEET_EXP1_EXP2.md`)

```
python notebooks/build_exp1.py                 # Experiment1_InBandNoise.ipynb (Colab): added in-band interference at test time, EXP1_PREREG.md
python notebooks/exp1_analysis.py <run1 oof> <run2 oof> <exp1 run folder> runs/exp1_analysis   # H1 and the estimation contrasts, locally (done 29 Oct: runs/exp1/20261029_1331, EXP1_RESULTS.md)
python -m extstudy.labels_acoustic --version labels_acoustic_v1 --pack paired_v3 --workers 12   # PCG-only S1/S2 labels (Springer), about 20 minutes
python notebooks/build_train_ac.py             # Extension_Train_AC.ipynb (Colab): the acoustic-label control arm A, EXP2_PREREG.md
python notebooks/build_run3.py <stamp>_ac       # Extension_Run3.ipynb (Colab): T_ac on CirCor, after the freeze record is downloaded
python notebooks/exp2_analysis.py <run1 oof> <run2 oof> <run3 run folder> <out>   # H1 (T minus T_ac) and the estimation contrasts
```

Experiment 1 corrupts the 610 murmur-present recordings with band-limited Gaussian noise, bursts, and ESC-50 crying, breathing and coughing at 10 and 0 dB in-band SNR (20 to 500 Hz), scores segmentation-only arms (S, T, W: corrupted boundaries, clean features) and both-stage arms (E, C, S, T, W), and keeps five clean controls that must reproduce Runs 1 and 2. Experiment 2 trains the T architecture with S1 and S2 heads on labels derived from the phonocardiogram alone and scores it on CirCor as Run 2 scored T; its label gate is reported, not required. Both plans were hashed before any corrupted audio, acoustic label or score existed; see `prereg/CHANGES_AFTER_FREEZE.md`.

## Drive layout for Colab

The R1 notebooks fixed the existing locations; this study adds one folder.

| Item | Drive path |
|---|---|
| CirCor training data | `MyDrive/JBHI-Revision/October Thesis Dataset/CirCor DigiScope/.../training_data` (found automatically, as in R1) |
| Current generator | `MyDrive/October Thesis Dataset/trail4_checkpoints/trail4_best.pt` |
| R1 runs | `MyDrive/JBHI-Revision/R1_Gating_Campaign/` |
| **This study** | `MyDrive/JBHI-Revision/Extension_Study/` with `corpus/`, `checkpoints/`, `runs/`, and optionally `paired_datasets/` |

**About today's upload.** Training does not need the raw paired datasets on Drive. The training pack (PCG at
1 kHz, labels, masks, splits, about 0.4 GB) is built locally from the raw files and uploaded to
`Extension_Study/corpus/`. A copy of the raw datasets in `Extension_Study/paired_datasets/` is a backup
only.
