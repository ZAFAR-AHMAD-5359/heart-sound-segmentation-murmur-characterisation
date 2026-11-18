# Pre-registration: EHJ-DH extension experiments (Runs 1 and 2)

**Frozen 16 October 2026, before any classification on CirCor in Run 1.** Prepared by Zafar Ahmad with Dr Nazia
Perwaiz (corresponding author) for manuscript EHJ-DH-D-26-00501 (revision due 20 November 2026). The SHA-256 of this
file is stored in `PREREG_EXT.sha256` and embedded in the Run 1 notebook; the notebook's run manifest records it.

At the time of freezing, no arm had been scored on the CirCor labels in this extension. The only CirCor contact was a
smoke test on about 120 recordings that ran stages 0 to 6 (audio, boundaries, features, folds) and stopped before
classification, plus the exposures listed in section 4.

The decisions below come from `08_EXTENSION_DESIGN.md` (15 October 2026, sections 2.1 to 2.9), amended only as listed
in section 6. Where the two differ, this file governs.

---

## 1. Question

Does an automatic phase-boundary method recover the gain that expert-quality boundaries bring to paediatric murmur
characterisation, and does a timing model supervised by ECG on paired adult data do at least as well as the classical
Springer segmenter, with no CirCor annotation in its training?

## 2. Arms

All arms share the same 3,000 CirCor recordings, the same child-grouped folds, the same 464 features, and the same
classifiers. Only the source of the cardiac phase boundaries differs.

| Arm | Boundaries | Role | Run |
|---|---|---|---|
| E | equal-time windows at 75 bpm | control | 1 |
| C | CirCor annotation | reference only, not a comparator in any test | 1 |
| D | annotated S1 onsets with fixed-fraction phases | descriptive | 1 |
| E2 | equal-time windows with the annotated cycle count | descriptive | 1 |
| A | R-peaks of the current synthetic ECG | continuity with R1 | 1 |
| B | synthetic R-peaks refined by the acoustic envelope | continuity with R1 | 1 |
| **S** (S_wpk) | Springer LR-HSMM, published parameters, heart-rate search 40 to 220 bpm at the highest local maximum, systole floor 0.10 s | **primary automatic comparator** | 1 |
| S_pub | Springer with the published heart-rate rule (30 to 120 bpm, global maximum, systole floor 0.20 s) | secondary | 1 |
| S_annot, E_annot | S and E cycles whose S1 onset lies inside an annotated stretch | matched coverage | 1 |
| CNN | PCG-only network, one declared seed (42) | detection reference | 1 |
| **T** | ECG-supervised timing network (section 7) | **primary new arm** | 2 |
| W | same network trained with only an ECG-waveform head, read out by the same decoder | secondary: reconstruction question | 2 |
| T_annot, W_annot | as S_annot | matched coverage | 2 |

A recording with fewer than two cycles from an arm falls back to equal-time windows and stays in the analysis; an
arm with more than 5% of recordings on this fallback is reported as failed.

## 3. Endpoint, inference, and hypotheses

**Primary endpoint.** Composite macro-F1: the mean of the MLP macro-F1 on grading, timing, and shape.

**Classifier.** The R1 MLP configuration (batch 32, AdamW 1e-3, weight decay 1e-4, up to 250 epochs, patience 25,
label smoothing 0.1, inner child-disjoint validation of 15%), refitted in every fold with five seed bases (42, 1042,
2042, 3042, 4042) plus the fold index; the out-of-fold prediction is the argmax of the mean probability over the five
fits. The random forest of R1 runs beside it (secondary).

**Folds.** StratifiedGroupKFold (10 folds, shuffle, random state 42) per task on the R1 labelled sets, grouped by
child: the connected components of Patient ID and Additional ID (816 children in the detection set). The assignment
was drawn once and hashed (section 5).

**Inference.** 2,000 bootstrap draws of the children (seed 42), one set of draws shared by every task and arm. In each
draw, every recording carries the multiplicity of its child; each task's macro-F1 and the composite are recomputed;
differences between arms use the same draws. Intervals are percentile intervals.

**Hypotheses (one-sided, total alpha 0.025, graphical procedure).**

1. H1: S minus E above 0, at 0.0125. Rejected when the 1.25th percentile of the bootstrap distribution of the composite
   difference lies above 0.
2. H2: T minus E above 0, at 0.0125, tested the same way.
3. Whichever of H1 and H2 is rejected passes its 0.0125 to the other, which is then tested at 0.025 (2.5th percentile).
4. H3, only if H2 is rejected: T non-inferior to S with margin 2.5 composite points, at the level at which H2 was
   rejected: the matching lower percentile of T minus S lies above -2.5.
5. H4, only if H3 holds: T minus S above 0 at that level.
6. If T is not frozen by 30 October 2026, 23:59 PKT, H1 is tested alone at 0.025 and T is reported as not completed.

**Three-way verdict for H3.** Non-inferior if the lower bound lies above -2.5; inferior if the upper bound lies below 0;
otherwise inconclusive. No language of trends.

**Estimation without p-values.** Per-task results; random forest; detection (MLP and CNN); W against T; T against A;
T against the CNN; S_pub; matched coverage, with the recovery fraction (X - E) / (C - E) computed only at matched
coverage (S_annot, E_annot, and in Run 2 T_annot, W_annot); heart-rate bands (below 100, 100 to 125, 125 to 150, 150 bpm
or more; infants separately); and a sensitivity analysis without fallback recordings. Statements relative to C come
only from the matched-coverage analysis.

## 4. The CirCor firewall and exposures

- No CirCor audio, annotation, label, offset, threshold, or early-stopping signal enters training or model selection
  for S, T, or W. S runs with published parameters; T and W are trained and selected only on paired adult data.
- No CirCor development set exists: all 816 children are test data.
- One scoring run per arm, logged in an append-only ledger with time stamps and prediction hashes. A crash rerun
  repeats the same code and is logged as such.
- Before T is frozen, one label-free interface test on 10 CirCor recordings is allowed (cycle validity and fallback
  count only; crash fixes only; logged).
- **Exposures to disclose.** Panel E's exploratory Springer pilot compared 219 CirCor recordings with the annotation,
  and the S_wpk rule was devised after the first 20 of them; S_wpk was chosen on the ECG-truth check on paired data.
  Three recordings from that pilot (50781_PV, 85152_PV, 50721_MV) served as unit tests of the Springer code (audio
  only). The smoke test of 16 October read labels only to check the fold files.

## 5. Frozen artefacts

| Artefact | SHA-256 |
|---|---|
| `08_EXTENSION_DESIGN.md` | f3bfe7b984cddcb478412460c010d0b49cd31ed943f7112e3d6de50dbf1afc12 |
| `extension_study/CODE_DESIGN.md` | 2e9fae7ca93a038bfc52973daa99e57dba34d123629ee4ef3e4db3b163157320 |
| CirCor child-grouped folds: detection | 29629ee58636f686b2c97e12c1051a5b4d0d298fb3fcf20b7c5972454deaf7cd |
| grading | 2904efb0ac9947e71828193b97e3309c8388ed35d8c9732f83b1a5ea59e10a4a |
| timing | 2d4731e58e15d813eea6ad82071cc0182c645af7784933d3e8fd5106b96a35f4 |
| shape | 6d65dd9cf2ba0af01c48e6ae5066d4a70b15ba62729373ab34330568a90a7108 |
| fold manifest (`runs/folds_child/folds_manifest.json`) | 8ff55a66110f60b59d914079ef496b1aee6bd5a3106fa9ab66524c9332811a84 |
| R1 notebook base reproduced by the Run 1 builder (source hash) | 8b233013dca2d7277cb73213f61c89f3282efbe286f020e67abe4d328925291e |
| Springer source (`springer_py.py`) | 33bf4c10b8ab6d1041c8eb951317c40400ff7cb735d39c1a6e2e0406991499f7 |
| Springer parameters: B matrix | 457621a433dc5ca6bd9faa80b63fdd427b176eba9ab2c888a803dbbf85ab35ea |
| pi vector | dbcd79495dc89bae1e1c48a1a0f58c0667ab3cb3992a10bd87610b5962a5fed9 |
| total observation distribution | 54949bf43814373b2028e23659619af2e27e4a97f7fbe2a9334996eb7d297fd3 |
| R-peak detector (`robust_beats.py`, Panel H) | 09bb9d7f22105eefd31cd7ae883da95af98e7d05dd8e126c8e20799035d4c077 |
| paired split table | f5ba2893a3dd0a24c72810155cb4b4078537f3b713f8a6d145b4a8c86264ab25 |
| label laws (`derived/labels_v1/laws.json`) | 394e830907073f18e615ed76a90c41d5539ff459b1c6bf4e7306969de2a631fc |
| training pack manifest (`derived/corpus/paired_v1/manifest.json`) | 7164b6e7ef9e37d75bc7f479df5cf97b5094826cfcc6ed1d7d42361cd5966dc2 |

The Run 1 notebook and builder hashes are recorded in `RUN1_LOCK.json` after the notebook is built with this file's
hash inside it.

## 6. Amendments to `08_EXTENSION_DESIGN.md`, made 15 to 16 October 2026 before any CirCor result

Each came from a test on paired or simulated data; none involved CirCor labels.

1. **Teacher R-peaks.** Panel H's frozen detector (energy-centroid reference point). A beat is kept when XQRS or GQRS
   confirms it within 50 ms after each detector's record offset; beats found only by XQRS and GQRS are not added. No
   labels: cycles holding a beat that both wfdb detectors found or a rejected candidate with QRS energy plus one wfdb
   detection (hidden premature beats, as in bigeminy); cycles next to a beat with template correlation below 0.8;
   neighbourhoods where under 80% of beats within plus or minus 5 s are confirmed and correlated; RR above 1.5 times the
   local median; NaN stretches. One lead per dataset (SensSmartTech II, BSSLAB channel 4; another lead only when that
   lead fails). Stress acceptance is a mean F1 per condition; Panel H's own detector did not reach 0.98 on every single
   signal. The S-versus-R lag gap is at most 0.5 ms in every dataset (gate 8 ms).
2. **Acoustic labels.** S1 and S2 are envelope maxima inside windows law(dataset, heart rate) plus or minus 40 ms and
   80 ms, with intercept and slope per dataset fitted by median regression on training subjects. A beat is valid when
   its cycle is labelled and regular and each sound sits off the window edge, at least 2 record noise units above the
   cycle floor, at least a quarter of the record's median height, at least as tall as anything else in its cycle, and
   (S1) not before the QRS reference point; a record whose median S1 lag is under 10 ms carries no labels. Valid
   training beats: 122,929. A hand check of 200 beats (accept at least 95% of marks within 20 ms of the visible sound)
   is pending; if it fails, the label rules are revised and logged before T training starts.
3. **Beat decoder shared by T and W.** Emission 4 log2(amplitude / local beat level) + 2; T's input is its beat head
   minus its S2 head, floored at 0. Suite of simulated traces: median F1 0.992 against the beats that carry a bump.
4. **Run 1 notebook.** Built from the R1 notebook, which its builder reproduces exactly before patching. The R1
   validation gate (reference values from patient-grouped folds and one seed) is replaced by a label-blind integrity
   gate: pinned kept set, embedded folds, zero child overlap, reproduction of the R1 feature matrices, and repeat fits
   that return identical probabilities. The R1 reviewer ablations and figures are not rerun. The Springer Viterbi pass
   runs as plain Python, the code that was checked. The MLP fits run in parallel worker processes on the GPU; a local
   test returned bit-identical probabilities to fits run one at a time, and the integrity gate repeats the check.
5. **Paired splits.** HFrEF published folds (0 to 3 training, 4 validation, community test set as test); other datasets
   70/15/15 by subject; PhysioNet 2016 training-a in training only: 2,095 training, 441 validation, and 471 test records.

## 7. Run 2: T and W

- Input: PCG at 1 kHz, band-pass 25 to 400 Hz, log-power spectrogram (32 ms Hann, 4 ms hop, 24 bins from 31 to 391 Hz),
  each bin's recording median subtracted.
- Network: 1-D U-Net, about 0.9M parameters; T heads S1, S2, R (Gaussian targets, sigma 12, 12, and 8 ms; focal loss
  weights 1, 1, 0.5); W head: the ECG waveform (L1 on the signal and its first difference).
- Training: 12 s crops; AdamW 2e-3, cosine schedule, 40 epochs, batch 64; fp32; three seeds per arm; CirCor traces
  averaged across seeds. Batch composition by dataset share (HFrEF 35%, training-a 15%, EPHNOGRAM 15%, SensSmartTech 12%,
  FOSTER 12%, BSSLAB 11%). Augmentation: interval warping to 70 to 180 bpm, device-response tilts, noise 0 to 20 dB,
  murmur-like systolic noise, frequency masking.
- Selection: beat-identity F1 at tolerance min(50 ms, 10% RR), equal weight over datasets and over native and
  110 to 170 bpm warped validation sets; best state re-scored.
- Freeze: weights hashed by 30 October 2026, 23:59 PKT. Decoder parameters as in amendment 3. Boundaries at 50% of
  the local envelope peak, output in the R1 cycle format.

## 8. Changes after freezing

Any change after this date is added below with its date, reason, and the results already seen at that time. No
change is made after a result it could influence.

| Date | Change | Reason | Results seen at the time |
|---|---|---|---|
| | | | |
