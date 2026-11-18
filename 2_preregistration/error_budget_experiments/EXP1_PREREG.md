# Experiment 1, in-band interference at test time: pre-registration

Written 29 October 2026, after the main comparison (Runs 1 and 2), the error-budget experiments (0, 0b, 0c), the
revision checks, and the independent code review, and before any interference-corrupted audio, boundary set, feature
or score of this experiment exists. Code: `extension_study/extstudy/noise.py`, `extension_study/notebooks/build_exp1.py`
(patches over the locked Run 1 notebook, as Runs 2 and Experiments 0 to 0c), `exp1_analysis.py`.

## Question

The first submission's rationale for cardiac-phase gating was that a murmur is phase-locked to the cardiac cycle and
recurs on every beat, whereas interference in the same frequency band is not. The revised manuscript states this as a
rationale and says that no experiment measured how much of the gain survives added in-band interference. This
experiment measures it: when interference confined to the murmur band and not locked to the cycle is added to the
recordings, how much of the gain of cardiac-phase gating over equal-time windows survives, (a) when only the
segmenter sees the interference and (b) when both the segmenter and the feature extractor see it?

## Known before writing

- Composite macro-F1 (MLP; grading, timing, shape; 610 murmur-present recordings): E 43.6, Springer 46.9, T 49.9,
  W 50.5, C 53.5; T minus E +6.3 [3.4, 9.3]; T minus S +3.0 [0.3, 5.7]; recovery at matched coverage T 0.63, W 0.68,
  S 0.31. Error budget: beat-level error is tolerated, recording-level failure is not; S2 failure shares S 25%, T 13%,
  W 9%.
- T and W were trained with augmentation that included additive noise at 0 to 20 dB SNR and murmur-like systolic
  noise; the Springer segmenter was not. The comparison is therefore of the deployed segmenters as they are, not of
  equally prepared methods, and the plan says so.
- An exploratory stratification of the existing out-of-fold predictions by label-free noise proxies (29 October, code
  review) found T minus E positive in every tertile; it is descriptive and is not part of this experiment.
- No interference has been added to any recording; no number of this experiment exists.

## Data and pipeline

The 610 recordings labelled for grading, timing or shape (the Run 1 kept set restricted to those tasks, as in
Experiments 0 to 0c); the shared analysed audio of Run 1 (4 kHz, unit peak); Run 1's 464-feature extractor, five-seed
MLP (primary) and random forest on the embedded child-grouped folds; the frozen T and W networks of Run 2 (checkpoint
folder 20261018_0908, every hash checked); the Springer segmenter at its primary setting (40 to 220 beats per minute,
published parameters). Arms C (expert-reviewed annotation) and E (equal-time windows) come from the annotation and
the recording length and need no audio.

Five reproduction controls run on the clean audio and must reproduce their runs (MLP predictions identical,
probabilities within 1e-5, random forest identical, on the three tasks): C and E and Springer against Run 1, T and W
against Run 2. They are part of the integrity gate; a failed gate means the run is not used.

## Interference (fixed now)

All interference is confined to the murmur band, 20 to 500 Hz (fourth-order Butterworth band-pass, zero phase), and
none of it is locked to the cardiac cycle. Three kinds:

1. **gauss**: stationary Gaussian noise band-passed to 20 to 500 Hz (the premise's own model of interference).
2. **burst**: band-limited bursts 0.2 to 0.5 s long with 20 ms raised-cosine edges, two per 10 s of audio on average,
   at uniformly random non-overlapping onsets, silence elsewhere (handling or friction).
3. **esc50**: recorded environmental sounds from the ESC-50 corpus, classes crying_baby, breathing and coughing (120
   five-second clips), resampled to 4 kHz, band-passed to 20 to 500 Hz, concatenated with 50 ms crossfades to cover the
   recording. The corpus is fetched by the notebook (GitHub archive of karolpiczak/ESC-50, master); the SHA-256 of
   every clip used and of the clip bank are written to the run manifest. If the corpus cannot be fetched, the notebook
   records this, runs the gauss and burst conditions, and reports the esc50 conditions as not run.

Two levels: 10 dB and 0 dB. SNR is the clean in-band RMS over the analysed portion divided by the interference's
in-band RMS over the same portion; for bursts, the interference RMS is taken inside the bursts only. After mixing, the
signal is re-normalised to unit peak, as the loader normalises every recording. Six conditions: gauss10, gauss0,
burst10, burst0, esc5010, esc500. Condition seed: 300000 + 1000 x kind index (gauss 0, burst 1, esc50 2) + level
index (10 dB 1, 0 dB 2); draws within a recording come from (condition seed, CRC-32 of its name).

## Arms (eight per condition, 48 in all, plus the five controls)

For each condition c:
- **Segmentation only** (`S_seg_c`, `T_seg_c`, `W_seg_c`): Springer, T and W segment the corrupted audio; the
  features are read from the clean audio with those boundaries. This isolates the segmenter. T's cycles come from its
  S1 and S2 heads on the corrupted audio; W's synthetic ECG is decoded from the corrupted audio and its phases follow
  arm B's acoustic construction on the corrupted audio. Fewer than two cycles: equal-time fallback, counted; the
  fallback share is reported per arm and condition.
- **Both stages** (`E_both_c`, `C_both_c`, `S_both_c`, `T_both_c`, `W_both_c`): the same boundaries as above for S, T
  and W (and the annotation for C, the equal-time windows for E), with the features read from the corrupted audio.
  `C_both_c` minus `E_both_c` is the gain of perfect boundaries under corrupted features; `E_both_c` and `C_both_c`
  measure the extractor's own sensitivity.

Every arm goes through the same features, classifier and folds. One scoring per arm.

## Measures and decision, fixed now

Composite macro-F1 (MLP; grading, timing, shape) with the shared child bootstrap (2,000 draws, seed 42), every arm and
control in the same draws (`extstudy/stats.py`, `analyse`). X denotes the clean control of the same arm (this run's
reproduction control), E the clean equal-time control.

- **Confirmatory, one hypothesis, one-sided, alpha 0.025.** H1: under gauss at 0 dB, the timing network's boundaries
  from the corrupted audio still beat equal-time windows on clean features: `T_seg_gauss0` minus E above 0, rejected
  when the 2.5th percentile of the bootstrap difference lies above 0. This is the premise's own interference at the
  harsher level, with the extractor held clean so that only segmentation is tested.
- **Estimation (no p values), for every condition c and arm X in {S, T, W}:**
  - the composite of `X_seg_c` with its interval, and the change from clean, `X_seg_c` minus X;
  - retention of the clean gain under corrupted segmentation, R_seg = (`X_seg_c` minus E) / (X minus E), computed per
    draw; draws with a non-positive denominator are excluded and counted;
  - the gain over equal-time windows under corrupted segmentation, `X_seg_c` minus E, with its 2.5th percentile (read
    as "the gain holds at this level" when the percentile lies above 0; descriptive for every arm and condition other
    than H1);
  - `T_seg_c` minus `S_seg_c`, the timing network against Springer under the same interference;
  - both stages: `X_both_c` minus `E_both_c`, and its change from the clean gain, (`X_both_c` minus `E_both_c`) minus
    (X minus E); `C_both_c` minus `E_both_c`; `E_both_c` minus E and `C_both_c` minus C (the extractor's sensitivity).
- **Diagnostics, descriptive:** per condition and arm, the fallback share, the median cycle-start F1 against arm C,
  the share of recordings whose median S2 lies more than 60 ms from the annotation (the error-budget failure
  criterion), and the achieved SNR distribution; these are read beside the error-budget curves of Experiments 0 to 0c.
- Every condition and arm is reported; no condition, level, seed or rule changes after a result is seen. The esc50
  conditions, if not run, are reported as not run.
- Precision: single contrasts of this size carried 95% intervals of about +/-3 composite points in Run 2 and
  retention intervals of about +/-0.25 in Experiment 0c, so changes smaller than that may not be shown.

## What each outcome means

- H1 rejected and retention high for T and W: the gating gain survives stationary in-band interference at the level
  of the heart sounds themselves when the segmenter is exposed; the premise holds for segmentation. The both-stage
  arms then show how much of the remaining loss is the extractor's.
- H1 not rejected: the premise is not supported at 0 dB for the timing network; the 10 dB level and the retention
  values bound where it fails.
- Springer and the networks diverging under interference: the networks' noise augmentation, or their training corpus,
  gives them an advantage the comparison cannot separate from their design; the plan says so.
- The both-stage arms falling to equal-time windows while the segmentation-only arms hold: the loss under
  interference lies in the features, not the boundaries.

## Boundaries of the claim

Draws on CirCor audio, annotations and labels; trains and selects nothing for any frozen model; no CirCor label or
annotation enters any segmenter. The interference is synthetic or borrowed from another corpus and added to field
recordings that already carry their own ambient noise, so the levels are relative to the recorded signal, not to a
quiet room. The experiment tests gating under interference in the frequency band of the murmur; it does not test
diagnoses, and it cannot attribute a difference between the networks and Springer to the electrocardiogram
supervision itself (no acoustic-label control is part of this experiment).
