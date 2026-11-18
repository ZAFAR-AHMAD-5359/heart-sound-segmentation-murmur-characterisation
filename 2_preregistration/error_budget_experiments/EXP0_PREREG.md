# Experiment 0, the beat-error budget: pre-registration

Written 18 October 2026, before any corrupted boundary set is built or scored. Study concept: `STUDY_CONCEPT.md`,
section 3. Code: `extension_study/extstudy/corrupt.py` and `extension_study/notebooks/build_exp0.py`.

## Question

How much beat-level error can a cycle segmentation carry before the gain of accurate gating over equal-time windows
(arm C minus arm E) is lost on murmur grading, timing and shape? The answer fixes the beat-accuracy target for any new
synthetic-ECG generator (Gate 1 of the concept) and gives the reference curve on which the existing arms (A, B, S,
and Run 2's T and W) are placed: if their points fall on the curve, boundary accuracy is the variable that carries
the gain.

## Data and pipeline

- The Run 1 analysed audio of CirCor, the Run 1 kept set restricted to the recordings labelled for grading, timing or
  shape (the murmur-present recordings), arm C's annotated cycles, the embedded child-grouped folds, the 464-feature
  extractor, and Run 1's five-seed MLP (primary) and random forest (secondary), all unchanged.
- Arm C itself is rerun as a reproduction control: its out-of-fold predictions must equal Run 1's on all three tasks
  (MLP predictions identical, probabilities within 1e-5). Arms C and E for the comparisons come from Run 1.
- Detection is not part of Experiment 0.

## Conditions (one factor at a time, five corruption seeds each: 60 boundary sets)

| error | levels | rule |
|---|---|---|
| missed beats | 10, 20, 30, 50% of annotated starts | the start's cycle merges into an abutting predecessor (a whole beat lies in "diastole"); with no abutting predecessor the cycle is removed |
| spurious beats | 10, 20, 30, 50% of annotated starts | one false start per chosen cycle, uniform in 10 to 90% of its length; that cycle becomes two cycles with arm D's fixed fractions |
| timing jitter | SD 5, 10, 20, 40 ms | R1's `jitter_cycles`: each cycle shifted rigidly by a Gaussian offset |

Seeds: condition seed 10000 x type (1 missed, 2 spurious, 3 jitter) + 100 x level index + s, s = 1 to 5; within a
condition each recording draws from (condition seed, CRC-32 of its name).

## Measures

- Boundary quality of every set against arm C's starts (Hungarian matching within 150 ms): boundary F1, insertion
  and deletion rates, signed-offset SD; median over recordings.
- Primary: composite macro-F1, the mean of the MLP macro-F1 on grading, timing and shape.
- Retention: R = (X - E) / (C - E), per task and for the composite, with C and E from Run 1.
- Inference: the shared child bootstrap of the extension (2,000 draws, seed 42, `extstudy/stats.py` analyse);
  per level, the composite averaged over the five seeds within each draw; 95% percentile intervals.

## Read-off, fixed now

1. Per error type, levels in increasing order: the threshold is the largest level whose seed-averaged composite
   retention (point estimate) is at least 0.80, scanning upwards and stopping at the first level below 0.80.
2. Gate 1 boundary target: the larger of the median boundary F1 values at the missed-beat and spurious-beat
   thresholds (the stricter requirement).
3. Target for timing error: the largest jitter SD whose seed-averaged retention is at least 0.80 on every one of the three tasks.
4. Every curve is reported, whatever it shows; no level, seed or rule is changed after a result is seen.

## Boundaries of the claim

Experiment 0 draws on CirCor audio, annotations and labels; it trains nothing and selects nothing for Run 2 (T and W are
frozen, Run 2's analysis is fixed), so it cannot alter Run 2. The corruptions are stylised: real segmenters make
correlated errors (runs of missed beats, errors concentrated in noisy stretches), so the curve is a reference, not a
prediction for any one method; the arms' own points test whether it transfers.
