# Experiment 0b, concentrated against spread segmentation failure: pre-registration

Written 19 October 2026, before any Experiment 0b boundary set is built or scored. It tests causally a pattern the
exploratory follow-up to Experiment 0 found after the fact (`S2_ANALYSIS_RESULTS.md`): among the automatic arms, the
share of recordings in which S2 is placed more than 60 ms off (Springer 25%, Springer published rule 19%, T 13%,
W 9%) orders the gain recovered (0.33, 0.50, 0.63, 0.70); the typical boundary error does not.
Code: `extension_study/extstudy/corrupt.py`, `extension_study/notebooks/build_exp0b.py`, `exp0b_analysis.py`.

## Question

For the same amount of boundary error, does concentrating it in a share of recordings (the rest exact) cost more of
the gain of accurate gating than spreading it evenly over every recording?

## Data and pipeline

As Experiment 0: the Run 1 analysed audio; the 610 recordings labelled for grading, timing or shape; arm C's
annotated cycles; embedded child-grouped folds; Run 1's 464-feature extractor and five-seed MLP (primary) and random
forest; arm C rerun as a reproduction control (predictions identical to Run 1, probabilities within 1e-5). Arms C and
E for retention come from Run 1.

## Conditions (five seeds each; 60 boundary sets)

**Primary, S2 misplacement (D = 100 ms, the typical size of Springer's failures).** A displaced cycle keeps its S1
window and its end; its S2 window (same length) moves by D in the recording's direction (sign drawn once per
recording), clipped to keep 10 ms of systole and of diastole; a cycle too short to move stays exact and is counted.
- concentrated: in a random 10, 20 or 30% of recordings every cycle is displaced; all other recordings exact;
- spread: in every recording a random 10, 20 or 30% of its cycles are displaced.
The two patterns displace the same share of cycles (to rounding, reported) with the same displacement; only its
distribution over recordings differs.

**Secondary, missed beats.** Concentrated: in a random 20, 40 or 60% of recordings half the beats are missed (the
Experiment 0 rule); spread: every recording misses 10, 20 or 30% (Experiment 0's conditions del010, del020, del030,
reused as they stand, which the arm C reproduction in both runs makes comparable).

**Secondary, total failure.** In a random 10, 20 or 30% of recordings the cycles are replaced by equal-time windows.

Seeds: condition seed 10000 x type (4 S2 concentrated, 5 S2 spread, 6 missed concentrated, 7 total failure) + 100 x
level index + s, s = 1 to 5. Recordings chosen with that seed over the sorted recording list; draws within a recording
from (condition seed, CRC-32 of its name).

## Measures and decision, fixed now

- Retention R = (X - E) / (C - E) on the composite (MLP; grading, timing, shape), seed-averaged per condition; the
  shared child bootstrap (2,000 draws, seed 42), Experiment 0's conditions included in the same draws.
- Primary statistic: Delta = R(spread) - R(concentrated) for S2 misplacement, averaged over the three levels; per
  level and per task reported. Decision: concentration costs more if the 2.5th percentile of the pooled Delta lies
  above 0 (one-sided, 0.025); it costs less if the 97.5th percentile lies below 0; otherwise no difference is shown.
- Secondary (estimation only): the same Delta for missed beats; the retention curve of total failure against the
  share of failing recordings.
- Transfer check (descriptive): for each condition, the share of recordings whose median S2 error exceeds 60 ms,
  set beside the arms' shares and retentions.
- Every condition is reported; no level, seed or rule changes after a result is seen.

## What each outcome means

- Concentration costs more: recording-level failure drives the lost gain over and above the amount of error, and a
  segmenter's failure rate across recordings is the quantity to minimise and report.
- No difference: the amount of error, not its distribution, matters; the ordering of the arms has another cause.
- Concentration costs less: errors spread over every recording do the most harm.

## Boundaries of the claim

Draws on CirCor audio, annotations and labels; trains and selects nothing for any frozen model. The failures are
stylised (a fixed 100 ms S2 shift; beats missed at random); real failures cluster in noisy or murmur-dominated
recordings, which this design does not reproduce.
