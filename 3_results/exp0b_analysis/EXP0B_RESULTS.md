# Experiment 0b results: concentrated against spread segmentation failure (21 Oct 2026)

Run `runs/exp0b/20261021_0913` (NVIDIA A100-SXM4-80GB); integrity gate PASSED (arm C's predictions identical to Run 1
on grading, timing and shape, probabilities within 5e-9; repeat fit and worker pool differences 0.0; folds verified).
The executed notebook equals the built one except for one trailing space in a line of `s7_run`. 610 murmur-present
recordings; 60 altered copies of arm C; analysis by `notebooks/exp0b_analysis.py`, run once, under the plan
`EXP0B_PREREG.md` (SHA-256 c63fa439...). Arms C and E from Run 1: composite 53.5 and 43.6 (gain 9.9; grading 5.6,
timing 13.9, shape 10.1). Experiment 0's conditions del010, del020 and del030 enter the same bootstrap draws, and arm
C reproduces Run 1 in both experiments.

Design checks: the two S2 patterns displaced the same amount (pooled share of cycles: concentrated 0.101, 0.211, 0.304
against spread 0.100, 0.199, 0.300; mean share per recording 0.10, 0.20, 0.30 against 0.10, 0.20, 0.30); the 10 ms clip
touched 14% and 13% of displaced cycles; no cycle was too short to move; no recording fell back to equal-time windows;
the S2 shares recompute exactly from the per-recording file.

## Primary result (pre-registered): S2 misplaced by 100 ms

Retention R = (X - E) / (C - E), composite, seed-averaged [95% CI]; Delta = R(spread) - R(concentrated).

| level | R, concentrated (all cycles of this share of recordings) | R, spread (this share of every recording's cycles) | Delta |
|---|---|---|---|
| 10% | 0.78 [0.65, 0.90] | 0.85 [0.73, 0.99] | +0.075 [-0.034, +0.193] |
| 20% | 0.68 [0.55, 0.80] | 0.82 [0.69, 0.94] | +0.139 [+0.020, +0.257] |
| 30% | 0.61 [0.46, 0.73] | 0.78 [0.66, 0.90] | +0.174 [+0.055, +0.306] |
| pooled | | | **+0.129 [+0.047, +0.216]** |

**Decision: concentration costs more.** The 2.5th percentile of the pooled Delta lies above 0. Delta grows with the
level. Per task (pooled): shape +0.21 [+0.05, +0.47], timing +0.10 [-0.01, +0.20], grading +0.07 [-0.21, +0.36]
(grading's gain is 5.6 points, so its retention is noisy, as in Experiment 0).

## Secondary results (estimation only)

**Missed beats.** Half the beats missed in 20, 40 or 60% of recordings retain 0.91, 0.89 and 0.88; the same number
of beats missed evenly (Experiment 0: 10, 20 and 30% of every recording) retain 0.87, 0.90 and 0.78. Pooled Delta
-0.044 [-0.114, +0.017]: no cost of concentration is shown. A missed beat merges into the previous cycle's diastole,
so the S1, systole and S2 windows of every remaining cycle stay where they were.

**Total failure.** Equal-time windows in 10, 20 or 30% of recordings retain 0.70 [0.58, 0.81], 0.66 [0.53, 0.80] and
0.60 [0.47, 0.72]: failing in one recording in ten already costs about 30% of the gain.

## Transfer check (descriptive)

Share of recordings whose median S2 onset lies more than 60 ms from the annotation, against composite retention.

| set | recordings with S2 > 60 ms off | R |
|---|---|---|
| S2 concentrated 10% / 20% / 30% | 0.10 / 0.20 / 0.30 | 0.78 / 0.68 / 0.61 |
| S2 spread 10% / 20% / 30% | 0 / 0 / 0 | 0.85 / 0.82 / 0.78 |
| total failure 10% / 20% / 30% | 0.09 / 0.18 / 0.28 | 0.70 / 0.66 / 0.60 |
| W (Run 2) | 0.09 | 0.70 |
| T (Run 2) | 0.13 | 0.63 |
| Springer, published rule (Run 1) | 0.19 | 0.50 |
| Springer (Run 1) | 0.25 | 0.33 |
| D, exact starts and fixed fractions (Run 1) | 0.25 | 0.86 (timing 0.51) |

W sits on the total-failure curve at its share; T sits a little below it. Springer loses about twice what either
curve gives at its share. Read off the two curves by linear interpolation, the shares alone would give a W minus
Springer retention gap of 0.10 to 0.16, against the 0.37 observed: a quarter to a half of the gap (an exploratory
reading, written after the results; the curves are not measured between 0 and 10% and each point carries an interval
of about +/-0.12). Arm D does not fit on the composite: its misplacement follows heart rate in a regular way, and its
timing retention (0.51) does fit.

## Exploratory, not pre-registered: where the gain is lost

Written after the primary result: `notebooks/exp0b_where_lost.py`. In the concentrated conditions, the net loss rate
relative to arm C (share right under C and wrong under X, minus share wrong under C and right under X), among the
recordings chosen for failure and among the rest, beside the net loss of arm E (equal-time) on the same recordings.

| condition, task | chosen recordings: X | chosen: E | other recordings: X | other: E |
|---|---|---|---|---|
| S2 concentrated, timing (10 / 20 / 30%) | 0.17 / 0.13 / 0.12 | 0.15 / 0.13 / 0.13 | 0.01 / 0.02 / 0.02 | 0.12 / 0.12 / 0.12 |
| total failure, grading (10 / 20 / 30%) | 0.11 / 0.13 / 0.08 | 0.03 / 0.08 / 0.07 | 0.01 / 0.01 / 0.00 | 0.07 / 0.06 / 0.06 |
| missed beats concentrated, timing (20 / 40 / 60%) | 0.02 / 0.02 / 0.03 | 0.12 / 0.09 / 0.12 | 0.02 / 0.02 / 0.01 | 0.12 / 0.15 / 0.12 |

The loss stays in the failed recordings, and there it is as large as with no gating at all (with equal-time windows
among gated training data, larger on grading). On timing and grading the other recordings lose 0.00 to 0.02, the same
as under concentrated missed beats, which do no harm: the general cost of refitting on altered features. On shape they
lose 0.03 to 0.035 at 20 and 30%, a small spill-over.
The extractor summarises each phase over the cycles of a recording, mostly by means: a minority of misplaced cycles
moves a summary part of the way; misplacing every cycle replaces it.

## Reading

- Established, in stylised form: for the same amount of boundary error, failure confined to some recordings costs
  more of the gain than error spread over every recording, and a failed recording loses its whole gain. The quantity
  a segmenter should minimise and report is the share of recordings in which it fails, not its typical boundary
  error. Missed beats that leave the phase windows of the remaining cycles in place cost little either way.
- For the thesis hypothesis: ECG supervision (T, W) fails in 9 to 13% of recordings against Springer's 19 to 25%, and
  this experiment shows that such a difference costs gain by itself. Together they give this project's first causal
  evidence for the route from better R-peak timing (fewer failed recordings) to better murmur characterisation.
- Not established: that failure share explains the whole gap. At the arms' shares the stylised failures account for
  W and T but for only a quarter to a half of Springer's extra loss. The rest needs another cause; the leading
  candidate is that real failures fall on noisy or murmur-dominated recordings, which carry the labels' information,
  whereas here they fall at random.

## Boundaries

CirCor audio, annotations and labels; no model trained or selected for any frozen model. Failures are stylised (a
fixed 100 ms S2 shift; recordings chosen at random). The per-level contrasts at 10% and the per-task contrasts are
estimation only; the decision rests on the pooled composite Delta alone.

## Files

`extension_study/runs/exp0b/20261021_0913/` (run files, SHA256SUMS); `extension_study/runs/exp0b_analysis/`
(`exp0b_results.json`, `exp0b_where_lost.json`); this file.
