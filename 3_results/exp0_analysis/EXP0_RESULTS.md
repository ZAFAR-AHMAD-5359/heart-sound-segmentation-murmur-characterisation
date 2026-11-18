# Experiment 0 results: the beat-error budget (18 to 19 Oct 2026)

Run `runs/exp0/20261018_1659` (NVIDIA A100-SXM4-80GB, about 2.8 h); integrity gate PASSED (arm C's predictions
identical to Run 1 on grading, timing and shape; repeat fit and worker pool differences 0.0; folds verified). 610
murmur-present recordings, 160 children; 60 corrupted copies of arm C; analysis by `notebooks/exp0_analysis.py`
under the plan `Next Study - Beat-Accurate Synthetic ECG/EXP0_PREREG.md` (SHA-256 82eb31f1...). Arms C and E from
Run 1: composite 53.5 and 43.6 (gain 9.9; grading 5.6, timing 13.9, shape 10.1).

## Composite retention, (X - E) / (C - E), seed-averaged [95% CI]

| level | missed beats | spurious beats | jitter SD | retention (missed / spurious / jitter) |
|---|---|---|---|---|
| 1 | 10% (F1 0.947) | 10% (F1 0.952) | 5 ms | 0.87 [0.77, 1.00] / 0.81 [0.68, 0.95] / 0.91 [0.79, 1.06] |
| 2 | 20% (F1 0.889) | 20% (F1 0.909) | 10 ms | 0.90 [0.78, 1.01] / 0.82 [0.70, 0.94] / 0.83 [0.72, 0.96] |
| 3 | 30% (F1 0.824) | 30% (F1 0.870) | 20 ms | 0.78 [0.64, 0.90] / 0.72 [0.58, 0.87] / 0.88 [0.75, 1.04] |
| 4 | 50% (F1 0.667) | 50% (F1 0.800) | 40 ms | 0.79 [0.67, 0.93] / 0.66 [0.53, 0.80] / 0.75 [0.62, 0.91] |

Per task, timing is the sensitive one (retention 0.83 to 0.65 over missed beats, 0.73 to 0.52 over spurious beats,
0.84, 0.74, 0.78, 0.61 over jitter); shape stays near 1.0; grading's gain is small (5.6 points) and its intervals
span about 0.1 to 1.3, so grading carries little information about the budget. Points that rise between levels
(missed 20% above 10%, jitter 20 ms above 10 ms) show a noise level of about 0.1 in retention.

## Pre-registered read-off

- Missed beats: threshold 20%. Spurious beats: threshold 20%.
- Gate 1 boundary target: boundary F1 at least 0.91 (the larger of 0.889 and 0.909).
- Jitter: no level qualifies under the every-task rule: grading's retention at 5 ms is 0.7996. On the
  composite, retention stays at or above 0.80 up to 20 ms and falls to 0.75 at 40 ms; timing alone keeps 0.84 at
  5 ms and 0.74 to 0.78 at 10 to 20 ms. The rule was fixed in advance and is reported as it came out.

## The arms on the curve (descriptive)

Boundary agreement with the annotation inside the annotated stretches (matched coverage; 610 recordings), and the
recovery fraction of Run 2 (composite; T, W and S against E_annot):

| arm | boundary F1 median | median start offset | recovery fraction |
|---|---|---|---|
| S_annot (Springer) | 0.97 | 20 ms | 0.31 [0.06, 0.52] |
| W_annot | 0.96 | 18 ms | 0.68 [0.46, 0.92] |
| T_annot | 0.92 | 18 ms | 0.63 [0.38, 0.87] |
| D (annotated starts, fixed phase fractions) | 1.00 | 0 ms | 0.86 (against E) |

Over whole recordings the agreement is lower for every automatic arm (insertions about 0.43 to 0.46), because the
annotation covers a median 58% of the analysed portion (IQR 31 to 88%); that view does not measure segmentation error.

## Reading

- Every automatic arm already meets the beat-level target inside the annotated stretches, and its start offsets
  (18 to 20 ms) lie inside the jitter range that kept the composite at 0.88. At that accuracy the curve predicts a
  recovery of about 0.8 to 0.9; Springer recovers 0.31 and T and W about two thirds.
- Beat detection is not what limits the gain at present accuracy. The shortfall lies inside the cycle:
  where S2 and the systolic window are placed (arm D, with exact starts and fixed fractions, already loses 14% of the
  gain), and in errors concentrated in murmur recordings, which random corruption does not mimic. The timing task,
  most tied to the S2 boundary, is also the one most sensitive to every corruption.
- Springer's agreement with the annotation is partly built in, since the CirCor annotations started from automatic
  segmentation that experts reviewed.
- Next analysis: S2 onset and systole-length agreement of S, T and W with the annotation, and its relation to the
  per-recording gain; for generator v2, an ECG reference for S2 (end of the T wave) ahead of a better R-peak.

## Addendum, 19 Oct 2026: the follow-up analysis revises the reading above

The exploratory follow-up (`runs/s2_analysis/S2_ANALYSIS_RESULTS.md`) does not support "the shortfall lies in where S2
is typically placed": Springer places S2 and the phase windows as well as T and W or better. What tracks the
recovered gain among the automatic arms is how many recordings the segmentation fails in (S2 more than 60 ms off: S
25%, S_pub 19%, T 13%, W 9%). Experiment 0 corrupted every recording at the same rate and could not test this; a
pre-registered Experiment 0b with failures concentrated in a share of recordings is proposed.
