# Extension Run 1: results

**16 October 2026.** Run folder `runs/run1/20261016_1516` (Colab, NVIDIA A100-SXM4-80GB, 2.0 h). Analysed once with
the frozen `extstudy/stats.py` (SHA-256 b44d78a8...), as pre-registered in `prereg/PREREG_EXT.md` (755b47da...).
Machine-readable output: `stats_results.json`; every file hash is in `runs/ledger.csv`.

## Provenance and integrity

- The run carries the pre-registration hash, and its notebook source equals the locked build (`RUN1_LOCK.json`).
- Integrity gate passed on every check: the kept set equals the 3,000 R1 recordings; all four tasks used the embedded
  child-grouped folds (816 children, no child on both sides of a split); all six R1 arms reproduced the R1 feature
  matrices exactly; a repeat five-seed fit and the parallel workers returned identical probabilities.
- Equal-time fallback: 0% of recordings for S and S_pub, 0.27% for S_annot, 0.03% for E_annot (failure threshold 5%).
- An earlier attempt on an NVIDIA L4 failed the gate on arms A and B only (GPU arithmetic in the generator) and stays
  sealed and unscored (`prereg/CHANGES_AFTER_FREEZE.md`).
- `stats.py` reproduces the notebook's per-task macro-F1 to within 2e-14.

## Primary hypothesis

**H1 (S minus E above 0) is rejected at the pre-registered level 0.0125.** Composite macro-F1 difference +3.3 points
(95% interval +0.8 to +5.7); the 1.25th percentile of the bootstrap distribution is +0.58. Rejection at 0.0125 holds
whatever happens to H2 in Run 2, since 0.0125 is the smallest level H1 can receive.

## Composite macro-F1 (MLP, five-seed ensemble, 2,000 child-level draws)

| Arm | Composite | 95% interval | Minus E | 95% interval |
|---|---|---|---|---|
| E, equal-time 75 bpm | 43.6 | 40.7 to 46.4 | | |
| **S, Springer (S_wpk)** | **46.9** | 43.9 to 49.7 | **+3.3** | +0.8 to +5.7 |
| S_pub, Springer, published heart-rate rule | 48.5 | 45.5 to 51.2 | +4.9 | +2.5 to +7.3 |
| A, synthetic-ECG R-peaks | 45.9 | 42.8 to 48.5 | +2.3 | -0.1 to +4.7 |
| B, synthetic R-peaks with acoustic refinement | 47.2 | 44.0 to 49.9 | +3.6 | +1.1 to +6.0 |
| D, annotated S1 onsets, fixed phases | 52.1 | 48.9 to 55.1 | +8.5 | +5.7 to +11.2 |
| C, annotation (reference) | 53.5 | 50.2 to 56.4 | +9.9 | +7.3 to +12.6 |
| E2, equal-time with the annotated count | 44.5 | 41.7 to 46.9 | +0.9 | -1.3 to +3.0 |
| S_annot, S inside annotated stretches | 47.3 | 44.5 to 49.6 | +3.6 | +1.3 to +5.9 |
| E_annot, E inside annotated stretches | 44.5 | 41.7 to 47.1 | +0.9 | -0.9 to +2.8 |

## Per task (MLP macro-F1)

| Arm | Detection | Grading | Timing | Shape |
|---|---|---|---|---|
| E | 71.0 | 53.8 | 38.1 | 38.9 |
| S | 78.6 | 54.7 | 43.3 | 42.8 |
| S_pub | 78.6 | 54.4 | 44.4 | 46.8 |
| A | 71.1 | 53.9 | 42.4 | 41.4 |
| B | 73.1 | 53.9 | 43.3 | 44.4 |
| D | 79.9 | 60.8 | 45.2 | 50.4 |
| C | 80.3 | 59.4 | 52.0 | 49.1 |

Intervals for every cell are in `stats_results.json` (about plus or minus 3 points for detection and 4 to 6 points
for the three characterisation tasks).

## Estimates (no tests)

- **Matched coverage.** Recovery fraction (S_annot - E_annot) / (C - E_annot) = **0.31** (95% interval 0.06 to 0.52):
  at the same coverage, Springer recovers about a third of the gain of the annotated boundaries. C minus S_annot:
  +6.2 points (+3.8 to +8.7).
- **Where S stands.** For detection, S (78.6) comes close to C (80.3); the shortfall sits in the characterisation
  tasks, above all timing (43.3 against 52.0) and shape (42.8 against 49.1).
- **Anchors and phases.** D, which keeps the annotated S1 onsets but replaces the annotated phases by fixed fractions,
  reaches +8.5 of C's +9.9 over E: most of the gain comes from where each cycle starts.
- **Synthetic ECG.** A lies +2.3 over E (interval crossing 0) and +1.0 below S (interval -1.4 to +3.5); its random
  forest result is -1.4 against E. The current synthetic ECG gives no reliable gain over equal-time windows.
- **S_pub against S.** +1.6 (-0.7 to +3.9). S_wpk stays the pre-registered primary comparator; the difference is
  within the interval.
- **Random forest (secondary).** Same order with smaller gaps: C +6.6, S +3.5, S_pub +3.3 over E; A -1.4.
- **Seed noise.** Single-seed composites spread over 1 to 3 points (S 44.0 to 47.1; C 49.2 to 51.5); the five-seed
  ensemble scores above every single seed for C and E.

## CNN baseline (PCG only, detection reference)

Run 17 October 2026 on the Run 1 folds (NVIDIA L4, 3.1 h, one declared seed 42, fp32, children as groups for the
fold checks, inner early-stopping split, and bootstrap). Its notebook equals the locked build except for trailing
spaces on blank lines in one cell (stripped by the editor on save; code identical). Recording-level predictions,
scored once on the same 2,000 child draws as the Run 1 arms (`cnn_stats.json`); the point values equal the
notebook's own.

| | Detection | Grading | Timing | Shape | Composite |
|---|---|---|---|---|---|
| CNN | 83.3 (80.3 to 85.9) | 51.2 | 42.1 | 35.4 | 42.9 (40.3 to 45.1) |

- Detection: the CNN scores highest of all arms: +12.3 over E (+10.1 to +14.4), +4.7 over S (+2.8 to +6.4), and +2.9
  over C (+0.9 to +4.8).
- Characterisation: the CNN's composite sits level with E (-0.7, -4.1 to +2.7) and below S (-4.0, -7.5 to -0.6).
- So detection is best served by a network on the whole recording, and characterisation by phase-aligned features:
  the same division as in R1, now on child-grouped folds.

## What this means for Run 2

The expected ceiling and floor for T are now measured: annotation-free Springer reaches 46.9, the annotated reference
53.5. T must be non-inferior to S (margin 2.5 points, H3) after beating E (H2). The room above S lies in the
characterisation tasks and in cycle-start precision, which is what the ECG-supervised S1 labels target. T against
the CNN will be reported as an estimate: the CNN leads on detection, the phase-aligned arms on characterisation.
