# Run 2 results: frozen T and W on CirCor (scored once, 18 Oct 2026)

Run `runs/run2/20261018_1358` (NVIDIA A100-SXM4-80GB, 67 min); integrity gate PASSED (kept set pinned, folds, Run 1
cycle hashes of C, E and E_annot, arm E's features and predictions identical to Run 1, frozen weights, repeat fit,
pool). Fallback to equal-time windows: T 0%, W 0%, T_annot 1.3%, W_annot 0.3% (no arm failed). Analysis: Run 2's
predictions merged with Run 1's (`merge_run2.py`; arm E reproduced on all four tasks), then the frozen `stats.py`
(SHA-256 b44d78a8...) with T frozen; 2,000 child-bootstrap draws (816 children), seed 42.

## Pre-registered hypotheses (composite macro-F1, MLP)

| | contrast | estimate [95% CI] | 1.25th pct | level | verdict |
|---|---|---|---|---|---|
| H1 | S - E | +3.3 [+0.8, +5.7] | +0.58 | 0.025 | rejected (S above E) |
| H2 | T - E | +6.3 [+3.4, +9.3] | +3.11 | 0.025 | rejected (T above E) |
| H3 | T - S, margin 2.5 | +3.0 [+0.3, +5.7] | -0.08 | 0.025 | non-inferior |
| H4 | T - S > 0 | +3.0 [+0.3, +5.7] | -0.08 | 0.025 | rejected (T above S) |

H1 and H2 were both rejected at 0.0125, so each passed its alpha to the other and both, with H3 and H4 after H2,
were judged at 0.025, as the graphical procedure fixes. Sensitivity: H4's 1.25th percentile is -0.08, so at 0.0125
the superiority of T over S would not be claimed; non-inferiority holds at either level.

## Arms (composite macro-F1 [95% CI]; per task: detection, grading, timing, shape)

| arm | composite | det | gra | tim | sha |
|---|---|---|---|---|---|
| E equal-time | 43.6 [40.7, 46.4] | 71.0 | 53.8 | 38.1 | 38.9 |
| A synthetic ECG (GAN), fixed phases | 45.9 [42.8, 48.5] | 71.1 | 53.9 | 42.4 | 41.4 |
| B synthetic ECG (GAN), acoustic refinement | 47.2 [44.0, 49.9] | 73.1 | 53.9 | 43.3 | 44.4 |
| S Springer (40 to 220 bpm) | 46.9 [43.9, 49.7] | 78.6 | 54.7 | 43.3 | 42.8 |
| S_pub Springer (published rule) | 48.5 [45.5, 51.2] | 78.6 | 54.4 | 44.4 | 46.8 |
| **T** timing network | **49.9 [46.7, 52.5]** | 78.1 | 55.8 | 47.4 | 46.4 |
| **W** waveform network (synthetic ECG), acoustic refinement | **50.5 [47.2, 53.5]** | 80.1 | 55.0 | 48.1 | 48.6 |
| D annotated S1 onsets, fixed phases | 52.1 [48.9, 55.1] | 79.9 | 60.8 | 45.2 | 50.4 |
| C annotation (reference) | 53.5 [50.2, 56.4] | 80.3 | 59.4 | 52.0 | 49.1 |

Random forest composite, same ordering: E 38.3, S 41.8, W 42.5, T 42.8, C 44.8.

## Estimation (no hypothesis tests)

| contrast | estimate [95% CI] |
|---|---|
| W - E | +6.9 [+4.1, +9.6] |
| W - T | +0.7 [-1.8, +3.1] |
| W - S | +3.6 [+1.1, +6.0] |
| T - A | +4.0 [+1.1, +7.0] |
| T - B | +2.6 [-0.1, +5.6] |
| W - A | +4.6 [+1.9, +7.4] |
| W - B | +3.3 [+0.8, +6.0] |
| C - T | +3.6 [+1.0, +6.2] |
| C - W | +2.9 [+0.6, +5.4] |

Recovery fraction at matched coverage, (X_annot - E_annot) / (C - E_annot): T 0.63 [0.38, 0.87], W 0.68 [0.46, 0.92],
S 0.31 [0.06, 0.52].

## Reading

- The ECG-supervised timing network beats equal-time windows and Springer on the pre-registered endpoint.
- W, the synthetic-ECG route with networks trained on paired adult data, performs as well as T (difference +0.7,
  interval -1.8 to +3.1) and above Springer and both GAN arms. W was reported as failed on Panel G (two of three seeds
  below kappa 0.5 on EPHNOGRAM validation, set before training); on CirCor it carries the same gain as T.
- Neither closes the gap to the annotation: C stays 2.9 to 3.6 points ahead; at matched coverage T and W recover about
  two thirds of the annotation's gain, Springer about one third.
- Still to compute (pre-registered estimation): heart-rate bands and infants, the sensitivity analysis without
  fallback recordings, T against the CNN, and the descriptive boundary agreement with the annotation (Gate B).
