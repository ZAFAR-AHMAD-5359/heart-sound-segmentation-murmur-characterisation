# Run 2: the remaining pre-registered estimation analyses (22 Oct 2026)

Heart-rate bands, infants, the sensitivity analysis without fallback recordings, and T against the CNN (prereg
section 3; estimation, no p-values). Definitions were fixed in `prereg/CHANGES_AFTER_FREEZE.md` before any of these
was computed: heart rate from the annotated cycles (Run 1 `ref_bpm`); infants = CirCor Neonate and Infant; each subset
scored with the shared draws of the full set. Script `notebooks/run2_estimation.py`; output `run2_estimation.json`.
Checks: without a subset the script reproduces the frozen `stats.py` exactly (composite points and all 2,000 draws), and
the CNN's detection score reproduces `runs/run1_stats/cnn_stats.json`.

## Heart-rate bands and infants (composite macro-F1 [95% CI], MLP)

| subset | n (grading / timing / shape) | C | T | W | S | E |
|---|---|---|---|---|---|---|
| all | 610 / 606 / 603 | 53.5 [50.2, 56.4] | 49.9 [46.7, 52.5] | 50.5 [47.2, 53.5] | 46.9 [43.9, 49.7] | 43.6 [40.7, 46.4] |
| below 100 bpm | 237 / 233 / 235 | 53.0 [47.7, 57.2] | 51.7 [46.6, 55.9] | 48.8 [43.6, 52.9] | 46.7 [41.9, 51.4] | 42.5 [37.8, 46.7] |
| 100 to 125 bpm | 264 / 264 / 259 | 55.3 [49.7, 59.8] | 48.6 [43.8, 52.2] | 54.1 [48.5, 58.3] | 47.7 [42.7, 51.4] | 46.2 [41.5, 50.0] |
| 125 to 150 bpm | 92 / 92 / 92 | 44.5 [35.1, 51.3] | 41.4 [33.7, 47.1] | 39.2 [31.2, 45.0] | 39.5 [30.5, 46.6] | 35.0 [26.9, 41.2] |
| 150 bpm or more | 16 / 16 / 16 | 28.1 [23.2, 33.3] | 25.8 [17.5, 33.9] | 26.8 [21.8, 32.6] | 27.0 [19.0, 35.6] | 26.1 [17.8, 38.0] |
| infants (under 1 year) | 56 / 56 / 56 | 47.9 [35.1, 55.9] | 43.4 [32.6, 49.2] | 42.9 [32.6, 49.9] | 43.6 [31.4, 51.5] | 42.8 [32.1, 49.7] |

| subset | T - E | W - E | T - S | W - S |
|---|---|---|---|---|
| all | +6.3 [+3.4, +9.3] | +6.9 [+4.1, +9.6] | +3.0 [+0.3, +5.7] | +3.6 [+1.1, +6.0] |
| below 100 bpm | +9.2 [+4.6, +14.1] | +6.3 [+1.8, +10.8] | +5.0 [+0.8, +9.4] | +2.0 [-1.7, +5.6] |
| 100 to 125 bpm | +2.4 [-1.8, +6.5] | +7.9 [+3.5, +11.8] | +0.9 [-2.9, +4.4] | +6.4 [+2.8, +9.8] |
| 125 to 150 bpm | +6.4 [-0.8, +14.0] | +4.3 [-1.0, +9.9] | +1.9 [-4.5, +9.0] | -0.3 [-5.8, +5.5] |
| 150 bpm or more | -0.3 [-10.3, +8.1] | +0.8 [-9.8, +8.5] | -1.2 [-7.5, +6.0] | -0.1 [-6.1, +5.6] |
| infants | +0.6 [-7.0, +8.2] | +0.1 [-7.0, +7.0] | -0.3 [-8.4, +7.6] | -0.8 [-7.7, +6.0] |

Recovery fraction at matched coverage, (X_annot - E_annot) / (C - E_annot), reported only where the annotation's own
gain at matched coverage is shown (C - E_annot: all +8.9 [+6.3, +11.7]; below 100 bpm +12.6 [+8.6, +16.2]; 100 to 125
bpm +7.2 [+3.2, +11.4]; 125 to 150 bpm +5.5 [-0.5, +11.8]; 150 bpm or more 0.0 [-3.8, +4.4]; infants +6.4 [-1.3,
+11.9]):

| subset | T | W | S |
|---|---|---|---|
| all | 0.63 [0.38, 0.87] | 0.68 [0.46, 0.92] | 0.31 [0.06, 0.52] |
| below 100 bpm | 0.68 [0.44, 0.96] | 0.62 [0.38, 0.85] | 0.59 [0.35, 0.87] |
| 100 to 125 bpm | 0.36 [-0.42, 0.82] | 0.68 [0.14, 1.30] | 0.05 [-0.85, 0.45] |
| 125 to 150 bpm, 150 or more, infants | not estimable: the denominator's interval includes 0 | | |

What the subsets show:
- The gain over equal-time windows and over Springer holds below 125 bpm, where 82% of the recordings of the composite
  lie (83% of the detection set); which ECG
  arm carries it differs between the two bands (T below 100 bpm, W from 100 to 125 bpm). W minus T within a band was not
  a planned contrast and is not tested here.
- Above 125 bpm and in infants no arm, the annotation included, shows a gain over equal-time windows that its
  interval separates from 0; the 16 recordings at 150 bpm or more carry no information. Of the infant recordings, 71%
  lie at 125 bpm or more (78% in the detection set; median 132 bpm), against 12% of the others, so the two findings
  overlap. The claim of the paper
  holds for children under about 125 bpm and is untested for infants and fast rates.
- In every subset, macro-F1 is over all classes of the task; a class absent from a small subset scores 0 for every
  arm alike, which lowers all arms in the smallest subsets.

## Sensitivity without fallback recordings

E, S, S_pub, T and W had no fallback recording, so H1 to H4 are unchanged. Without the 52 recordings in which a
matched-coverage arm fell back (T_annot 38, W_annot 10, S_annot 8, E_annot 1; some overlap), the recovery fractions
are T 0.61 [0.36, 0.87], W 0.69 [0.46, 0.93] and S 0.31 [0.07, 0.53], as in the main analysis.

## T against the CNN (detection reference)

Detection macro-F1: CNN 83.3 [80.3, 85.9]; C 80.3 [77.3, 83.2]; W 80.1 [77.3, 82.7]; S 78.6 [75.7, 81.2]; T 78.1
[75.2, 80.7]; E 71.0 [67.8, 74.0]. Differences from the CNN: T -5.2 [-7.1, -3.1], W -3.2 [-5.1, -1.1], S -4.7 [-6.4,
-2.8], C -2.9 [-4.8, -0.9]. On the composite of grading, timing and shape, T leads the CNN by +7.0 [+3.2, +10.8] (CNN
42.9). A PCG-only network detects murmurs better than any gated feature set, the annotation's included; gated
features characterise detected murmurs better.

## Gate B (descriptive boundary agreement)

Reported in `Next Study - Beat-Accurate Synthetic ECG/EXP0_RESULTS.md` (arms at matched coverage: boundary F1 median
S_annot 0.97, W_annot 0.96, T_annot 0.92; median start offset 18 to 20 ms) and in the S2 follow-up.
