# Experiment 0c results: failure placed where Springer fails (22 Oct 2026)

Run `runs/exp0c/20261022_0916` (NVIDIA A100-SXM4-80GB, about 1.5 h); integrity gate PASSED: arm C reproduces Run 1,
Springer everywhere reproduces Run 1's Springer and W everywhere reproduces Run 2's W (MLP and random forest
predictions identical, probabilities within 5e-9); repeat fit and worker pool differences 0.0; folds verified. The
boundary sets equal, byte for byte, those of the local full-set check made before the run, and the executed notebook
equals the built one except for one trailing space in `s7_run`. Analysis by `notebooks/exp0c_analysis.py`, run once,
under the plan `EXP0C_PREREG.md` (SHA-256 b27aff85...). Arms C and E from Run 1: composite 53.5 and 43.6 (gain 9.9).
Failure sets: Springer 163 of 610 recordings (F_S), W 69 (F_W), both 29; no set fell back to equal-time windows.

## Primary result (pre-registered): placement

Retention R = (X - E) / (C - E), composite, seed-averaged [95% CI]; Delta = R(at random) - R(on F_S).

| failure form | on Springer's failed recordings | on as many random recordings | Delta |
|---|---|---|---|
| S2 moved by 100 ms in every cycle | 0.72 [0.57, 0.87] | 0.62 [0.50, 0.74] | -0.10 [-0.21, +0.002] |
| equal-time windows | 0.55 [0.32, 0.75] | 0.58 [0.44, 0.69] | +0.03 [-0.15, +0.21] |
| pooled | | | **-0.037 [-0.150, +0.074]** |

**Decision: no difference shown.** A failure costs no more when it falls on the recordings where Springer fails than
on random recordings. Per task (pooled): grading +0.17 [-0.12, +0.72], timing -0.08 [-0.25, +0.06], shape -0.09
[-0.36, +0.12]. On timing, the S2 failure costs less on Springer's recordings than on random ones (-0.22 [-0.42, -0.09];
per-task and per-form contrasts are estimation only).

The descriptive gain held by the sets accounts for this. Net gain of arm C over arm E (share wrong under E and right
under C, minus the reverse):

| task | inside F_S | other recordings | inside F_W | other recordings |
|---|---|---|---|---|
| grading | 0.080 | 0.056 | 0.087 | 0.059 |
| timing | 0.081 | 0.137 | 0.174 | 0.115 |
| shape | 0.123 | 0.091 | 0.119 | 0.097 |

Springer fails where accurate gating adds less to timing; W fails where it adds more.

## Secondary results (estimation only): Springer's and W's own cycles

| set | R | loss (1 - R) |
|---|---|---|
| Springer inside (its cycles in F_S, arm C elsewhere) | 0.63 [0.44, 0.81] | 0.37 [0.19, 0.56] |
| Springer outside (its cycles in the other 447, arm C in F_S) | 0.69 [0.48, 0.92] | 0.31 [0.08, 0.52] |
| Springer everywhere (Run 1) | 0.33 [0.10, 0.55] | 0.67 [0.45, 0.90] |
| W inside (its cycles in F_W, arm C elsewhere) | 0.63 [0.43, 0.82] | 0.37 [0.18, 0.57] |
| W outside (its cycles in the other 541, arm C in F_W) | 0.84 [0.61, 1.08] | 0.16 [-0.08, 0.39] |
| W everywhere (Run 2) | 0.70 [0.47, 0.93] | 0.30 [0.07, 0.53] |

- Springer's loss splits into two near-equal parts that add up: whole minus the sum of the parts -0.01 [-0.30, +0.30].
  About half of it arises in the 447 recordings where Springer does not fail by the 60 ms rule.
- Springer's real failures cost about what the stylised ones cost on the same recordings: they lie between the S2
  shift and equal-time windows (real minus S2 shift -0.10 [-0.27, +0.08]; real minus equal-time +0.08 [-0.15, +0.35]).
- W's failures in 69 recordings cost as much as Springer's in 163 (0.37 each), in line with W failing where more timing
  gain lies. W's parts add up to more than its whole (whole minus sum -0.23 [-0.51, +0.07]). A likely reason, not
  tested: W places S2 about 19 ms later than the annotation in every recording, so a training set that mixes W's and
  arm C's cycles carries a mismatch that neither pure arm has; Springer's S2 bias is 0 ms, and its parts add up.

## Exploratory, not pre-registered: boundaries where the arms do not fail

Written after the results (`runs/exp0c_analysis/exploratory_boundaries_outside_failure.csv`, from the S2 follow-up's
per-recording measures). Medians over the 407 recordings in which neither Springer nor W fails:

| arm | S2 onset error | cycles with S2 within 40 ms | IoU S1 / systole / S2 / diastole | start error SD within a recording | S2 bias |
|---|---|---|---|---|---|
| Springer | 18 ms | 100% | 0.83 / 0.83 / 0.80 / 0.89 | 14.5 ms | 0 ms |
| W | 21 ms | 86% | 0.66 / 0.73 / 0.53 / 0.81 | 12.8 ms | +19 ms |
| T | 23 ms | 82% | 0.66 / 0.72 / 0.53 / 0.80 | 12.7 ms | +22 ms |

Where it does not fail, Springer agrees with the annotation at least as well as W and T on every measure except the
cycle-to-cycle scatter of its starts, and still loses about a third of the gain there. In Experiment 0, timing jitter
of 10 to 20 ms cost 0.12 to 0.17 of the gain; a scatter difference of under 2 ms is too small to carry this loss on that
curve. No boundary measure in hand explains the second half of Springer's loss.

## Reading

- Not supported: the explanation that Springer's failures fall on the recordings that carry the gain. Its failed
  recordings carry less timing gain than the rest, and a failure placed there costs no more than at random.
- Supported with Experiment 0b: a failed recording loses its gain, so the share of failed recordings matters. For
  Springer this accounts for about half of its loss.
- Open: the other half arises where Springer's boundaries agree closely with the annotation. Agreement with the
  annotation, measured by boundary timing, overlap or failure share, does not capture everything the classifier needs
  from a segmentation.
- For the thesis hypothesis: ECG supervision helps through fewer failed recordings (W 11%, Springer 27%), which
  Experiments 0b and 0c tie to about half of Springer's shortfall. The rest of W's advantage over Springer is not
  explained by any measured boundary property, so the evidence does not yet support a claim that it comes from better
  R-peak timing.

## Boundaries

CirCor audio, annotations and labels; nothing trained or selected for any frozen model. Hybrid sets mix two
segmenters in one training set, which can add a cost of its own (see W above). The failure sets are defined against
the annotation by one rule (no paired cycle, or median S2 more than 60 ms off). Single contrasts carry intervals of
about +/-0.12 to 0.3.

## Files

`extension_study/runs/exp0c/20261022_0916/` (run files, SHA256SUMS); `extension_study/runs/exp0c_analysis/`
(`exp0c_results.json`, `exploratory_boundaries_outside_failure.csv`); this file.
