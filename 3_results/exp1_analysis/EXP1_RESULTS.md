# Experiment 1 results: in-band interference at test time (29 Oct 2026)

Run `runs/exp1/20261029_1331` (Colab, NVIDIA A100-SXM4-40GB, 12 CPU workers; 2.8 h of stage time: boundaries 17 min, features 42 min, classifiers 104 min). Integrity gate PASSED: the five clean controls reproduce their runs (C, E and Springer against Run 1, T and W against Run 2; MLP and random forest predictions identical, probabilities within 5e-9), the frozen T and W weights match the freeze record, arm B's construction equals the shared module, the ESC-50 bank was fetched and every clip matched its hash, repeat-fit and worker-pool differences 0.0, folds verified. No arm failed, no worker failed, no cache was rebuilt. The executed notebook carries the built source hash c43bd605... (ledger, 29 Oct 17:41). Analysis by `notebooks/exp1_analysis.py`, run once, under the plan `EXP1_PREREG.md` (SHA-256 c0f0269d...; the run flags, the boundary file and the noise manifest carry the same hash). Achieved in-band SNR: median 10.00 and 0.00 dB in every condition (5th to 95th percentiles within 1e-13 dB of the target). Clean composites, reproduced: E 43.6, S 46.9, T 49.9, W 50.5, C 53.5; clean gains over E: S +3.3 [+0.9, +5.6], T +6.3 [+3.3, +9.1], W +6.9 [+4.1, +9.6]. All 610 murmur-present recordings (grading 610, timing 606, shape 603) and all six conditions ran.

## Primary result (pre-registered): H1

`T_seg_gauss0` minus E (T's boundaries read from audio with band-limited Gaussian noise at 0 dB in-band SNR, features from the clean audio): **+3.70 [+0.75, +6.71]; 2.5th percentile +0.75, above 0. H1 rejected: the gain holds.** Per task: grading +1.9 [-2.3, +6.1], timing +6.8 [+2.3, +11.6], shape +2.4 [-2.3, +7.1]. Retention of T's clean gain in this condition, R_seg = 0.59 [0.16, 0.98].

## Estimation: segmentation only (corrupted boundaries, clean features)

Composite macro-F1 (MLP), shared child bootstrap (2,000 draws, seed 42). X denotes the clean control of the same arm; R_seg = (X_seg_c - E)/(X - E) per draw, draws with a non-positive denominator excluded and counted.

| Condition | Arm | Composite | Change from clean | Gain over E [95% CI] | 2.5th pct | R_seg [95% CI] | Excluded draws |
|---|---|---|---|---|---|---|---|
| Gaussian, 10 dB | S | 48.5 | +1.6 [-0.0, +3.2] | +4.9 [+2.5, +7.2] | +2.48 | 1.49 [0.99, 3.41] | 7 |
| Gaussian, 10 dB | T | 49.5 | -0.4 [-2.3, +1.6] | +5.9 [+3.0, +8.4] | +2.98 | 0.94 [0.66, 1.36] | 0 |
| Gaussian, 10 dB | W | 50.5 | +0.0 [-2.1, +2.2] | +6.9 [+4.2, +9.4] | +4.17 | 1.00 [0.71, 1.45] | 0 |
| Gaussian, 0 dB | S | 46.8 | -0.2 [-2.4, +2.3] | +3.1 [+0.8, +5.5] | +0.81 | 0.95 [0.34, 2.38] | 7 |
| Gaussian, 0 dB | T | 47.3 | -2.6 [-4.9, -0.1] | +3.7 [+0.7, +6.7] | +0.75 | 0.59 [0.16, 0.98] | 0 |
| Gaussian, 0 dB | W | 49.0 | -1.5 [-3.7, +0.9] | +5.4 [+3.1, +7.6] | +3.14 | 0.78 [0.52, 1.19] | 0 |
| bursts, 10 dB | S | 47.6 | +0.7 [-0.9, +2.5] | +4.0 [+1.7, +6.2] | +1.67 | 1.22 [0.76, 2.59] | 7 |
| bursts, 10 dB | T | 50.1 | +0.2 [-1.7, +2.2] | +6.4 [+3.3, +9.1] | +3.35 | 1.03 [0.74, 1.49] | 0 |
| bursts, 10 dB | W | 50.0 | -0.6 [-2.5, +1.5] | +6.4 [+3.7, +9.0] | +3.67 | 0.92 [0.66, 1.29] | 0 |
| bursts, 0 dB | S | 46.6 | -0.3 [-2.2, +1.7] | +3.0 [+0.5, +5.3] | +0.51 | 0.91 [0.28, 1.93] | 7 |
| bursts, 0 dB | T | 49.1 | -0.7 [-2.5, +1.1] | +5.5 [+3.0, +8.0] | +3.02 | 0.88 [0.65, 1.23] | 0 |
| bursts, 0 dB | W | 49.5 | -1.0 [-3.0, +0.9] | +5.9 [+2.6, +8.9] | +2.65 | 0.85 [0.53, 1.15] | 0 |
| ESC-50, 10 dB | S | 48.5 | +1.6 [-0.3, +3.5] | +4.9 [+2.3, +7.4] | +2.30 | 1.49 [0.92, 3.67] | 7 |
| ESC-50, 10 dB | T | 49.4 | -0.5 [-2.2, +1.2] | +5.8 [+3.0, +8.2] | +2.97 | 0.92 [0.68, 1.29] | 0 |
| ESC-50, 10 dB | W | 49.4 | -1.2 [-3.0, +0.7] | +5.7 [+3.0, +8.1] | +3.04 | 0.83 [0.56, 1.15] | 0 |
| ESC-50, 0 dB | S | 45.7 | -1.2 [-3.5, +1.4] | +2.1 [-0.3, +4.6] | -0.27 | 0.64 [-0.13, 1.81] | 7 |
| ESC-50, 0 dB | T | 47.8 | -2.0 [-4.3, +0.4] | +4.2 [+1.6, +6.8] | +1.62 | 0.67 [0.34, 1.08] | 0 |
| ESC-50, 0 dB | W | 48.9 | -1.6 [-3.6, +0.5] | +5.3 [+2.9, +7.5] | +2.91 | 0.77 [0.51, 1.09] | 0 |

T against Springer under the same interference (segmentation only): Gaussian, 10 dB +1.0 [-1.3, +3.3]; Gaussian, 0 dB +0.6 [-2.6, +3.8]; bursts, 10 dB +2.4 [-0.4, +5.1]; bursts, 0 dB +2.5 [+0.2, +4.9]; ESC-50, 10 dB +0.8 [-1.7, +3.2]; ESC-50, 0 dB +2.1 [-0.7, +4.9]. Clean (Run 2): +3.0 [+0.3, +5.7].

Per task, gain over E of the segmentation-only arms (grading / timing / shape):

| Condition | S | T | W |
|---|---|---|---|
| Gaussian, 10 dB | +1.4 [-3.0, +5.9] / +5.4 [+1.6, +9.1] / +8.0 [+3.4, +12.4] | +2.4 [-1.4, +6.3] / +9.8 [+4.8, +14.6] / +5.4 [+0.7, +10.1] | +1.7 [-3.0, +6.1] / +10.1 [+5.7, +14.6] / +9.0 [+3.9, +13.7] |
| Gaussian, 0 dB | +0.2 [-3.9, +4.2] / +3.6 [-0.6, +7.6] / +5.6 [+1.5, +9.9] | +1.9 [-2.3, +6.1] / +6.8 [+2.3, +11.6] / +2.4 [-2.3, +7.1] | +1.2 [-3.7, +5.7] / +7.3 [+3.3, +11.2] / +7.7 [+3.2, +12.4] |
| bursts, 10 dB | +1.2 [-2.8, +5.0] / +4.9 [+1.2, +8.8] / +6.0 [+1.7, +10.4] | +0.5 [-4.0, +4.8] / +12.0 [+6.2, +18.0] / +6.8 [+2.2, +11.7] | +2.6 [-1.7, +6.8] / +8.0 [+4.1, +12.0] / +8.5 [+3.9, +13.4] |
| bursts, 0 dB | +1.1 [-3.1, +5.1] / +4.4 [+0.4, +8.5] / +3.5 [-1.4, +8.2] | +2.8 [-1.4, +7.2] / +7.2 [+2.5, +11.9] / +6.5 [+1.8, +11.1] | +0.3 [-3.1, +3.7] / +8.3 [+2.9, +13.4] / +9.1 [+3.7, +14.6] |
| ESC-50, 10 dB | +1.0 [-3.2, +5.0] / +6.2 [+2.0, +10.7] / +7.6 [+2.8, +12.3] | +2.1 [-2.0, +6.2] / +7.4 [+2.7, +11.8] / +7.9 [+3.0, +12.7] | +0.9 [-2.8, +4.6] / +7.5 [+3.1, +11.6] / +8.8 [+3.6, +13.9] |
| ESC-50, 0 dB | +1.2 [-2.4, +4.9] / +2.4 [-1.7, +6.1] / +2.8 [-1.5, +7.0] | +0.5 [-3.5, +4.3] / +5.5 [+1.0, +10.1] / +6.7 [+2.0, +11.3] | +1.7 [-2.1, +5.7] / +9.2 [+5.4, +13.2] / +5.1 [+0.3, +9.8] |

## Estimation: both stages (corrupted boundaries and corrupted features)

E_both and C_both measure the extractor's own sensitivity (equal-time windows and the annotation need no audio for their boundaries). Change of gain = (X_both_c - E_both_c) - (X - E), per draw.

| Condition | E_both | E_both - E | C_both - C | S over E_both | T over E_both | W over E_both | C over E_both | T - S (both) |
|---|---|---|---|---|---|---|---|---|
| Gaussian, 10 dB | 42.3 | -1.3 [-3.4, +0.7] | -2.3 [-4.5, -0.1] | +5.8 [+3.4, +8.0] | +5.7 [+2.7, +8.6] | +7.9 [+4.7, +10.8] | +8.8 [+5.8, +11.9] | -0.1 [-3.3, +2.9] |
| Gaussian, 0 dB | 40.3 | -3.3 [-6.0, -0.7] | -5.3 [-7.5, -2.9] | +4.7 [+1.9, +7.4] | +3.6 [+0.9, +6.2] | +2.9 [+0.5, +5.3] | +7.9 [+5.0, +10.9] | -1.1 [-3.7, +1.5] |
| bursts, 10 dB | 44.4 | +0.8 [-0.6, +2.1] | -2.3 [-3.8, -0.6] | +3.1 [+0.6, +5.4] | +5.6 [+2.6, +8.4] | +6.4 [+3.5, +9.2] | +6.8 [+4.1, +9.4] | +2.5 [-0.3, +5.2] |
| bursts, 0 dB | 44.0 | +0.3 [-1.1, +1.9] | +0.3 [-1.5, +2.1] | +4.0 [+1.3, +6.6] | +5.0 [+2.1, +7.6] | +6.3 [+3.7, +8.8] | +9.8 [+6.6, +13.0] | +1.0 [-1.5, +3.6] |
| ESC-50, 10 dB | 44.3 | +0.7 [-1.4, +2.8] | -0.6 [-2.8, +1.6] | +5.8 [+3.1, +8.5] | +6.7 [+4.3, +9.1] | +6.1 [+3.6, +8.6] | +8.6 [+5.7, +11.3] | +1.0 [-1.8, +3.6] |
| ESC-50, 0 dB | 40.5 | -3.1 [-5.4, -0.7] | -4.0 [-6.6, -1.2] | +4.0 [+1.6, +6.5] | +5.8 [+3.3, +8.2] | +7.1 [+4.4, +9.7] | +9.0 [+6.1, +11.7] | +1.8 [-0.7, +4.1] |

| Condition | Change of gain S | Change of gain T | Change of gain W | Change of gain C |
|---|---|---|---|---|
| Gaussian, 10 dB | +2.5 [-0.5, +5.4] | -0.5 [-3.6, +2.3] | +0.9 [-2.0, +3.9] | -1.0 [-3.8, +1.8] |
| Gaussian, 0 dB | +1.4 [-2.0, +4.9] | -2.7 [-6.3, +1.2] | -4.0 [-7.1, -0.7] | -2.0 [-4.9, +1.2] |
| bursts, 10 dB | -0.2 [-2.2, +1.9] | -0.6 [-2.7, +1.4] | -0.6 [-2.8, +1.9] | -3.0 [-4.8, -1.0] |
| bursts, 0 dB | +0.7 [-1.9, +3.4] | -1.3 [-3.5, +0.9] | -0.6 [-3.1, +1.8] | -0.0 [-2.4, +2.2] |
| ESC-50, 10 dB | +2.5 [-0.4, +5.4] | +0.5 [-2.3, +3.2] | -0.8 [-3.5, +1.9] | -1.3 [-4.1, +1.5] |
| ESC-50, 0 dB | +0.7 [-2.3, +3.8] | -0.4 [-3.5, +2.7] | +0.2 [-3.1, +3.4] | -0.9 [-3.7, +2.3] |

## Diagnostics (descriptive)

Whole-recording measures against the annotation (arm C), as in the Experiment 0 diagnostics: share of recordings that fell back to equal-time windows; median cycle-start F1; share of recordings whose median S2 onset lies more than 60 ms from the annotation (the error-budget failure criterion); median within-recording SD of the start offset. The whole-recording F1 counts cycles outside the annotated stretches as unmatched, so it is lower than the inside-annotation F1 of the main text (0.92 to 0.97) and is read as a change from the clean row of the same arm.

| Arm | Condition | Fallback share | Cycle-start F1 (median) | S2 > 60 ms share | Offset SD (ms) |
|---|---|---|---|---|---|
| S | clean | 0.000 | 0.667 | 0.254 | 15.9 |
| S | Gaussian, 10 dB | 0.000 | 0.657 | 0.249 | 16.0 |
| S | Gaussian, 0 dB | 0.000 | 0.600 | 0.273 | 21.5 |
| S | bursts, 10 dB | 0.000 | 0.667 | 0.257 | 16.3 |
| S | bursts, 0 dB | 0.000 | 0.615 | 0.313 | 23.1 |
| S | ESC-50, 10 dB | 0.000 | 0.642 | 0.266 | 19.8 |
| S | ESC-50, 0 dB | 0.000 | 0.483 | 0.536 | 41.7 |
| T | clean | 0.000 | 0.646 | 0.126 | 12.5 |
| T | Gaussian, 10 dB | 0.000 | 0.619 | 0.173 | 12.5 |
| T | Gaussian, 0 dB | 0.002 | 0.446 | 0.325 | 14.5 |
| T | bursts, 10 dB | 0.000 | 0.641 | 0.137 | 12.3 |
| T | bursts, 0 dB | 0.000 | 0.602 | 0.159 | 12.4 |
| T | ESC-50, 10 dB | 0.000 | 0.611 | 0.156 | 12.7 |
| T | ESC-50, 0 dB | 0.000 | 0.517 | 0.273 | 15.3 |
| W | clean | 0.000 | 0.667 | 0.089 | 12.8 |
| W | Gaussian, 10 dB | 0.000 | 0.650 | 0.111 | 13.4 |
| W | Gaussian, 0 dB | 0.000 | 0.512 | 0.255 | 16.8 |
| W | bursts, 10 dB | 0.000 | 0.667 | 0.096 | 12.7 |
| W | bursts, 0 dB | 0.000 | 0.642 | 0.112 | 13.2 |
| W | ESC-50, 10 dB | 0.000 | 0.650 | 0.111 | 13.4 |
| W | ESC-50, 0 dB | 0.000 | 0.590 | 0.147 | 18.9 |
| E | any | 0.000 | 0.324 | 0.935 | 80.5 |

## Reading, against the outcomes fixed in the plan

- **H1 rejected; retention.** At 10 dB in-band SNR the three segmenters kept their clean gain (R_seg 0.92 to 1.03 for T and W under every kind; Springer 1.22 to 1.49 with wide intervals, its clean gain being 3.3 points). At 0 dB, T kept 0.59 [0.16, 0.98] of its gain under Gaussian noise, 0.88 [0.65, 1.23] under bursts and 0.67 [0.34, 1.08] under ESC-50 sounds; W kept 0.78, 0.85 and 0.77. The gain over equal-time windows held (2.5th percentile above 0) for T and W in all six conditions and for Springer in five (ESC-50 at 0 dB: +2.1 [-0.3, +4.6]). The premise holds for segmentation: boundaries read from audio that carries interference at the level of the heart sounds themselves still beat equal-time windows on clean features.
- **Where the loss lies when both stages are exposed: in the features.** E_both fell 3.3 [0.7, 6.0] points under Gaussian noise at 0 dB and 3.1 [0.7, 5.5] under ESC-50 at 0 dB (continuous interference), and moved within 1.3 points under bursts and at 10 dB; C_both (annotated boundaries, corrupted features) fell 5.3 [2.9, 7.5] under Gaussian noise at 0 dB. The gain over E_both held for every arm in every condition (T +3.6 to +6.7, W +2.9 to +7.9, S +3.1 to +5.8, C +6.8 to +9.8). Of the 24 changes of gain, two had intervals excluding 0: W under Gaussian noise at 0 dB (-4.0 [-7.1, -0.7]) and C under bursts at 10 dB (-3.0 [-4.8, -1.0]). The both-stage arms did not fall to equal-time windows.
- **T against Springer under the same interference:** +0.6 to +2.5 points (segmentation only), the interval above 0 only under bursts at 0 dB (+2.5 [+0.2, +4.9]); clean +3.0 [+0.3, +5.7]. No divergence between the networks and Springer was shown, so the networks' noise augmentation cannot be read from this experiment, as the plan anticipated.
- **Boundary error against gain.** Under Gaussian noise at 0 dB, T's median cycle-start F1 fell from 0.646 to 0.446 and its S2 failure share rose from 0.126 to 0.325 (above Springer's clean 0.254), and T still gained +3.7 over E, more than Springer's clean gain. Under ESC-50 at 0 dB Springer's S2 failure share doubled to 0.536 and its gain over E was no longer shown. W's boundaries degraded least (S2 failure share at most 0.255). One recording (0.2%) fell back to equal-time windows, for T under Gaussian noise at 0 dB.
- **Precision.** Contrasts carry intervals of about +/-3 points and the 0 dB retention values about +/-0.4, as the plan expected, so changes smaller than that are not shown. Every condition and arm is reported; nothing changed after the result was seen.

## Boundaries of the claim

CirCor audio, annotations and labels; nothing trained or selected for any frozen model; no CirCor label or annotation entered any segmenter. The interference is synthetic or borrowed from ESC-50 and added to field recordings that carry their own ambient noise, so the levels are relative to the recorded signal, not to a quiet room. The experiment tests gating under interference confined to the murmur band and not locked to the cycle; it does not test diagnoses, and it cannot attribute the difference between the networks and Springer to the electrocardiogram supervision (Experiment 2 addresses that).

## Files

- `runs/exp1/20261029_1331/results.json`: SHA-256 dfbb2567208495c3d58b994dde33880f2ec7652b087a6aa2c4139a54e5e93c64
- `runs/exp1/20261029_1331/oof_predictions.npz`: SHA-256 42d55af03d4a11a4fc4e6983f589cdca1c95427b052b2a451a60b69d9a61a2a4
- `runs/exp1/20261029_1331/exp1_boundaries.json`: SHA-256 252f6a9dfac84722fdb3b0bd8d7aad8b1f2fc89251d25baaa4eff91405b094ab
- `runs/exp1/20261029_1331/exp1_noise_manifest.json`: SHA-256 47d8deab5a508b92c73fb152bff838b392fe49e1e54fbe082aa31c85cff67318
- `runs/exp1/20261029_1331/run_manifest.json`: SHA-256 a67cac2f543650d540d0546e32f5992f2ebd85d5152619c0ead9e195a38ca595
- `runs/exp1/20261029_1331/run_flags.json`: SHA-256 ad6da7b95b3cf93b149a7f7e77528a5d3cf65fd20d92bd8fd192f9ffc55be6ba
- `runs/exp1/20261029_1331/index.json`: SHA-256 fa83f4a5ddeff0388910ce5d9b86dc3ae88b2fb84dc0f177ee68cc795e02f90a
- `runs/exp1/20261029_1331/stage_seconds.json`: SHA-256 9cafe30ad19e96233978352e668aef6acd9e1f8b932f5694c357d0bcaf0aebf8
- `runs/exp1/20261029_1331/Experiment1_InBandNoise.ipynb`: SHA-256 5990ea1d873d44bb4903ded2503d7858ceb5251a7d8782c83117c888956fadda
- `runs/exp1_analysis/exp1_results.json`: SHA-256 d7b22db5e85a575ce75a3ca4198afa9325995e2106105f33ae2a2bd8a362bc70
- Not downloaded from Drive (remain in `Extension_Study/runs/exp1/20261029_1331/`): `cache/` (230 files), `diagnostics.csv`, `arm_failures.json` (the manifest records an empty list), `cache_hashes.json`, `folds/`.
- Plan: `Next Study - Beat-Accurate Synthetic ECG/EXP1_PREREG.md`, SHA-256 c0f0269d27bfed486240ff3a1ce1ca53c05f57531f6c5e05625d875bc1f3df9b.
