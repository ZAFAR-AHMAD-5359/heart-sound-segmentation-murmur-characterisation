# Experiment 2 results: the acoustic-label control arm T_ac (Run 3)

Run 3 `20261029_1955` (NVIDIA A100-SXM4-80GB; 0.7 h of stage time). Integrity gate PASSED: arm E reproduces Run 1 (detection: MLP identical, RF identical, max |dp| 0.0e+00; grading: MLP identical, RF identical, max |dp| 0.0e+00; timing: MLP identical, RF identical, max |dp| 0.0e+00; shape: MLP identical, RF identical, max |dp| 0.0e+00), the frozen A weights match the freeze record. Analysis by `notebooks/exp2_analysis.py`, run once, under the plan `EXP2_PREREG.md` (SHA-256 f0b9fe2b...), on the merged predictions of Runs 1, 2 and 3 with the shared child bootstrap (2,000 draws, seed 42). Fallback share: {"timing_net_ac": 0.0, "timing_net_ac_annot": 0.01633333333333333}.

## Primary result (pre-registered): H1, T minus T_ac, two-sided 95%

**T minus T_ac on the composite: -0.89 [-3.06, +1.31]. Reading fixed in the plan: no difference shown at this precision.** Per task (grading / timing / shape): -0.1 [-3.5, +3.2] / -1.5 [-5.0, +2.1] / -1.1 [-4.5, +2.3].

## Composite macro-F1 (MLP)

| Arm | Composite [95% CI] |
|---|---|
| E | 43.6 [40.7, 46.2] |
| S | 46.9 [44.0, 49.5] |
| S_pub | 48.5 [45.5, 51.2] |
| T | 49.9 [46.9, 52.4] |
| T_ac | 50.8 [47.3, 53.7] |
| W | 50.5 [47.3, 53.4] |
| C | 53.5 [50.1, 56.4] |

## Estimation (no p values)

| Contrast | Composite [95% CI] | Grading / timing / shape |
|---|---|---|
| T_ac - E | +7.1 [+4.3, +9.8] | +2.1 [-1.8, +5.8] / +10.8 [+6.0, +15.9] / +8.6 [+4.0, +13.1] |
| T_ac - S | +3.8 [+1.2, +6.5] | +1.2 [-2.7, +4.8] / +5.6 [+0.9, +10.2] / +4.7 [+0.4, +9.5] |
| T_ac - S_pub | +2.2 [-0.4, +4.9] | +1.5 [-2.4, +5.3] / +4.5 [-0.5, +9.4] / +0.7 [-4.3, +5.9] |
| T - E (Run 2) | +6.3 [+3.3, +9.1] | +2.0 [-2.3, +6.2] / +9.3 [+4.3, +14.1] / +7.5 [+2.6, +12.4] |
| T - S (Run 2) | +3.0 [+0.3, +5.6] | +1.1 [-3.2, +5.3] / +4.1 [-0.4, +8.5] / +3.7 [-0.8, +8.0] |

Recovery at matched coverage, (X_annot - E_annot)/(C - E_annot): T_ac 0.85 [0.59, 1.14] (draws with a non-positive denominator: 0); T 0.63 [0.37, 0.88].

Per-task macro-F1 (MLP): equal_time grading 53.8, timing 38.1, shape 38.9; springer_wpk grading 54.7, timing 43.3, shape 42.8; timing_net grading 55.8, timing 47.4, shape 46.4; timing_net_ac grading 55.9, timing 48.9, shape 47.5; pcg_seg grading 59.4, timing 52.0, shape 49.1.

Random forest composite: equal_time 38.3, springer_wpk 41.8, springer_pub 41.6, timing_net 42.8, timing_net_ac 42.3.

Detection macro-F1 (MLP): equal_time 71.0 [67.8, 74.0], springer_wpk 78.6 [75.7, 81.2], timing_net 78.1 [75.2, 80.7], timing_net_ac 80.4 [77.5, 83.1].

## Descriptive: labels, training beats, Gate A, boundary diagnostics

Label gate (reported, not required): acoustic labels S1 99.68% [99.17, 100.00], S2 99.66% [99.17, 99.98] of marks inside the hand-corrected training-a sound widened by 20 ms (5879 beats, 242 records); ECG-derived labels v3: S1 99.98%, S2 99.88% (5805 beats).
Acoustic training beats: 92,567 valid of 237,778 candidate beats in 2113 records (T: 69,176 valid beats); lag laws fitted on 36,349 beats of 687 subjects.

Gate A, kappa on the held-out paired test records (descriptive):

| Model | hfref | senssmart | bsslab | ephnogram | foster |
|---|---|---|---|---|---|
| A_seed11 | 0.987 | 0.969 | 0.957 | 0.686 | 0.952 |
| A_seed22 | 0.987 | 0.971 | 0.955 | 0.697 | 0.947 |
| A_seed33 | 0.985 | 0.954 | 0.953 | 0.715 | 0.945 |
| T_seed11 | 0.990 | 0.933 | 0.962 | 0.892 | 0.949 |
| T_seed22 | 0.986 | 0.974 | 0.958 | 0.938 | 0.949 |
| T_seed33 | 0.990 | 0.995 | 0.961 | 0.817 | 0.945 |

Boundary diagnostics against the annotation, medians over kept recordings (from results.json of Run 3):

| Arm | bpm | deletion_rate | failed_arm | fallback_frac | frac_fast | frac_s2_acoustic_median | insertion_rate | malformed_total | median_abs_offset_ms | median_offset_ms | median_rr_ms | n | n_cycles | n_matched_pairs | offset_sd_ms | onset_fallback_total | overdetection_ratio | rr_cv | rr_jump | s2_head_at_choice_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pcg_seg | 103.448 | 0.000 |  |  | 0.000 |  | 0.000 |  | 0.000 | 0.000 | 580.000 | 3000 | 17.000 | 60956 | 0.000 |  | 1.000 | 0.051 | 0.000 |  |
| equal_time | 75.000 | 0.625 |  |  | 0.000 |  | 0.739 |  | 79.750 | -0.250 | 800.000 | 3000 | 26.000 | 22785 | 86.172 |  | 1.389 | 0.000 | 0.000 |  |
| timing_net_ac | 99.358 | 0.077 | False | 0.000 | 0.000 | 1.000 | 0.463 | 376 | 18.000 | 17.375 | 603.875 | 3000 | 31.000 | 52942 | 20.581 | 277 | 1.536 | 0.338 | 0.125 | 0.757 |
| timing_net_ac_annot | 99.917 | 0.085 | False | 0.016 | 0.000 |  | 0.000 |  | 18.000 | 17.500 | 600.500 | 3000 | 15.000 | 52747 | 20.375 |  | 0.938 | 0.164 | 0.000 |  |

## Reading, against the outcomes fixed in the plan

- No difference shown at this precision (interval -0.9 [-3.1, +1.3] covers 0; no equivalence margin was set): the gain of T over Springer is not attributed to the ECG teacher; "electrocardiogram-supervised" describes how T was trained, not why it works, and the title and abstract are reworded as the plan states. A difference smaller than about 3 points was not expected to be shown.
- T_ac against Springer at its primary setting: +3.8 [+1.2, +6.5] (plan: T_ac below Springer would mean the acoustic labels are too poor to train the network; the gate shares say whether label quality or the absence of beat identity explains it).
- Every arm and contrast is reported; nothing changed after the result was seen.

## Boundaries of the claim

CirCor audio, annotations and labels; nothing trained or selected on CirCor; no CirCor label or annotation entered the acoustic labels or the network. The acoustic teacher is the Springer segmenter, whose own model was trained on ECG-derived labels in its development, so the control removes the ECG from this study's training, not from the lineage of the acoustic method. Record selection and the usable spans are shared with T.

## Files

- `runs/run3/20261029_1955/Extension_Run3.ipynb`: SHA-256 bd159118398085d601fa7bae64ce17ba6045248be8438cec1e1a26861908fcfb
- `runs/run3/20261029_1955/arm_failures.json`: SHA-256 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
- `runs/run3/20261029_1955/cache_hashes.json`: SHA-256 52ccc80b361d64a4aa13c8b6d18271f19a74a1167f6a59a1262fdeef30f46e7a
- `runs/run3/20261029_1955/diagnostics.csv`: SHA-256 836d3fa63243503336745e533f3e7aa4b7eb0866ac3f974e691f9fd168a615bf
- `runs/run3/20261029_1955/index.json`: SHA-256 fa83f4a5ddeff0388910ce5d9b86dc3ae88b2fb84dc0f177ee68cc795e02f90a
- `runs/run3/20261029_1955/oof_predictions.npz`: SHA-256 7175a1f37dede3ce0a0fe9e347e8dd336136373d77f8144a3e9b60193cc4bc3f
- `runs/run3/20261029_1955/results.json`: SHA-256 8a709c3d60cff04f1e3ecd7ae69b0167db5532b391f7c89dab8798b1a50e44fc
- `runs/run3/20261029_1955/run_flags.json`: SHA-256 f7e62cb1b7ef19e701f60210ec308a547f74ca551f91a4ce5088945f6585b7d1
- `runs/run3/20261029_1955/run_manifest.json`: SHA-256 38f980b9c7483e159fc7ac3d40ebc19185a6561ccb97b3cd5dd4a5ad3ebb5ae1
- `runs/run3/20261029_1955/stage_seconds.json`: SHA-256 b877a2e3bcb3ae358c3c42682733ce5424024ccca3927d63a29d3943a8b58f64
- `runs/exp2_analysis/exp2_results.json`: SHA-256 1abc88106f41ae98ee8e9ce3e4ca891702b48bcaeef9d92420e7b21be597cb8a
- Plan: `Next Study - Beat-Accurate Synthetic ECG/EXP2_PREREG.md`, SHA-256 f0b9fe2b9ddc76a2ffaa0c170cf13d0392e5e68f7a65a4dcdfbf190e5a31a184.
