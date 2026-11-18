# Experiment 2, the acoustic-label control arm: pre-registration

Written 29 October 2026, after the main comparison (Runs 1 and 2), the error-budget experiments, the revision checks,
the independent code review, and the pre-registration of Experiment 1, and before any network of this experiment was
trained and before any CirCor score of it exists. Code: `extension_study/extstudy/labels_acoustic.py`, the kind `A`
additions to `model.py`, `losses.py`, `train.py` and `infer.py`, `notebooks/build_train_ac.py`,
`notebooks/build_run3.py`, `exp2_analysis.py`.

## Question

The revised manuscript states that its design does not isolate the electrocardiogram (ECG) supervision itself: no arm
trained the same network without ECG-derived labels, so the gain of the timing network T over the Springer segmenter
may come from the ECG as teacher, or from the network, its training corpus and its augmentation. This experiment
adds that arm. Does a network identical to T in architecture, training data, crops, augmentation, schedule, seeds and
selection rule, but trained with S1 and S2 labels derived from the phonocardiogram alone, recover the gain that T
recovers on CirCor?

## Known before writing

- Composite macro-F1 (MLP; grading, timing, shape): E 43.6, Springer 46.9 (published rule 48.5), T 49.9, W 50.5,
  C 53.5; T minus E +6.3 [3.4, 9.3]; T minus S +3.0 [0.3, 5.7]; T minus S_pub +1.3 [-1.4, 4.2]; recovery at matched
  coverage T 0.63, W 0.68, S 0.31; T's beat kappa on paired test data 0.82 to 0.995.
- The label gate of T's labels (v3) against the hand-corrected PhysioNet/CinC 2016 training-a annotations: S1 99.98%,
  S2 99.88% of marks inside the annotated sound widened by 20 ms.
- No acoustic-only label, network or score exists. The acoustic labels are derived while this plan is written; their
  gate result (below) is reported, not required, and no CirCor number exists before this plan is hashed.

## The control arm

**Labels (`labels_acoustic.py`).** The same paired records and usable spans as T's labels (record selection is shared,
so both arms train on the same audio). Beat identity comes from the Springer segmenter run on the phonocardiogram
alone (published parameters, the primary heart-rate setting of arm S: 40 to 220 beats per minute, highest local
maximum, systole floor 0.10 s), in windows of at most 60 s. S1 and S2 are the envelope maxima inside Springer's S1 and
S2 states, each widened by 30 ms; the validity rules are those of labels v3 (z-floors of 4 and 6, relative heights,
dominance, rival peaks, lag consistency over 11 neighbouring beats, regular RR), with the Springer S1-state onset as the
beat reference in place of the R-peak. No ECG sample, R-peak or ECG-derived mask enters any mark or rule. The lag laws
needed by the interval warping of training are fitted on the valid beats of training subjects with the same median
regression as T's. The R target is undefined: the network has no R head.

**Label gate (reported, not required).** The acoustic labels of PhysioNet/CinC 2016 training-a are scored against the
hand-corrected annotations exactly as T's were (`refcheck.py`: mark inside the annotated sound widened by 20 ms).
The shares are reported beside T's 99.98% and 99.88%. The arm trains whatever the result, because a difference in
label quality between an ECG teacher and an acoustic one is part of what the experiment measures; the plan states
this so that a poor gate cannot be used afterwards to discard the arm.

**Network and training (`build_train_ac.py`).** Kind `A`: the T architecture with two heads (S1, S2; 872,322
parameters), trained on the same pack (paired_v3 audio and splits) with the acoustic events and laws embedded in the
notebook and checked by hash, the same 12 s crops, batch composition, augmentation (interval warping to 70 to 180
beats per minute, device tilt, additive noise at 0 to 20 dB, murmur-like noise, frequency masking), AdamW schedule,
40 epochs, three seeds (11, 22, 33), and the same selection rule (beat-identity F1 of S1 from the S1-minus-S2 trace
through the shared decoder, native and 110 to 170 beats per minute warped validation sets, equal weight over
datasets). Nothing is tuned for this arm. The weights are hashed and frozen before any CirCor scoring.

**Scoring on CirCor (`build_run3.py`).** As Run 2: the frozen seeds averaged; T_ac's cycles from its S1 and S2 heads
through `cycles.t_cycles` (the construction of T, unchanged); fewer than two cycles, equal-time fallback, counted;
T_ac_annot restricted to the annotated stretches as T_annot was; arm E rebuilt as the reproduction control (its
features and predictions must equal Run 1's); the 3,000 recordings, the embedded child-grouped folds, the 464
features, the five-seed MLP and the random forest of Run 1, unchanged. One scoring run, logged in the ledger.

## Measures and decision, fixed now

Composite macro-F1 (MLP; grading, timing, shape) with the shared child bootstrap (2,000 draws, seed 42) on the
predictions of Run 1, Run 2 and this run merged (`extstudy/stats.py`, `analyse`).

- **Confirmatory, one hypothesis, two-sided, alpha 0.05.** H1: T and T_ac differ: T minus T_ac on the composite, read
  from the 2.5th and 97.5th percentiles of the bootstrap difference. Three readings: T above T_ac (lower bound above
  0): the ECG teacher contributes to the gain; T_ac above T (upper bound below 0): the acoustic teacher does at least
  as well and the ECG is a convenience, not a mechanism; neither (the interval covers 0): the difference is not shown
  at this precision, and the text says so. No equivalence margin is set.
- **Estimation (no p values):** T_ac minus E; T_ac minus S and minus S_pub; T_ac's recovery fraction at matched
  coverage, (T_ac_annot minus E_annot) / (C minus E_annot), beside T's 0.63; T_ac's per-task macro-F1; the random
  forest; T_ac's detection macro-F1 against the PCG-only convolutional network; T_ac's beat kappa on the paired test
  records beside T's; the fallback share and the boundary diagnostics against arm C (cycle-start F1, S2 failure share)
  beside T's.
- **Descriptive:** the acoustic label gate shares; the number of valid training beats of the acoustic labels against
  T's 69,176; the per-dataset lag laws of both label sets.
- Every arm and contrast is reported; no seed, rule or construction changes after a result is seen.
- Precision: the T minus S contrast carried a 95% interval of about +/-2.7 points, so a difference between T and T_ac
  smaller than about 3 points may not be shown.

## What each outcome means

- T above T_ac: the title's "electrocardiogram-supervised" names a mechanism; the ECG as teacher adds to what an
  acoustic teacher gives the same network. The size of the difference bounds how much.
- No difference shown or T_ac above T: the gain of T over Springer comes from the network, its corpus and its
  augmentation; "electrocardiogram-supervised" describes how T was trained, not why it works, and the manuscript's
  title and abstract are reworded accordingly.
- T_ac below Springer at its primary setting: the acoustic labels are too poor to train the network; the gate shares
  say whether label quality or the absence of beat identity explains it.

## Boundaries of the claim

Draws on CirCor audio, annotations and labels; trains and selects nothing on CirCor; no CirCor label or annotation
enters the acoustic labels or the network. The acoustic teacher is itself the Springer segmenter, whose model was
trained on ECG-derived labels in its own development, so the control removes the ECG from this study's training, not
from the lineage of the acoustic method; the plan states this. Record selection and the usable spans of long records
are shared with T, so the control does not test an entirely ECG-free pipeline from raw data.
