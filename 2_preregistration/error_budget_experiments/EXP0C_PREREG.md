# Experiment 0c, failure placed where Springer fails: pre-registration

Written 21 October 2026, after Experiment 0b (`EXP0B_RESULTS.md`) and before any Experiment 0c feature, classifier
or score exists. Revised the same day before any code existed (the two reproduction controls for Springer and W were
added; the first version's hash is in the ledger). The failure sets below were fixed from boundaries alone (`extension_study/notebooks/exp0c_prep.py`).
Code: `extension_study/extstudy/placement.py`, `extension_study/notebooks/build_exp0c.py`, `exp0c_analysis.py`.

## Question

Experiment 0b showed that failure confined to some recordings costs more than error spread over all of them. At
Springer's failure share, the stylised failure placed on random recordings retains 0.61 to 0.68 of the gain; Springer
retains 0.33. Does the same stylised failure cost more when it falls on the recordings where Springer fails
than when it falls on as many recordings drawn at random?

## Known before writing

- Composite retentions: Springer 0.33, Springer published rule 0.50, T 0.63, W 0.70; the Experiment 0b curves.
- Per recording, Springer's S2 error goes with its classification losses only weakly (timing: rho 0.14; S2 follow-up).
- Failure sets (boundaries only): Springer fails in 163 of the 610 recordings (26.7%: 11 with no cycle paired with
  arm C, 152 with the median S2 onset more than 60 ms off); W fails in 69 (11.3%: 16 and 53); 29 recordings are in
  both. The cycles of arms C, Springer and W reproduce the hashes Run 1 and Run 2 recorded.
- No gain, loss or prediction has been computed for these sets.

## Data and pipeline

As Experiment 0b: the 610 recordings labelled for grading, timing or shape; arm C's annotated cycles; Run 1's
464-feature extractor, five-seed MLP (primary) and random forest on the embedded child-grouped folds; arm C rerun as
a reproduction control (predictions identical to Run 1, probabilities within 1e-5). Arms C, E and Springer come from
Run 1 and W from Run 2. Two further reproduction controls, Springer everywhere and W everywhere (the embedded cycles
in all 610 recordings), must give predictions identical to Run 1's Springer and Run 2's W; they are part of the
integrity gate, and a failed gate means the run is not used.

## Failure sets

An arm fails in a recording when none of its cycles pairs with arm C's within 150 ms, or when the median signed
difference of S2 onsets over the paired cycles exceeds 60 ms in size. F_S: Springer's failure set (163 recordings);
F_W: W's (69). Both are embedded in the notebook and recomputed there from the embedded Springer and W cycles and the
rebuilt arm C cycles; a mismatch stops the run.

## Conditions (30 boundary sets and three reproduction controls)

**Primary, placement of a stylised failure (163 recordings in every set).**
- S2 failure on F_S: in each recording of F_S every cycle's S2 window moves by 100 ms (the Experiment 0b rule; the
  direction drawn once per recording; clipped to keep 10 ms of systole and of diastole); five seeds.
- S2 failure at random: the same in 163 recordings drawn at random from the 610; ten seeds.
- Total failure on F_S: equal-time windows in every recording of F_S; one set, since it has no random part.
- Total failure at random: equal-time windows in 163 random recordings; ten seeds.

All other recordings keep arm C's cycles.

**Secondary, the real failures.**
- Springer inside: Springer's own cycles (Run 1) in F_S, arm C's elsewhere. Springer outside: Springer's cycles in the
  other 447 recordings, arm C's in F_S.
- W inside and W outside: likewise with W's cycles (Run 2) and F_W.

Seeds: condition seed 10000 x type (8 S2 failure on F_S, 9 S2 failure at random, 11 total failure at random) + s.
Random recordings: the first 163 of the seed's permutation of the sorted recording list. Draws within a recording
come from (condition seed, CRC-32 of its name).

## Measures and decision, fixed now

- Retention R = (X - E) / (C - E) on the composite (MLP; grading, timing, shape), seed-averaged per condition; the
  shared child bootstrap (2,000 draws, seed 42), with Run 1's Springer and Run 2's W in the same draws.
- Primary statistic: Delta = R(at random) - R(on F_S), averaged over the two failure forms (S2 failure, total
  failure). Decision, two-sided: placement on Springer's failures costs more if the 2.5th percentile of Delta lies
  above 0; it costs less if the 97.5th percentile lies below 0; otherwise no difference is shown. Delta per form and
  per task is reported.
- Secondary (estimation only): R(Springer inside) against R(S2 failure on F_S) and R(total failure on F_S), which
  compares real with stylised failure in the same recordings; the loss of Springer inside plus the loss of Springer
  outside, against Springer's own loss; the same for W.
- Descriptive: per task, the net gain of arm C over arm E (Run 1; share wrong under E and right under C, minus the
  reverse) in F_S and in the other recordings; in each primary condition, the net loss rate against arm C among the
  failed recordings and among the rest.
- Precision: single contrasts in Experiment 0b carried 95% intervals of about +/-0.12, so a Delta below about 0.1 may
  not be shown.
- Every condition is reported; no set, seed or rule changes after a result is seen.

## What each outcome means

- Costs more: Springer fails where the gain lies, so the placement of failures, and not only their number, explains
  part of its extra loss; a segmenter's failure share should be reported on the recordings that carry the gain.
- No difference: placement does not explain Springer's extra loss; the secondary contrasts show whether its real
  failures cost more than the stylised ones, or whether its errors outside F_S carry the loss.
- Costs less: Springer fails where gating has little to give, and its extra loss lies elsewhere.

## Boundaries of the claim

Draws on CirCor audio, annotations and labels; trains and selects nothing for any frozen model. F_S is defined by
Springer's S2 placement against the annotation; the recordings in it may be hard for reasons that also limit arm C,
and the experiment measures the cost of placement, not its cause. The stylised failures are those of Experiment 0b.
