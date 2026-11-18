# Why do automatic arms recover less of the gain? Exploratory follow-up to Experiment 0 (19 Oct 2026)

Not pre-registered. Prompted by Experiment 0: inside the annotated stretches Springer, T and W all meet the beat-level
target (boundary F1 0.92 to 0.97), yet recover 0.31 to 0.68 of the gain where the corruption curve predicts about
0.85. Three measures were tried in turn, each fixed before it was computed; the sequence is reported in full because
the later ones were chosen after the earlier ones failed.

Inputs verified: the cycles of all nine arms (C, A, B, S, S_pub, T, W, D, E) reproduce the cycle hashes their runs
recorded; T and W were rebuilt from Run 2's traces and the analysed audio. 610 recordings labelled for grading,
timing or shape; cycles paired with the annotation by start (Hungarian, 150 ms). Scripts:
`notebooks/s2_phase_analysis.py`, `notebooks/s2_variability_analysis.py`.

## 1. Typical phase placement: does not explain it

| arm | S2 error, median abs | S2 within 40 ms | mean phase overlap (IoU) | systole IoU | retention |
|---|---|---|---|---|---|
| S | 20 ms | 90% | 0.70 | 0.71 | 0.33 |
| S_pub | 20 ms | 92% | 0.75 | 0.72 | 0.50 |
| T | 24 ms (bias +22) | 78% | 0.67 | 0.72 | 0.63 |
| W | 23 ms (bias +19) | 82% | 0.67 | 0.72 | 0.70 |
| D (exact starts, fixed fractions) | 34 ms | 67% | 0.66 | 0.64 | 0.86 |

Springer places S2 and the phase windows as well as T and W or better, yet recovers about half as much. Per recording,
a larger S2 error goes with more classification losses only weakly (Springer, timing: rho 0.14; T and W: none).

## 2. Cycle-to-cycle spread within a recording: ranks all arms, not the close ones

The within-recording SD of the start error ranks the eight arms with retention (rho 0.95), but the ranking rests on
A (57 ms) and E (82 ms); Springer, T and W differ by about 3 ms (15.9 against 12.5 to 12.8), where Experiment 0's
curve is flat.

## 3. Recording-to-recording consistency: fits the four automatic arms

| arm | recordings with S2 more than 60 ms off | between-recording SD of start offset | retention |
|---|---|---|---|
| S | 25% | 37 ms | 0.33 |
| S_pub | 19% | 29 ms | 0.50 |
| T | 13% | 17 ms | 0.63 |
| W | 9% | 17 ms | 0.70 |

Springer is near exact in about a third of recordings and about 100 ms off in another third; T and W carry a uniform
offset of about +20 ms with fewer failures. A classifier trained across recordings can absorb a uniform offset but not
failures confined to some recordings. Arm D fits too: its fixed fractions misplace S2 in 25% of recordings, and it
keeps only half of the gain on timing (0.51); grading and shape hold its composite at 0.86.

## Reading and limits

- Hypothesis carried forward: the gain is lost through recordings where the segmentation fails, not through the typical
  precision of the boundaries; ECG supervision helps by failing in fewer recordings.
- Hypothesis-generating only: the third measure was chosen after two others did not explain the pattern, and its
  perfect ordering rests on four arms. Experiment 0 corrupted every recording at the same rate, so it could not test
  this.
- Confirmatory test proposed (Experiment 0b, to pre-register before running): from arm C, misplace S2 (or the whole
  cycle anchor) in a random 10, 20 or 30% of recordings, leaving the rest exact, against the same total error spread
  evenly over all recordings; if the concentrated version costs clearly more, recording-level failure is the variable.
