# Reproducibility package

Code accompanying **"Synthetic-ECG-gated phonocardiography for paediatric
heart-murmur characterisation: a cross-modal deep-learning method"**.

The package is the method end to end: a phonocardiogram goes in, a synthetic ECG is
reconstructed from it, R-peaks on that reconstruction divide each cardiac cycle into
four phases, 464 phase-aligned features are extracted, and one classifier per task
is trained and evaluated under stratified ten-fold cross-validation.

`PIPELINE.md` explains every stage and the reasoning behind each choice. Start there.

## Contents

| path | purpose |
|---|---|
| `run_pipeline.py` | end-to-end: index, reconstruct, gate, extract, classify, report |
| `src/data.py` | dataset indexing and audio loading |
| `src/generator.py` | the conditional adversarial network and windowed inference |
| `src/gating.py` | R-peak detection on the synthetic ECG and cycle boundaries |
| `src/features.py` | the 464 acoustic features |
| `src/model.py` | classifier, focal loss, per-fold training |
| `src/evaluate.py` | stratified cross-validation, paired fold tests, bootstrap intervals |
| `weights/README.md` | how to obtain the trained generator checkpoint |
| `PIPELINE.md` | stage-by-stage explanation |

## Data

Not redistributed here. All three datasets are public:

- **CirCor DigiScope**: https://physionet.org/content/circor-heart-sound/
  The phonocardiograms and the murmur labels used for classification.
- **EPHNOGRAM**: https://physionet.org/content/ephnogram/
  Simultaneous ECG and phonocardiogram, used to train the reconstruction model.
- **Michigan Heart Sound and Murmur Library**: University of Michigan,
  open.umich.edu. Used only as an external check.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the trained generator checkpoint under `weights/` (see `weights/README.md`).

## Run

```bash
python run_pipeline.py \
    --data     /path/to/circor/training_data \
    --metadata /path/to/circor/training_data.csv \
    --weights  weights/generator.pt \
    --boundaries both \
    --out      results/
```

`--boundaries both` runs the synthetic-ECG-gated analysis and the fixed-window
control through an otherwise identical pipeline, then reports the paired
difference: the contribution of cycle alignment. Both arms analyse identical
audio; only the boundaries differ.

`--split recording` (default) is the article's protocol, recording-level
stratified ten-fold cross-validation; `--split patient` groups folds by
participant instead. `--classifier` selects the multilayer perceptron (default)
or the random forest. `--limit 40` processes a small subset first, as an
installation check (tasks without enough labelled recordings for ten folds are
skipped and named). A GPU shortens the synthetic arm considerably;
`--device cpu` works but is slow.

## Output

- `results/results.json`: macro-F1, accuracy, 95% bootstrap interval and
  majority-class baseline per task and arm, plus the paired between-arm differences
- `results/out_of_fold_predictions.npz`: pooled out-of-fold predictions, true labels
  and patient identifiers, so any reported metric can be recomputed without
  rerunning the pipeline

## Extending the method

The stage most worth improving is R-peak detection (`src/gating.py`). The detector
returns an array of peak indices and everything downstream consumes only that
array, so a replacement drops in without touching any other stage. The timing
precision of the synthetic R-peaks bounds the quality of the phase alignment, and
better alignment should carry through to characterisation. Candidate directions:
a learned peak detector on the synthetic trace, fusion with acoustic S1/S2
segmentation, or a reconstruction objective that penalises peak-timing error.

## Notes for reviewers

- The default fold protocol is the article's: **recording-level stratified
  ten-fold cross-validation**. `--split patient` groups folds by participant, so
  no participant's recordings appear on both sides of a split; recordings from
  one participant are correlated, and the grouped protocol treats them as one
  unit.
- Between-arm comparisons are **paired**: both arms are scored on identical folds,
  compared with a paired t-test across folds (the article's test) and with a
  paired bootstrap interval resampled at the unit of the split.
- Macro-F1 and accuracy are both reported, with a majority-class baseline for
  every task; the labels are imbalanced, so a score is only interpretable against
  that baseline.
- `SEED = 42` throughout. See the determinism note at the end of `PIPELINE.md`.
