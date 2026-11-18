# Springer LR-HSMM segmenter (arm S)

`springer_py.py` is Panel E's line-by-line Python port of the PhysioNet/CinC 2016 sample entry
(`sample2016.zip`, Springer's MATLAB code), so it is licensed GPL-3.0 (`LICENSE_sample2016_GPL3.txt`). Keep it a
separate module when the code is released.

`params/` holds the published pretrained parameters from that entry (B matrix, pi vector, total observation
distribution). `springer_arm.py` turns states into R1-format cycles at 4 kHz.

Variants fixed for the pre-registration:

| Variant | Heart-rate search | Rule | Systole floor | Role |
|---|---|---|---|---|
| S_wpk | 40 to 220 bpm | highest local maximum | 0.10 s | primary |
| S_pub | 30 to 120 bpm | global maximum (published) | 0.20 s | secondary |

Exposure disclosure: the exploratory pilot compared 219 CirCor recordings with the TSV before the
pre-registration, and S_wpk was devised after the first 20; it was chosen on ECG-truth heart rates from the paired
datasets, not on the TSV.
