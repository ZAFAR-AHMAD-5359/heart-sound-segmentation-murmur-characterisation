# Code: harmonised paired PCG-ECG data

Python package `pcgecg`, written 15 October 2026 for the extension experiments and the PhD study.

## Modules

| Module | What it does |
|---|---|
| `pcgecg/readers.py` | one reader per dataset, all returning the same `Record`: EPHNOGRAM (MAT), SensSmartTech (WFDB), PhysioNet 2016 training-a (WFDB), FOSTER (CSV inside a nested zip, read in place), BSSLAB (LabScribe MAT, custom reader), HFrEF (WAV inside zips) |
| `pcgecg/signals.py` | resampling, consensus R-peaks from four detectors with polarity correction and R-maximum refinement, RR statistics, PCG envelope, and the R-triggered synchronisation statistics |
| `pcgecg/audit.py` | audits every record in parallel and writes harmonised arrays |
| `pcgecg/calibrate.py` | negative control for the synchronisation rule: PCG paired with another subject's R-peaks |
| `pcgecg/report.py` | per-dataset summary table and figures |

## Run

```
cd "E:\NUST MS Thesis\PhD Level Research\code"
python -m pcgecg.audit --workers 12          # about 10 minutes; --limit 3 for a smoke test
python -m pcgecg.calibrate
python -m pcgecg.report
```

## Outputs, under `..\derived\`

| Path | Content |
|---|---|
| `harmonised\<dataset>\<record>.npz` | `pcg` at 2 kHz, `ecg` and `ecg_clean` at 500 Hz, consensus `rpeaks` at 500 Hz, the ECG lead used, metadata, and extra channels at 500 Hz (FOSTER mechanical sensors and respiration; BSSLAB respiration; SensSmartTech accelerometer) |
| `audit\audit_records.csv` | one row per record: duration, ECG agreement, heart rate, RR outliers, PCG quality, S1 lag, lock statistic, per-beat spread, drift, flags |
| `audit\triggered_<dataset>.npz` | R-triggered PCG-envelope average per record |
| `audit\sync_negative_control.csv` | the negative control |
| `audit\summary_by_dataset.csv`, `audit\figures\` | the report |

## Rules coded in

- **ecg_ok:** the median F1 of each detector against the consensus is at least 0.90, at least 5 beats, and at
  most 10% of RR intervals deviate by more than 20% from an 11-beat running median.
- **pcg_ok:** the PCG is not flat.
- **sync_ok** (final, applied by `refresh_sync.py`): lock effect (mean over beats of the S1-envelope excess
  divided by its SD) at least 1.0; S1 lag within 60 ms of the dataset's own centre, computed without the HFrEF
  community test set; per-beat lag spread (IQR) at most 80 ms. Calibrated on the negative control.
- **drift_ok:** the S1 lag jumps by at most 40 ms between consecutive 60 s windows (slow change with heart
  rate is physiological and allowed).
- **usable:** all four.
- **BSSLAB:** each stethoscope channel is a separate record sharing one ECG, so its beat counts appear twice;
  count unique beats per ECG file.

Filters are second-order sections with a pole-radius assertion; every saved row carries its record ID.
