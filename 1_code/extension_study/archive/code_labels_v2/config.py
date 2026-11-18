"""Paths, constants, and seeds for the extension study (see ../CODE_DESIGN.md and ../../08_EXTENSION_DESIGN.md).

The same code runs locally (Windows, raw data on E:) and on Google Colab (Drive mounted at /content/drive).
CirCor paths are exposed only through `circor_paths()`; tests/test_firewall.py fails if a training or
selection module references them (the CirCor firewall).
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import os
import sys

ON_COLAB = os.path.isdir('/content') and ('google.colab' in sys.modules or os.path.isdir('/content/drive'))

# ------------------------------------------------------------------ local layout
PHD_LOCAL = 'E:/NUST MS Thesis/PhD Level Research'
LOCAL = dict(
    phd=PHD_LOCAL,
    pcgecg_code=PHD_LOCAL + '/code',
    harmonised=PHD_LOCAL + '/derived/harmonised',
    audit_table=PHD_LOCAL + '/derived/audit/audit_records.csv',
    beats=PHD_LOCAL + '/derived/beats_v2',                     # teacher R-peaks (qrs.py, relabel.py)
    study=PHD_LOCAL + '/extension_study',
    runs=PHD_LOCAL + '/extension_study/runs',
    r1_run='E:/NUST MS Thesis/Final Draft/Submitted File/Revision-R1/notebooks/Results/R1_Gating_Campaign/20261003_1236',
    r1_builder='E:/NUST MS Thesis/Final Draft/Submitted File/Revision-R1/notebooks/build_gating_campaign.py',
    generator='E:/NUST MS Thesis/Final Draft/Submission/materials/code/weights/generator.pt',
    springer_sample2016=PHD_LOCAL + '/Datasets/Dataset-4/Dataset-Cinc2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0',
)
_CIRCOR_LOCAL = dict(
    training_data='E:/NUST MS Thesis/Milestone-3/Dataset/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data',
    metadata='E:/NUST MS Thesis/Milestone-3/Dataset/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv',
)

# acoustic labels and the training pack built from them; v1 (labels_v1, paired_v1) failed the 200-beat label check
# on 17 Oct 2026 and stays on disk unchanged as its record (prereg/CHANGES_AFTER_FREEZE.md)
LABELS = 'labels_v2'
PACK = 'paired_v2'

# ------------------------------------------------------------------ Drive layout (latest, from the R1 notebooks)
DRIVE_ROOT = '/content/drive/MyDrive'
DRIVE = dict(
    root=DRIVE_ROOT,
    study=DRIVE_ROOT + '/JBHI-Revision/Extension_Study',                 # new: everything this study writes
    corpus=DRIVE_ROOT + '/JBHI-Revision/Extension_Study/corpus',          # the training pack built locally, uploaded
    checkpoints=DRIVE_ROOT + '/JBHI-Revision/Extension_Study/checkpoints',
    runs=DRIVE_ROOT + '/JBHI-Revision/Extension_Study/runs',
    paired_raw=DRIVE_ROOT + '/JBHI-Revision/Extension_Study/paired_datasets',   # optional raw upload (not needed for training)
    r1_root=DRIVE_ROOT + '/JBHI-Revision/R1_Gating_Campaign',
    generator=DRIVE_ROOT + '/October Thesis Dataset/trail4_checkpoints/trail4_best.pt',
)
_CIRCOR_DRIVE_HINT = (DRIVE_ROOT + '/JBHI-Revision/October Thesis Dataset/CirCor DigiScope/'
                      'the-circor-digiscope-phonocardiogram-dataset-1.0.3/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data')


def paths():
    """The active path set for this environment (CirCor excluded on purpose)."""
    return DRIVE if ON_COLAB else LOCAL


def circor_paths():
    """CirCor locations. Allowed callers: infer_circor, springer, splits (metadata only), the R1 adapter, stats."""
    if not ON_COLAB:
        return dict(_CIRCOR_LOCAL)
    td = _CIRCOR_DRIVE_HINT if os.path.isdir(_CIRCOR_DRIVE_HINT) else _discover_circor_drive()
    return dict(training_data=td, metadata=os.path.join(os.path.dirname(td), 'training_data.csv'))


def _discover_circor_drive(maxdepth=8):
    """Same rule as the R1 notebook: the training_data folder with the most paired .wav + .tsv files."""
    best, best_n = None, -1
    root = os.path.abspath(DRIVE_ROOT)
    for d, subdirs, files in os.walk(root):
        if d[len(root):].count(os.sep) >= maxdepth:
            subdirs[:] = []
            continue
        subdirs[:] = [s for s in subdirs if s not in {'.Trash', '.ipynb_checkpoints', '__pycache__'}]
        if os.path.basename(d) == 'training_data':
            wav = {f[:-4] for f in files if f.endswith('.wav')}
            n = sum(f[:-4] in wav for f in files if f.endswith('.tsv'))
            if n > best_n:
                best, best_n = d, n
    if best is None:
        raise FileNotFoundError('no CirCor training_data folder on Drive')
    return best


# ------------------------------------------------------------------ constants (fixed by 08_EXTENSION_DESIGN.md)
@dataclass(frozen=True)
class Signal:
    fs_ecg: int = 500
    fs_pcg_harmonised: int = 2000
    fs_pcg_model: int = 1000
    fs_frames: int = 250                    # 4 ms hop
    band_hz: tuple = (25, 400)
    stft_win_ms: int = 32
    stft_bins: int = 24                     # 31 to 390 Hz
    fs_circor: int = 4000


@dataclass(frozen=True)
class Folds:
    n_splits: int = 10
    seed: int = 42
    mlp_seed_bases: tuple = (42, 1042, 2042, 3042, 4042)
    bootstrap_draws: int = 2000


@dataclass(frozen=True)
class Analysis:
    alpha_total_one_sided: float = 0.025
    ni_margin_points: float = 2.5
    fallback_fail_share: float = 0.05
    gate_b_tolerances_ms: tuple = (40, 60, 100)


PKT = timezone(timedelta(hours=5))
FREEZE_T = datetime(2026, 10, 30, 23, 59, tzinfo=PKT)       # T and W weights frozen
FREEZE_S_RT = datetime(2026, 10, 29, 23, 59, tzinfo=PKT)    # retrained Springer: frozen or dropped
FREEZE_ALL = datetime(2026, 11, 5, 23, 59, tzinfo=PKT)      # results enter the paper as they stand
DUE = datetime(2026, 11, 20, 23, 59, tzinfo=PKT)            # EHJ-DH revision due (editor confirmed 16 Oct 2026)

@dataclass(frozen=True)
class Beats:
    """Teacher R-peak rule (Panel H, 15 Oct 2026)."""
    tol_ms: int = 50                        # vote and agreement tolerance, after each detector's record offset
    offset_search_ms: int = 100             # pairs used to estimate a detector's record offset
    wfdb_clip: float = 3.0                  # XQRS/GQRS input clipped at this many QRS amplitudes
    xg_f1_min: float = 0.90                 # XQRS against GQRS
    rx_f1_min: float = 0.95                 # robust detector against XQRS
    local_s: float = 5.0                    # confirmation share measured over robust beats within +/- local_s
    local_confirm_min: float = 0.80         # below this share of good robust beats, the neighbourhood carries no labels
    extra_amp_min: float = 0.50             # rejected candidate with this share of the local beat level + one wfdb
                                            # detection: its cycle carries no labels (hidden premature beat)
    corr_min: float = 0.80                  # template correlation of a good beat (real ECG: median record 0% below;
                                            # beats placed in electrode noise: median 0.5)
    window_s: float = 60.0                  # reporting windows (agreement per minute); the sync windows live in the audit
    gap_ratio: float = 1.5                  # 08 design: RR above 1.5 x the local median carries no labels
    refractory_s: float = 0.20              # shared decoder: max(0.20 s, 0.35 x expected RR)
    refractory_rel: float = 0.35
    lag_gap_max_ms: float = 8.0             # S-dominant minus R-dominant S1 lag, per dataset
    min_s_beats: int = 5                    # S-dominant beats a record needs to enter the lag-gap check


SIGNAL, FOLDS, ANALYSIS, BEATS = Signal(), Folds(), Analysis(), Beats()

# EPHNOGRAM records that duplicate other records byte for byte (Panel A, D2; verified 15 Oct 2026)
EXCLUDE_RECORDS = {('ephnogram', 'ECGPCG0001'), ('ephnogram', 'ECGPCG0002')}
# one ECG lead per dataset (Panel A, D3); fall back only when this lead fails the ECG check
PRIMARY_LEAD = {'senssmart': 'II', 'bsslab': 'ecg_ch4'}
LEAD_ORDER = {'senssmart': ('II', 'I', 'V3', 'V4'), 'bsslab': ('ecg_ch4', 'ecg_ch5', 'ecg_ch6', 'ecg_ch7')}
