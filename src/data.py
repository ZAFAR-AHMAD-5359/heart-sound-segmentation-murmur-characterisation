"""
Dataset indexing and audio loading.

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

CirCor DigiScope distributes one .wav phonocardiogram per auscultation site and a
metadata table carrying the murmur labels per patient. The pipeline consumes the
audio and the labels; the cardiac-phase boundaries come from the synthetic ECG
reconstructed inside the pipeline (src/generator.py, src/gating.py), so no
annotation other than the class labels is read.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sig
from scipy.io import wavfile

TARGET_SR = 4000   # working sample rate for the acoustic feature analysis


def load_audio(wav_path, target_sr):
    """Mono audio at target_sr, max-abs normalised (matching training)."""
    sr_raw, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr_raw != target_sr:
        audio = sig.resample(audio, int(len(audio) * target_sr / sr_raw))
    return audio / (np.max(np.abs(audio)) + 1e-10)


def build_record_index(audio_dir, metadata_csv):
    """Pair every recording with its patient identifier and labels.

    Returns a list of (name, patient_id, labels, wav_path). The patient
    identifier is the leading numeric part of the file name and is what the
    cross-validation groups on.
    """
    meta = pd.read_csv(metadata_csv)
    labels = {}
    for _, r in meta.iterrows():
        pid = int(r['Patient ID'])
        labels[pid] = {
            'murmur':  {'Absent': 0, 'Present': 1}.get(str(r.get('Murmur')), -1),
            'grading': _grade_code(r.get('Systolic murmur grading')),
            'timing':  _timing_code(r.get('Systolic murmur timing')),
            'shape':   _shape_code(r.get('Systolic murmur shape')),
        }

    records = []
    for wav in sorted(Path(audio_dir).glob('*.wav')):
        m = re.match(r'^(\d+)', wav.stem)
        if not m:
            continue
        pid = int(m.group(1))
        if pid not in labels:
            continue
        records.append((wav.stem, pid, labels[pid], str(wav)))
    return records


def _grade_code(v):
    return {'I/VI': 0, 'II/VI': 1, 'III/VI': 2, 'IV/VI': 2,
            'V/VI': 2, 'VI/VI': 2}.get(str(v), -1)


def _timing_code(v):
    # The article's timing classes: holosystolic, early-systolic and
    # mid-systolic. The rare late-systolic label falls outside them and maps
    # to -1 (excluded), as does any diastolic finding.
    return {'Holosystolic': 0, 'Early-systolic': 1,
            'Mid-systolic': 2}.get(str(v), -1)


def _shape_code(v):
    # The article's shape classes: plateau, decrescendo and diamond, with the
    # rare crescendo shape excluded.
    return {'Plateau': 0, 'Decrescendo': 1, 'Diamond': 2}.get(str(v), -1)
