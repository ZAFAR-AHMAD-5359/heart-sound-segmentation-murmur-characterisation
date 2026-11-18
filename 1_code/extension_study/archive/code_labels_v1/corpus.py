"""Training pack for Colab, built locally from the six paired PCG-ECG datasets only.

Files in <out>/<version>/:
  pcg_bp_1k_int16.npy   every usable record, band-passed 25 to 400 Hz at 1 kHz (frontend.bandpass), concatenated;
                        record k occupies [offset_k, offset_k + n_k) and equals int16 / scale_k
  bin_medians.npy       (records, 24) per-bin log-power medians of each whole recording (frontend.bin_medians)
  records.csv           record_id, dataset, record, subject, group, role, offset, n_samples, scale, and counts
  events.npz            per beat, concatenated in record order: record_id, r_s, r_next_s, s1_s, s2_s, r_target_s,
                        s1_on_s, s1_off_s, s2_on_s, s2_off_s, hr, rr_ms, valid
  manifest.json         SHA-256 of every file; the detector, label-law, and split-table hashes; counts per dataset
                        and role
Loss masks follow from events: the cycles [r_s, r_next_s) of valid beats; everything else carries no loss.
Usage:  python -m extstudy.corpus [--version paired_v1]
"""
import argparse, hashlib, json, os, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from extstudy import frontend as F
from extstudy.config import LOCAL

EVENT_KEYS = ('r_s', 'r_next_s', 's1_s', 's2_s', 'r_target_s', 's1_on_s', 's1_off_s', 's2_on_s', 's2_off_s', 'hr', 'rr_ms', 'valid')
INT16_TARGET = 30000.0


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _one(row):
    z = np.load(os.path.join(LOCAL['harmonised'], row['dataset'], row['record'] + '.npz'))
    x = F.bandpass(F.to_1k(z['pcg'], float(z['fs_pcg'])))
    med = F.bin_medians(x)
    a = np.percentile(np.abs(x), 99.99) + 1e-12
    scale = INT16_TARGET / a
    q = np.clip(np.round(x * scale), -32767, 32767).astype(np.int16)
    L = np.load(os.path.join(LOCAL['phd'], 'derived', 'labels_v1', row['dataset'], row['record'] + '.npz'))
    ev = {k: np.asarray(L[k]) for k in EVENT_KEYS}
    return row, q, float(scale), med.astype(np.float32), ev, str(L['laws_sha256'])


def build(version='paired_v1', workers=12):
    from multiprocessing import Pool
    from extstudy import splits, qrs
    out = os.path.join(LOCAL['phd'], 'derived', 'corpus', version)
    os.makedirs(out, exist_ok=True)
    lab = pd.read_csv(os.path.join(LOCAL['phd'], 'derived', 'labels_v1', 'labels_records.csv'))
    T = splits.paired_splits()
    T = T.merge(lab[['dataset', 'record', 'n_beats', 'n_valid', 'valid_s']], on=['dataset', 'record'], how='left')
    T = T[T['n_valid'].fillna(0) > 0].sort_values(['dataset', 'record']).reset_index(drop=True)
    rows = T.to_dict('records')
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_one, rows, chunksize=4)
    n_tot = sum(len(q) for _, q, _, _, _, _ in res)
    pcg_path = os.path.join(out, 'pcg_bp_1k_int16.npy')
    big = np.lib.format.open_memmap(pcg_path, mode='w+', dtype=np.int16, shape=(n_tot,))
    recs, meds, evs, laws = [], [], {k: [] for k in EVENT_KEYS + ('record_id',)}, set()
    off = 0
    for k, (row, q, scale, med, ev, lh) in enumerate(res):
        assert (row['dataset'], row['record']) == (rows[k]['dataset'], rows[k]['record'])      # IDs travel with data
        big[off:off + len(q)] = q
        recs.append(dict(record_id=k, dataset=row['dataset'], record=row['record'], subject=row['subject'], group=row['group'],
                         role=row['role'], offset=off, n_samples=len(q), scale=scale, duration_s=len(q) / F.FS,
                         n_beats=int(row['n_beats']), n_valid=int(row['n_valid']), valid_s=float(row['valid_s'])))
        meds.append(med)
        for key in EVENT_KEYS:
            evs[key].append(ev[key])
        evs['record_id'].append(np.full(len(ev['r_s']), k, np.int32))
        laws.add(lh)
        off += len(q)
    big.flush(); del big
    assert len(laws) == 1, 'labels from more than one law set'
    R = pd.DataFrame(recs)
    R.to_csv(os.path.join(out, 'records.csv'), index=False)
    np.save(os.path.join(out, 'bin_medians.npy'), np.stack(meds))
    np.savez_compressed(os.path.join(out, 'events.npz'), **{k: np.concatenate(v) for k, v in evs.items()})
    counts = R.groupby(['dataset', 'role']).agg(records=('record', 'size'), subjects=('group', 'nunique'),
                                                valid_beats=('n_valid', 'sum'), valid_hours=('valid_s', lambda s: round(s.sum() / 3600, 3)))
    man = dict(version=version, created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               fs=F.FS, band_hz=F.BAND, frames=dict(win=F.WIN, hop=F.HOP, nfft=F.NFFT, bins=[F.BINS.start, F.BINS.stop]),
               int16_target=INT16_TARGET, n_records=len(R), n_samples=int(n_tot),
               detector_sha256=qrs.FROZEN_SHA256, laws_sha256=laws.pop(), split_table_sha256=splits.table_hash(T[
                   ['dataset', 'subject', 'group', 'record', 'site', 'role', 'duration_s', 'n_rpeaks', 'hr_median']]),
               files={f: sha256(os.path.join(out, f)) for f in ('pcg_bp_1k_int16.npy', 'bin_medians.npy', 'records.csv', 'events.npz')},
               counts={'%s/%s' % k: v for k, v in counts.to_dict('index').items()})
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1, default=str)
    print('pack %s: %d records, %.1f h, %.0f MB, %.0f s' % (version, len(R), n_tot / F.FS / 3600,
                                                           os.path.getsize(pcg_path) / 1e6, time.time() - t0))
    print(counts.to_string())
    return man


def load(folder):
    """Colab side: memory-mapped PCG, record table, medians, events; checks every hash in the manifest."""
    man = json.load(open(os.path.join(folder, 'manifest.json')))
    for f, h in man['files'].items():
        assert sha256(os.path.join(folder, f)) == h, 'hash mismatch: %s' % f
    pcg = np.load(os.path.join(folder, 'pcg_bp_1k_int16.npy'), mmap_mode='r')
    R = pd.read_csv(os.path.join(folder, 'records.csv'))
    med = np.load(os.path.join(folder, 'bin_medians.npy'))
    ev = dict(np.load(os.path.join(folder, 'events.npz')))
    assert len(R) == len(med) == man['n_records'] and len(pcg) == man['n_samples']
    return man, pcg, R, med, ev


def record_signal(pcg, R, k):
    r = R.iloc[k]
    return pcg[int(r.offset):int(r.offset) + int(r.n_samples)].astype(np.float32) / np.float32(r.scale)


# ------------------------------------------------------------------ ECG targets for W (added 17 Oct 2026)
ECG_FS = 250
ECG_INT16_PER_UNIT = 8000.0              # one unit = the record's median R amplitude; clipped at about +/- 4 units


def _ecg_one(row):
    """The teacher lead of one record: 0.5 to 40 Hz (SOS, pole check), QRS made positive, scaled to the median R amplitude
    at the teacher beats, resampled to 250 Hz, length = PCG samples at 1 kHz // 4 (sample k at k / 250 s)."""
    import json as _json, sys as _sys
    import scipy.signal as sg
    if LOCAL['pcgecg_code'] not in _sys.path:
        _sys.path.insert(0, LOCAL['pcgecg_code'])
    from pcgecg import readers as Rd, signals as Sg
    from extstudy.config import LEAD_ORDER
    b = np.load(os.path.join(LOCAL['beats'], row['dataset'], row['record'] + '.npz'))
    lead = str(b['lead'])
    if row['dataset'] in LEAD_ORDER:
        rec = [r for r in Rd.load(tuple(_json.loads(row['key']))) if r.record == row['record']][0]
        pool = {rec.ecg_lead: rec.ecg, **rec.ecg_alt}
        e500 = Sg.resample(pool[lead], rec.fs_ecg, 500)
    else:
        e500 = np.asarray(np.load(os.path.join(LOCAL['harmonised'], row['dataset'], row['record'] + '.npz'))['ecg'], np.float64)
    e500 = np.nan_to_num(np.asarray(e500, np.float64))
    sos = sg.butter(2, [0.5, 40.0], btype='bandpass', fs=500, output='sos')
    assert np.abs(sg.sos2zpk(sos)[1]).max() < 0.9999
    x = sg.sosfiltfilt(sos, e500) * float(row['polarity'] or 1.0)
    beats = np.asarray(b['beats'], int)
    peaks = [np.max(x[max(0, p - 20):p + 21]) for p in beats if 0 <= p < len(x)]
    amp = float(np.median(peaks)) if len(peaks) >= 3 and np.median(peaks) > 0 else float(np.percentile(np.abs(x), 99))
    y = sg.resample_poly(x / (amp + 1e-12), 1, 2)
    n = int(row['n_samples']) // 4
    out = np.zeros(n, np.float32)
    out[:min(n, len(y))] = y[:n]
    q = np.clip(np.round(out * ECG_INT16_PER_UNIT), -32767, 32767).astype(np.int16)
    return row['record_id'], q, dict(lead=lead, polarity=float(row['polarity'] or 1.0), r_amplitude=amp,
                                     n_short=int(max(0, n - len(y))))


def build_ecg(version='paired_v1', workers=12):
    """Adds ecg_250_int16.npy, ecg_records.csv and ecg_manifest.json beside the pack; the pack's own files and its
    manifest stay untouched (their hashes are frozen in the pre-registration)."""
    from multiprocessing import Pool
    out = os.path.join(LOCAL['phd'], 'derived', 'corpus', version)
    R = pd.read_csv(os.path.join(out, 'records.csv'))
    A = pd.read_csv(os.path.join(LOCAL['beats'], 'audit_v2.csv'), low_memory=False)[['dataset', 'record', 'polarity']]
    K = pd.read_csv(LOCAL['audit_table'], low_memory=False)[['dataset', 'record', 'key']]
    R2 = R.merge(A, on=['dataset', 'record'], how='left').merge(K, on=['dataset', 'record'], how='left')
    assert len(R2) == len(R) and R2['record_id'].tolist() == R['record_id'].tolist()
    t0 = time.time()
    with Pool(workers) as pool:
        res = pool.map(_ecg_one, R2.to_dict('records'), chunksize=4)
    res.sort(key=lambda r: r[0])
    assert [r[0] for r in res] == R['record_id'].tolist()
    big = np.concatenate([q for _, q, _ in res])
    rows, off = [], 0
    for (k, q, info), (_, r) in zip(res, R.iterrows()):
        assert len(q) == int(r.n_samples) // 4
        rows.append(dict(record_id=k, dataset=r.dataset, record=r.record, ecg_offset=off, ecg_n=len(q), **info))
        off += len(q)
    np.save(os.path.join(out, 'ecg_250_int16.npy'), big)
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(out, 'ecg_records.csv'), index=False)
    man = dict(version=version, created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), fs=ECG_FS,
               int16_per_unit=ECG_INT16_PER_UNIT, band_hz=[0.5, 40.0],
               alignment='ECG sample k at k / 250 s; spectrogram frame j (centre j / 250 + 0.016 s) pairs with ECG sample j + 4',
               pack_manifest_sha256=sha256(os.path.join(out, 'manifest.json')),
               files={f: sha256(os.path.join(out, f)) for f in ('ecg_250_int16.npy', 'ecg_records.csv')},
               n_records=len(E), n_samples=int(len(big)))
    json.dump(man, open(os.path.join(out, 'ecg_manifest.json'), 'w'), indent=1)
    print('ECG targets: %d records, %.1f h, %.0f MB, %.0f s; records with ECG shorter than PCG: %d' % (
        len(E), len(big) / ECG_FS / 3600, os.path.getsize(os.path.join(out, 'ecg_250_int16.npy')) / 1e6, time.time() - t0,
        int((E['n_short'] > 0).sum())))
    return man


def load_ecg(folder):
    """Colab side: memory-mapped ECG targets, checked against ecg_manifest.json and tied to the pack's manifest."""
    man = json.load(open(os.path.join(folder, 'ecg_manifest.json')))
    for f, h in man['files'].items():
        assert sha256(os.path.join(folder, f)) == h, 'hash mismatch: %s' % f
    assert sha256(os.path.join(folder, 'manifest.json')) == man['pack_manifest_sha256'], 'ECG file belongs to another pack'
    return man, np.load(os.path.join(folder, 'ecg_250_int16.npy'), mmap_mode='r'), pd.read_csv(os.path.join(folder, 'ecg_records.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='paired_v1')
    ap.add_argument('--ecg', action='store_true', help='add the ECG targets for W to an existing pack')
    a = ap.parse_args()
    build_ecg(a.version) if a.ecg else build(a.version)
