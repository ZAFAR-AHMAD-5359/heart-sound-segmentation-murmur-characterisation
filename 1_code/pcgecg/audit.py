"""Audit and harmonise every paired PCG-ECG record.

Per record: resample (ECG 500 Hz, PCG 2 kHz), consensus R-peaks on every candidate ECG lead (best lead kept),
RR statistics, PCG quality, and R-triggered PCG-envelope synchronisation statistics. Final flags come from
refresh_sync.py, whose rule was calibrated on the negative control in calibrate.py.
Writes derived/harmonised/<dataset>/<record>.npz, derived/audit/audit_records.csv, derived/audit/triggered_<dataset>.npz.

Usage:  python -m pcgecg.audit [--datasets a,b] [--limit N] [--workers W]
"""
import argparse, csv, json, os, sys, time, traceback, zlib
from multiprocessing import Pool
import numpy as np

from pcgecg import readers as R
from pcgecg import signals as S

OUT = 'E:/NUST MS Thesis/PhD Level Research/derived'
EXTRA_KEEP = {'foster': ('PVDF', 'PZT', 'ACC', 'ERB'), 'bsslab': ('resp_ch3', 'resp_ch8'), 'senssmart': ('acc_az',)}


def process(key):
    rows, avgs = [], {}
    try:
        recs = R.load(key)
    except Exception as e:
        return [dict(dataset=key[0], key=json.dumps(key), error='load: %s' % e)], {}
    for rec in recs:
        t0 = time.time()
        try:
            dur = len(rec.pcg) / rec.fs_pcg
            cands = {rec.ecg_lead: rec.ecg}; cands.update(rec.ecg_alt)
            best = None
            scores = {}
            for lead, sig in cands.items():
                e500 = S.resample(sig, rec.fs_ecg, S.FS_ECG)
                rp, info, ec = S.consensus_rpeaks(e500)
                rrq = S.rr_stats(rp)
                score = info['det_vs_cons_f1_median'] - (rrq['frac_rr_outlier_local'] if np.isfinite(rrq['frac_rr_outlier_local']) else 1.0)
                scores[lead] = round(score, 4)
                if best is None or score > best[5]:
                    best = (lead, rp, info, ec, e500, score)
            lead, rp, info, ec, e500, _ = best
            p2k = S.resample(rec.pcg, rec.fs_pcg, S.FS_PCG)
            env = S.pcg_envelope(p2k)
            n = min(len(env), len(ec)); env = env[:n]; rp = rp[rp < n]
            sync, avg = S.sync_metrics(env, rp)
            q = S.pcg_quality(p2k, S.FS_PCG)
            rr = S.rr_stats(rp)
            row = dict(dataset=rec.dataset, subject=rec.subject, record=rec.record, site=rec.site, key=json.dumps(key),
                       duration_s=round(dur, 2), fs_pcg_native=rec.fs_pcg, fs_ecg_native=rec.fs_ecg,
                       pcg_ecg_len_ratio=round((len(rec.pcg) / rec.fs_pcg) / (len(rec.ecg) / rec.fs_ecg), 4),
                       ecg_lead_used=lead, ecg_lead_scores=json.dumps(scores), n_rpeaks=int(len(rp)),
                       ecg_inverted=info['inverted'], ecg_pairwise_f1_mean=round(info['pairwise_f1_mean'], 4),
                       ecg_pairwise_f1_min=round(info['pairwise_f1_min'], 4), ecg_frac_unanimous=round(info['frac_unanimous'], 4),
                       ecg_det_vs_cons_f1_median=round(info['det_vs_cons_f1_median'], 4),
                       ecg_det_vs_cons_f1_min=round(info['det_vs_cons_f1_min'], 4),
                       ecg_n_by_detector=json.dumps(info['n_by_detector']))
            row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rr.items()})
            row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in q.items()})
            row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sync.items()})
            row['ecg_ok'] = bool(info['det_vs_cons_f1_median'] >= 0.90 and len(rp) >= 5 and
                                 np.isfinite(rr['frac_rr_outlier_local']) and rr['frac_rr_outlier_local'] <= 0.10)
            row['pcg_ok'] = bool(q['pcg_rms'] > 0 and q['pcg_flat_frac'] < 0.5)
            lag = sync['s1_lag_ms']
            # preliminary; refresh_sync.py applies the calibrated final rule (needs the dataset's lag centre)
            row['sync_ok'] = bool(np.isfinite(sync['lock_effect']) and sync['lock_effect'] >= 1.0 and -50 <= lag <= 250
                                  and sync['perbeat_lag_iqr_ms'] <= 80)
            # the S1 lag shortens as heart rate rises (physiology), so only abrupt minute-to-minute jumps flag a fault
            row['drift_ok'] = bool(not np.isfinite(sync['max_jump_ms']) or sync['max_jump_ms'] <= 40)
            row['usable'] = bool(row['ecg_ok'] and row['pcg_ok'] and row['sync_ok'] and row['drift_ok'])
            row.update({'meta_' + k: v for k, v in rec.meta.items() if k != 'note'})
            row['seconds'] = round(time.time() - t0, 1)
            d = os.path.join(OUT, 'harmonised', rec.dataset); os.makedirs(d, exist_ok=True)
            extra = {}
            for name in EXTRA_KEEP.get(rec.dataset, ()):
                if name in rec.extra:
                    x, fs = rec.extra[name]; extra['extra_' + name] = S.resample(x, fs, S.FS_ECG)
            np.savez_compressed(os.path.join(d, rec.record + '.npz'), pcg=p2k, fs_pcg=S.FS_PCG, ecg=e500, ecg_clean=ec,
                                fs_ecg=S.FS_ECG, rpeaks=rp.astype(np.int32), ecg_lead=lead,
                                meta=json.dumps({k: str(v) for k, v in rec.meta.items()}), **extra)
            rows.append(row); avgs[rec.record] = avg
        except Exception as e:
            rows.append(dict(dataset=rec.dataset, record=rec.record, key=json.dumps(key), error='%s: %s' % (type(e).__name__, e),
                             trace=traceback.format_exc()[-600:]))
    return rows, avgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default=','.join(R.DATASETS))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    ds = a.datasets.split(',')
    keys = []
    for d in ds:
        k = R.LISTERS[d]()
        keys += k[:a.limit] if a.limit else k
    os.makedirs(os.path.join(OUT, 'audit'), exist_ok=True)
    subset = set(ds) != set(R.DATASETS)
    tag = '_test' if a.limit else ('_' + '_'.join(ds) if subset else '')
    path = os.path.join(OUT, 'audit', 'audit_records%s.csv' % tag)
    allrows, avgs = [], {d: {} for d in ds}
    t0 = time.time()
    order = sorted(keys, key=lambda k: {'ephnogram': 0, 'bsslab': 1, 'foster': 2}.get(k[0], 3))   # heavy first
    with Pool(a.workers) as pool:
        for i, (rows, av) in enumerate(pool.imap_unordered(process, order, chunksize=1)):
            allrows += rows
            for r in rows:
                if 'error' not in r:
                    avgs[r['dataset']][r['record']] = av[r['record']]
            if (i + 1) % 50 == 0 or i + 1 == len(order):
                print('%d/%d keys, %.0f s' % (i + 1, len(order), time.time() - t0), flush=True)
    cols = sorted({c for r in allrows for c in r}, key=lambda c: (c.startswith('meta_'), c in ('trace', 'error'), c))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(allrows)
    for d, av in avgs.items():
        if av:
            names = sorted(av)
            np.savez_compressed(os.path.join(OUT, 'audit', 'triggered_%s%s.npz' % (d, '_test' if a.limit else '')), names=np.array(names),
                                avg=np.stack([av[n] for n in names]))
    if subset and not a.limit:                      # merge a partial rerun into the main table
        import pandas as pd
        main_p = os.path.join(OUT, 'audit', 'audit_records.csv')
        if os.path.exists(main_p):
            M = pd.read_csv(main_p)
            keep = ~M['key'].astype(str).apply(lambda k: any(('"%s"' % d) in k or ("'%s'" % d) in k for d in ds))
            new = pd.read_csv(path)
            pd.concat([M[keep], new], ignore_index=True).to_csv(main_p, index=False)
            print('merged %d rows of %s into %s' % (len(new), ds, main_p))
    n_err = sum('error' in r for r in allrows)
    print('done: %d rows, %d errors, %.0f s -> %s' % (len(allrows), n_err, time.time() - t0, path))


if __name__ == '__main__':
    main()
