"""Recompute the synchronisation metrics and flags from the harmonised arrays (no re-reading of raw data),
so the audit table always matches the current rule in signals.py and audit.py."""
import json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from pcgecg import signals as S

OUT = 'E:/NUST MS Thesis/PhD Level Research/derived'
SYNC_EFFECT, SYNC_LAG_HALFWIDTH, SYNC_IQR = 1.0, 60, 80


def one(args):
    d, rec = args
    z = np.load(os.path.join(OUT, 'harmonised', d, rec + '.npz'))
    env = S.pcg_envelope(z['pcg']); rp = z['rpeaks']; rp = rp[rp < len(env)]
    m, _ = S.sync_metrics(env, rp)
    return rec, d, m


def main(datasets):
    p = os.path.join(OUT, 'audit', 'audit_records.csv')
    A = pd.read_csv(p)
    ok = A['error'].isna() & A['dataset'].isin(datasets)
    jobs = list(zip(A.loc[ok, 'dataset'], A.loc[ok, 'record']))
    with Pool(12) as pool:
        res = pool.map(one, jobs, chunksize=4)
    for rec, d, m in res:
        i = A.index[(A.record == rec) & (A.dataset == d)]
        for k, v in m.items():
            A.loc[i, k] = v
    # final rule, calibrated on the negative control (calibrate.py): length-free lock effect >= 1.0, S1 lag within
    # 60 ms of the dataset's own centre (median lag of clean, clearly locked records), per-beat lag IQR <= 80 ms
    # centres from development data only: the HFrEF community-scenario test set never informs a rule
    held_out = A['meta_split'].astype(str).eq('community_test') if 'meta_split' in A else False
    clean = A['error'].isna() & A['ecg_ok'].astype(bool) & A['pcg_ok'].astype(bool) & (A['lock_effect'] >= 1.0) & ~held_out
    centre = A[clean].groupby('dataset')['s1_lag_ms'].median().to_dict()
    json.dump(centre, open(os.path.join(OUT, 'audit', 'sync_lag_centres.json'), 'w'), indent=1)
    A['s1_lag_centre_ms'] = A['dataset'].map(centre)
    A.loc[ok, 'sync_ok'] = (A['lock_effect'] >= SYNC_EFFECT) & ((A['s1_lag_ms'] - A['s1_lag_centre_ms']).abs() <= SYNC_LAG_HALFWIDTH) &                            (A['perbeat_lag_iqr_ms'] <= SYNC_IQR)
    A.loc[ok, 'drift_ok'] = A['max_jump_ms'].isna() | (A['max_jump_ms'] <= 40)
    A.loc[ok, 'usable'] = A['ecg_ok'].astype(bool) & A['pcg_ok'].astype(bool) & A['sync_ok'].astype(bool) & A['drift_ok'].astype(bool)
    A.to_csv(p, index=False)
    print('refreshed %d records in %s' % (len(res), datasets))


if __name__ == '__main__':
    main(sys.argv[1].split(',') if len(sys.argv) > 1 else ['ephnogram', 'foster', 'bsslab'])
