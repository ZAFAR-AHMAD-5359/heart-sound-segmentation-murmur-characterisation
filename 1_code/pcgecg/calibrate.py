"""Negative control for the synchronisation statistic.

For each dataset, pair the PCG envelope of one record with the R-peaks of a record from a DIFFERENT subject
(same dataset, so the same device and rate range) and compute the same lock statistics. For the HFrEF set, also
pair sites of the SAME subject: same heart and heart rate, but asynchronous recordings, the hardest null.
Writes derived/audit/sync_negative_control.csv and prints percentiles against the matched values.
"""
import csv, glob, json, os
import numpy as np
import pandas as pd
from pcgecg import signals as S

OUT = 'E:/NUST MS Thesis/PhD Level Research/derived'


def load_npz(dataset, record):
    z = np.load(os.path.join(OUT, 'harmonised', dataset, record + '.npz'), allow_pickle=False)
    return z['pcg'], z['rpeaks']


from pcgecg.refresh_sync import SYNC_EFFECT, SYNC_LAG_HALFWIDTH, SYNC_IQR


def passes(m, c):
    return bool(np.isfinite(m['lock_effect']) and m['lock_effect'] >= SYNC_EFFECT and abs(m['s1_lag_ms'] - c) <= SYNC_LAG_HALFWIDTH
                and m['perbeat_lag_iqr_ms'] <= SYNC_IQR)


def main(n_per=80, seed=7):
    rng = np.random.default_rng(seed)
    A = pd.read_csv(os.path.join(OUT, 'audit', 'audit_records.csv'))
    A = A[A['error'].isna()] if 'error' in A else A
    good = A[A.ecg_ok & A.pcg_ok]
    centre = json.load(open(os.path.join(OUT, 'audit', 'sync_lag_centres.json')))
    rows = []
    for d, g in good.groupby('dataset'):
        g = g.reset_index(drop=True)
        idx = rng.choice(len(g), size=min(n_per, len(g)), replace=False)
        for i in idx:
            a = g.loc[i]
            others = g[g.subject != a.subject]
            if len(others) == 0:
                continue
            b = others.iloc[int(rng.integers(len(others)))]
            pa, _ = load_npz(d, a.record); _, rb = load_npz(d, b.record)
            env = S.pcg_envelope(pa); n = len(env)
            rb = rb[rb < n]
            m, _ = S.sync_metrics(env, rb)
            rows.append(dict(dataset=d, kind='other_subject', record=a.record, rpeaks_from=b.record,
                             lock_t=m['lock_t'], lock_effect=m['lock_effect'], n_beats=m['n_beats_used'], s1_lag_ms=m['s1_lag_ms'], perbeat_lag_iqr_ms=m['perbeat_lag_iqr_ms'],
                             passes_rule=passes(m, centre[d])))
            if d == 'hfref':
                same = g[(g.subject == a.subject) & (g.record != a.record)]
                if len(same):
                    c = same.iloc[int(rng.integers(len(same)))]
                    _, rc = load_npz(d, c.record); rc = rc[rc < n]
                    m, _ = S.sync_metrics(env, rc)
                    rows.append(dict(dataset=d, kind='same_subject_other_site', record=a.record, rpeaks_from=c.record,
                                     lock_t=m['lock_t'], lock_effect=m['lock_effect'], n_beats=m['n_beats_used'], s1_lag_ms=m['s1_lag_ms'], perbeat_lag_iqr_ms=m['perbeat_lag_iqr_ms'],
                                     passes_rule=passes(m, centre[d])))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'audit', 'sync_negative_control.csv'), index=False)
    print('negative control: lock_t percentiles and share passing the sync rule')
    for (d, k), g in R.groupby(['dataset', 'kind']):
        mg = good[good.dataset == d]
        print('  %-10s %-24s n=%-3d null effect p95 %.2f p99 %.2f | null pass %.1f%% | matched effect p05 %.2f p50 %.2f | matched sync_ok %.1f%%' % (
            d, k, len(g), g.lock_effect.quantile(.95), g.lock_effect.quantile(.99), 100 * g.passes_rule.mean(),
            mg.lock_effect.quantile(.05), mg.lock_effect.median(), 100 * mg.sync_ok.astype(bool).mean()))


if __name__ == '__main__':
    main()
