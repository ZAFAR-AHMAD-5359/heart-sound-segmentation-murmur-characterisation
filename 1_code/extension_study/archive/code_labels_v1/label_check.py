"""Visual check of the acoustic labels (08 design: hand check of 200 beats; accept if at least 95% of S1 and S2 marks
sit on the visible sound within 20 ms).

Draws 200 valid beats (stratified by dataset, at most 2 per record, seed 42) on 8 pages of 25 panels: ECG (grey),
PCG envelope (black), teacher beat (blue), S1 (red), S2 (green), and the pass-2 search windows (shaded). Writes
derived/labels_v1/check/page_<k>.png and check_beats.csv (one row per panel, with an empty 'verdict' column to fill).
Usage:  python -m extstudy.label_check
"""
import json, os, sys
import numpy as np
import pandas as pd

from extstudy.config import LOCAL
from extstudy import labels as L

N_BEATS, PER_PAGE, SEED = 200, 25, 42


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if LOCAL['pcgecg_code'] not in sys.path:
        sys.path.insert(0, LOCAL['pcgecg_code'])
    from pcgecg.signals import pcg_envelope
    base = os.path.join(LOCAL['phd'], 'derived', 'labels_v1')
    out = os.path.join(base, 'check'); os.makedirs(out, exist_ok=True)
    laws = json.load(open(os.path.join(base, 'laws.json')))
    S = pd.read_csv(os.path.join(base, 'labels_records.csv'))
    S = S[S.n_valid > 0]
    rng = np.random.default_rng(SEED)
    ds = sorted(S.dataset.unique())
    per = {d: N_BEATS // len(ds) + (1 if i < N_BEATS % len(ds) else 0) for i, d in enumerate(ds)}
    picks = []
    for d in ds:
        g = S[S.dataset == d]
        recs = g.record.values[rng.permutation(len(g))]
        for rec in recs:
            if sum(p[0] == d for p in picks) >= per[d]:
                break
            z = np.load(os.path.join(base, d, rec + '.npz'))
            v = np.flatnonzero(z['valid'])
            for i in rng.choice(v, size=min(2, len(v), per[d] - sum(p[0] == d for p in picks)), replace=False):
                picks.append((d, rec, int(i)))
    rows = []
    cache = {}
    for page in range(int(np.ceil(len(picks) / PER_PAGE))):
        fig, axs = plt.subplots(5, 5, figsize=(20, 14))
        for ax, (d, rec, i) in zip(axs.ravel(), picks[page * PER_PAGE:(page + 1) * PER_PAGE]):
            if (d, rec) not in cache:
                h = np.load(os.path.join(LOCAL['harmonised'], d, rec + '.npz'))
                cache = {(d, rec): (pcg_envelope(h['pcg']), np.asarray(h['ecg'], float))}
            env, ecg = cache[(d, rec)]
            z = np.load(os.path.join(base, d, rec + '.npz'))
            r, rn, s1, s2, hr = z['r_s'][i], z['r_next_s'][i], z['s1_s'][i], z['s2_s'][i], z['hr'][i]
            a, b = int((r - 0.15) * 500), int(min(rn + 0.1, len(env) / 500) * 500)
            t = np.arange(a, b) / 500 - r
            e = env[a:b]
            ax.plot(t * 1000, e, 'k', lw=0.8)
            eg = ecg[a:min(b, len(ecg))]
            if len(eg):
                eg = (eg - np.median(eg)) / (np.ptp(eg) + 1e-9) * np.ptp(e) * 0.5 + np.min(e)
                ax.plot(t[:len(eg)] * 1000, eg, color='0.65', lw=0.6)
            c1 = L.law_ms(laws['s1'], d, hr); c2 = L.law_ms(laws['s2'], d, hr)
            ax.axvspan(c1 - L.LBL['s1_halfwidth'], c1 + L.LBL['s1_halfwidth'], color='tab:red', alpha=0.08)
            ax.axvspan(c2 - L.LBL['s2_halfwidth'], c2 + L.LBL['s2_halfwidth'], color='tab:green', alpha=0.08)
            ax.axvline(0, color='tab:blue', lw=1); ax.axvline((s1 - r) * 1000, color='tab:red', lw=1.2)
            ax.axvline((s2 - r) * 1000, color='tab:green', lw=1.2); ax.axvline((rn - r) * 1000, color='tab:blue', lw=0.6, ls='--')
            k = len(rows)
            ax.set_title('#%d %s %s b%d %.0f bpm' % (k, d, rec[:14], i, hr), fontsize=7)
            ax.tick_params(labelsize=6)
            rows.append(dict(panel=k, page=page, dataset=d, record=rec, beat=i, hr=round(float(hr), 1),
                             s1_ms=round((s1 - r) * 1000, 1), s2_ms=round((s2 - r) * 1000, 1), verdict=''))
        fig.suptitle('Label check page %d: envelope (black), ECG (grey), beat (blue), S1 (red), S2 (green), windows shaded' % page)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'page_%d.png' % page), dpi=60); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out, 'check_beats.csv'), index=False)
    print('%d panels on %d pages in %s' % (len(rows), page + 1, out))


if __name__ == '__main__':
    main()
