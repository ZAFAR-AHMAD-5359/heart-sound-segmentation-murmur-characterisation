"""Visual check of the acoustic labels (prereg section 6, item 2: accept if at least 95% of 200 checked beats carry
S1 and S2 marks on the visible sounds within 20 ms).

Sample: 200 valid beats of config.LABELS, stratified by dataset (as equal as possible), at most 2 per record, from
records outside any earlier check (`exclude_from`), seed SEED; a dataset with too few such records is filled from the
earlier-check records without reusing any checked beat. Blinded controls: N_CONTROL beats that the previous
label version accepted and this one rejects (at most 1 per record, as equal as possible across datasets, same
record exclusion), shuffled in among the sample so the checker cannot tell them apart; they measure whether the
checker separates good marks from bad ones and do not count towards the 95%.
Each page holds 10 panels (5 x 2): PCG envelope (black), band-passed PCG (grey, below), ECG (blue, bottom), R and
the next R (dotted), S1 (red), S2 (green), the pass-2 search windows (shaded), 20 ms grid.
Writes <LABELS>/check/: review_<page>.png, check_beats.csv (panel, page, dataset, record, beat, hr, s1_ms, s2_ms,
verdict, notes; verdict and notes empty) and check_key.csv (panel -> 'sample' or 'control'; opened only after every
verdict is in). Refuses to overwrite a sheet that already holds verdicts.
Usage:  python -m extstudy.label_check
"""
import json, os, sys
import numpy as np
import pandas as pd

from extstudy.config import LOCAL, LABELS
from extstudy import labels as L

N_BEATS, N_CONTROL, PER_PAGE, SEED = 200, 40, 10, 20261018
PREVIOUS = 'labels_v2'
EARLIER_CHECKS = ('labels_v1', 'labels_v2')


def _equal_shares(total, keys):
    return {d: total // len(keys) + (1 if i < total % len(keys) else 0) for i, d in enumerate(keys)}


def draw(base, prev_base, exclude, rng, exclude_beats=frozenset()):
    """Returns (sample, controls): lists of (dataset, record, beat). Sample records come from outside `exclude`; a
    dataset with too few such records is filled from the excluded records, never reusing a beat in `exclude_beats`
    (FOSTER, 40 records, in the v3 check)."""
    S_all = pd.read_csv(os.path.join(base, 'labels_records.csv'))
    S = S_all[~S_all.set_index(['dataset', 'record']).index.isin(exclude)]
    ds = sorted(S_all.dataset.unique())
    per = _equal_shares(N_BEATS, ds)
    sample = []
    for d in ds:
        n = 0
        fresh = S[(S.dataset == d) & (S.n_valid > 0)]
        reused = S_all[(S_all.dataset == d) & (S_all.n_valid > 0) & ~S_all.record.isin(fresh.record)]
        for g in (fresh, reused):
            for rec in g.record.values[rng.permutation(len(g))]:
                if n >= per[d]:
                    break
                v = np.flatnonzero(np.load(os.path.join(base, d, rec + '.npz'))['valid'])
                v = np.array([i for i in v if (d, rec, int(i)) not in exclude_beats], int)
                if len(v) == 0:
                    continue
                for i in rng.choice(v, size=min(2, len(v), per[d] - n), replace=False):
                    sample.append((d, rec, int(i))); n += 1
        assert n == per[d], 'not enough valid beats in %s' % d
    used = {(d, r) for d, r, _ in sample}
    removed = {}
    for d in ds:
        removed[d] = []
        for rec in S[(S.dataset == d) & (S.n_beats > 0)].record.values:
            if (d, rec) in used or not os.path.exists(os.path.join(prev_base, d, rec + '.npz')):
                continue
            new = np.load(os.path.join(base, d, rec + '.npz'))['valid'].astype(bool)
            old = np.load(os.path.join(prev_base, d, rec + '.npz'))['valid'].astype(bool)
            assert len(new) == len(old), 'beat tables differ between label versions: %s %s' % (d, rec)
            idx = np.flatnonzero(old & ~new)
            if len(idx):
                removed[d].append((rec, idx))
    controls = []
    want = _equal_shares(N_CONTROL, ds)
    short = 0
    for d in ds:                                    # datasets with few removed beats pass their share on
        k = min(want[d], len(removed[d]))
        short += want[d] - k
        want[d] = k
    for d in sorted(ds, key=lambda d: -len(removed[d])):
        extra = min(short, len(removed[d]) - want[d])
        want[d] += extra; short -= extra
    for d in ds:
        for j in rng.permutation(len(removed[d]))[:want[d]]:
            rec, idx = removed[d][j]
            controls.append((d, rec, int(rng.choice(idx))))
    return sample, controls


def panel(ax, d, rec, i, base, laws, cache):
    import scipy.signal as sg
    from pcgecg.signals import pcg_envelope
    if cache.get('key') != (d, rec):
        h = np.load(os.path.join(LOCAL['harmonised'], d, rec + '.npz'))
        fs = float(h['fs_pcg'])
        pcg = np.asarray(h['pcg'], float)
        sos = sg.butter(4, [25, 400], btype='bandpass', fs=fs, output='sos')
        cache.clear()
        cache.update(key=(d, rec), env=pcg_envelope(pcg), ecg=np.asarray(h['ecg'], float), bp=sg.sosfiltfilt(sos, pcg), fs=fs)
    env, ecg, bp, fs = cache['env'], cache['ecg'], cache['bp'], cache['fs']
    z = np.load(os.path.join(base, d, rec + '.npz'))
    rt, rn, s1, s2, hr = z['r_s'][i], z['r_next_s'][i], z['s1_s'][i], z['s2_s'][i], z['hr'][i]
    a, b = rt - 0.12, min(rn + 0.12, len(env) / L.FS)
    ia, ib = int(a * L.FS), int(b * L.FS)
    t = (np.arange(ia, ib) / L.FS - rt) * 1000
    e = env[ia:ib]
    lo, rng_ = np.min(e), np.max(e) - np.min(e) + 1e-9
    ja, jb = int(a * fs), int(b * fs)
    w = bp[ja:jb]
    ax.plot((np.arange(ja, jb) / fs - rt) * 1000, lo + rng_ * (0.5 + 0.45 * w / (np.max(np.abs(w)) + 1e-12)) - rng_ * 1.15, color='0.6', lw=0.4)
    ax.plot(t, e, 'k', lw=1.1)
    eg = ecg[ia:min(ib, len(ecg))]
    if len(eg):
        ax.plot(t[:len(eg)], lo - rng_ * 1.35 + 0.25 * rng_ * (eg - np.median(eg)) / (np.ptp(eg) + 1e-9), color='tab:blue', lw=0.7)
    c1, c2 = L.law_ms(laws['s1'], d, hr), L.law_ms(laws['s2'], d, hr)
    ax.axvspan(c1 - L.LBL['s1_halfwidth'], c1 + L.LBL['s1_halfwidth'], color='tab:red', alpha=0.07)
    ax.axvspan(c2 - L.LBL['s2_halfwidth'], c2 + L.LBL['s2_halfwidth'], color='tab:green', alpha=0.07)
    ax.axvline(0, color='tab:blue', lw=0.8, ls=':'); ax.axvline((rn - rt) * 1000, color='tab:blue', lw=0.8, ls=':')
    ax.axvline((s1 - rt) * 1000, color='tab:red', lw=1.0); ax.axvline((s2 - rt) * 1000, color='tab:green', lw=1.0)
    ax.set_xticks(np.arange(-100, (b - rt) * 1000, 100)); ax.set_xticks(np.arange(-120, (b - rt) * 1000, 20), minor=True)
    ax.grid(which='minor', axis='x', color='0.92', lw=0.5); ax.grid(which='major', axis='x', color='0.8', lw=0.6)
    ax.set_xlim(t[0], t[-1]); ax.set_yticks([])
    return dict(hr=round(float(hr), 1), s1_ms=round((s1 - rt) * 1000, 1), s2_ms=round((s2 - rt) * 1000, 1))


def main(exclude_from=EARLIER_CHECKS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if LOCAL['pcgecg_code'] not in sys.path:
        sys.path.insert(0, LOCAL['pcgecg_code'])
    base = os.path.join(LOCAL['phd'], 'derived', LABELS)
    out = os.path.join(base, 'check'); os.makedirs(out, exist_ok=True)
    sheet = os.path.join(out, 'check_beats.csv')
    if os.path.exists(sheet) and pd.read_csv(sheet, dtype={'verdict': str})['verdict'].fillna('').str.strip().ne('').any():
        raise SystemExit('%s already holds verdicts; not overwritten' % sheet)
    exclude, exclude_beats = set(), set()
    for v in exclude_from:
        C0 = pd.read_csv(os.path.join(LOCAL['phd'], 'derived', v, 'check', 'check_beats.csv'))
        exclude |= set(zip(C0.dataset, C0.record))
        exclude_beats |= set(zip(C0.dataset, C0.record, C0.beat.astype(int)))
    laws = json.load(open(os.path.join(base, 'laws.json')))
    rng = np.random.default_rng(SEED)
    sample, controls = draw(base, os.path.join(LOCAL['phd'], 'derived', PREVIOUS), exclude, rng, frozenset(exclude_beats))
    items = [(p, 'sample') for p in sample] + [(p, 'control') for p in controls]
    items = [items[j] for j in rng.permutation(len(items))]
    rows, key, cache = [], [], {}
    n_pages = int(np.ceil(len(items) / PER_PAGE))
    for page in range(n_pages):
        fig, axs = plt.subplots(5, 2, figsize=(22, 18))
        for ax in axs.ravel()[len(items[page * PER_PAGE:(page + 1) * PER_PAGE]):]:
            ax.axis('off')
        for ax, ((d, rec, i), kind) in zip(axs.ravel(), items[page * PER_PAGE:(page + 1) * PER_PAGE]):
            k = len(rows)
            info = panel(ax, d, rec, i, base, laws, cache)
            # no record name in the title: sample records give two beats and controls one, so names could unblind
            ax.set_title('#%d %s, %.0f bpm | S1 %.0f ms, S2 %.0f ms' % (k, d, info['hr'], info['s1_ms'], info['s2_ms']), fontsize=10)
            rows.append(dict(panel=k, page=page, dataset=d, record=rec, beat=i, verdict='', notes='', **info))
            key.append(dict(panel=k, set=kind))
        fig.suptitle('Label check %s, page %d: envelope (black), filtered PCG (grey, below), ECG (blue, bottom), R (dotted), '
                     'S1 red, S2 green; grid 20 ms' % (LABELS, page), fontsize=12)
        plt.tight_layout(); plt.savefig(os.path.join(out, 'review_%02d.png' % page), dpi=70); plt.close(fig)
    cols = ['panel', 'page', 'dataset', 'record', 'beat', 'hr', 's1_ms', 's2_ms', 'verdict', 'notes']
    pd.DataFrame(rows)[cols].to_csv(sheet, index=False)
    pd.DataFrame(key).to_csv(os.path.join(out, 'check_key.csv'), index=False)
    print('%d panels (%d sample, %d controls) on %d pages in %s' % (len(rows), len(sample), len(controls), n_pages, out))


def score(base=None):
    """After every verdict is in: the pass rule on the sample, and the checker's verdicts on the controls."""
    base = base or os.path.join(LOCAL['phd'], 'derived', LABELS)
    C = pd.read_csv(os.path.join(base, 'check', 'check_beats.csv'), dtype={'verdict': str})
    K = pd.read_csv(os.path.join(base, 'check', 'check_key.csv'))
    C = C.merge(K, on='panel', validate='one_to_one')
    v = C['verdict'].fillna('').str.strip().str.lower()
    assert v.isin(['ok', 'wrong']).all(), 'every panel needs a verdict, ok or wrong'
    s, c = C['set'] == 'sample', C['set'] == 'control'
    n_ok = int((v[s] == 'ok').sum())
    return dict(status='PASSED' if n_ok >= 0.95 * int(s.sum()) else 'FAILED', panels=int(s.sum()), ok=n_ok,
                controls=int(c.sum()), controls_wrong=int((v[c] == 'wrong').sum()),
                by_dataset={d: '%d/%d ok' % ((g['verdict'].str.lower() == 'ok').sum(), len(g)) for d, g in C[s].groupby('dataset')})


if __name__ == '__main__':
    main()
