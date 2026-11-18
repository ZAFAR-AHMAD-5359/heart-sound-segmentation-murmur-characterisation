"""Summary tables and figures from the audit. Writes derived/audit/summary_by_dataset.csv and figures."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcgecg import signals as S

OUT = 'E:/NUST MS Thesis/PhD Level Research/derived'
FIG = os.path.join(OUT, 'audit', 'figures')
NAMES = dict(ephnogram='EPHNOGRAM', senssmart='SensSmartTech', cinc2016a='PhysioNet 2016 training-a', foster='FOSTER',
             bsslab='BSSLAB (KCL)', hfref='HFrEF (Zenodo)')
ORDER = ['ephnogram', 'senssmart', 'cinc2016a', 'foster', 'bsslab', 'hfref']
COL = dict(ephnogram='#184f95', senssmart='#3d97cf', cinc2016a='#c7531a', foster='#00876a', bsslab='#8a4fa3', hfref='#e09a00')
CIRCOR_RUN = 'E:/NUST MS Thesis/Final Draft/Submitted File/Revision-R1/notebooks/Results/R1_Gating_Campaign/20261003_1236/diagnostics.csv'


def q(s, p):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return float(np.percentile(s, p)) if len(s) else np.nan


def main():
    os.makedirs(FIG, exist_ok=True)
    A = pd.read_csv(os.path.join(OUT, 'audit', 'audit_records.csv'))
    err = A[A['error'].notna()] if 'error' in A else A.iloc[0:0]
    A = A[A['error'].isna()] if 'error' in A else A
    rows = []
    for d in ORDER:
        g = A[A.dataset == d]
        if not len(g):
            continue
        u = g[g.usable]
        rows.append(dict(dataset=NAMES[d], records=len(g), subjects=g.subject.nunique(), hours=g.duration_s.sum() / 3600,
                         ecg_ok_pct=100 * g.ecg_ok.mean(), pcg_ok_pct=100 * g.pcg_ok.mean(), sync_ok_pct=100 * g.sync_ok.mean(),
                         drift_ok_pct=100 * g.drift_ok.mean(), usable_records=len(u), usable_subjects=u.subject.nunique(),
                         usable_hours=u.duration_s.sum() / 3600, usable_beats=int(u.n_rpeaks.sum()),
                         hr_median=q(u.hr_median, 50), hr_p05=q(u.hr_p05, 50), hr_p95=q(u.hr_p95, 50),
                         hr_min=q(u.hr_p05, 0), hr_max=q(u.hr_p95, 100),
                         s1_lag_median=q(u.s1_lag_ms, 50), s1_lag_q25=q(u.s1_lag_ms, 25), s1_lag_q75=q(u.s1_lag_ms, 75),
                         perbeat_iqr_median=q(u.perbeat_lag_iqr_ms, 50), drift_median=q(u.drift_ms, 50),
                         ecg_f1_median=q(g.ecg_det_vs_cons_f1_median, 50)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'audit', 'summary_by_dataset.csv'), index=False)
    pd.set_option('display.width', 250)
    print(T.round(2).to_string(index=False))
    print('errors:', len(err))
    if len(err):
        print(err[['dataset', 'record', 'error']].head(20).to_string(index=False))

    # figure 1: grand-average R-triggered PCG envelope per dataset (usable records)
    lag = (np.arange(int((S.PRE_MS + S.POST_MS) / 1000 * S.FS_ENV)) - int(S.PRE_MS / 1000 * S.FS_ENV)) / S.FS_ENV * 1000
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for d in ORDER:
        p = os.path.join(OUT, 'audit', 'triggered_%s.npz' % d)
        if not os.path.exists(p):
            continue
        z = np.load(p); names = list(z['names']); avg = z['avg']
        use = set(A[(A.dataset == d) & A.usable].record)
        sel = [i for i, n in enumerate(names) if n in use]
        if sel:
            m = np.nanmean(avg[sel], 0)
            ax.plot(lag, (m - np.median(m)) / (np.nanmax(m) - np.median(m)), color=COL[d], lw=1.6, label=NAMES[d])
    ax.axvline(0, color='0.4', lw=0.8, ls=':'); ax.set_xlabel('ms from the R-peak'); ax.set_ylabel('PCG envelope, scaled')
    ax.set_title('R-triggered heart-sound envelope, usable records', fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_triggered_envelope.png'), dpi=150)

    # figure 2: heart-rate distributions against CirCor
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); bins = np.arange(30, 225, 5)
    for d in ORDER:
        u = A[(A.dataset == d) & A.usable]
        if len(u):
            ax.hist(u.hr_median, bins=bins, histtype='step', lw=1.6, color=COL[d], density=True, label=NAMES[d])
    if os.path.exists(CIRCOR_RUN):
        c = pd.read_csv(CIRCOR_RUN); c = c[(c.arm == 'pcg_seg') & (c.kept == True)]
        ax.hist(c.bpm, bins=bins, color='0.75', density=True, alpha=0.6, label='CirCor target (annotated)')
    ax.set_xlabel('median heart rate per record, bpm'); ax.set_ylabel('density'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('Heart-rate coverage of the paired sources against the paediatric target', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_heart_rate.png'), dpi=150)

    # figure 3: S1 lag distributions
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); bins = np.arange(-50, 252, 6)
    for d in ORDER:
        u = A[(A.dataset == d) & A.usable]
        if len(u):
            ax.hist(u.s1_lag_ms, bins=bins, histtype='step', lw=1.6, color=COL[d], density=True, label=NAMES[d])
    ax.set_xlabel('lag of the S1 envelope peak after the R-peak, ms'); ax.set_ylabel('density'); ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_s1_lag.png'), dpi=150)
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
