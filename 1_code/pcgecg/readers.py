"""Readers for the six synchronous PCG-ECG datasets.

Every reader returns Record objects with native-rate signals, so later stages can resample once.
Keys are plain tuples (picklable) so multiprocessing workers can load records independently.
"""
from dataclasses import dataclass, field
import csv, glob, io, os, re, struct, wave, zipfile
import numpy as np

DATA = 'E:/NUST MS Thesis/PhD Level Research/Datasets'
PATHS = dict(
    # Dataset-5 and Dataset-6 are byte-identical to the copies the first audit read (E:/NUST MS Thesis/Dataset-2 and
    # E:/NUST MS Thesis/EPHONGRAM-Dataset/MAT); MAT/ was extracted from the PhysioNet zip and passes SHA256SUMS
    ephnogram=DATA + '/Dataset-6/ephnogram-a-simultaneous-electrocardiogram-and-phonocardiogram-database-1.0.0/MAT',
    senssmart=DATA + '/Dataset-5/senssmarttech-database-of-cardiovascular-signals-synchronously-recorded-by-an-electrocardiograph-phonocardiograph-photoplethysmograph-and-accelerometer-1.0.0/WFDB',
    cinc2016a=DATA + '/Dataset-4/Dataset-Cinc2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/training-a',
    foster=DATA + '/Dataset-3/3u6yb-osfstorage-archive.zip',
    bsslab=DATA + '/Dataset-1/Data.zip',
    hfref=DATA + '/Dataset-2',
)
DATASETS = ['ephnogram', 'senssmart', 'cinc2016a', 'foster', 'bsslab', 'hfref']


@dataclass
class Record:
    dataset: str
    subject: str
    record: str
    site: str
    pcg: np.ndarray
    fs_pcg: float
    ecg: np.ndarray            # primary ECG lead, native rate
    fs_ecg: float
    ecg_lead: str
    ecg_alt: dict = field(default_factory=dict)   # other ECG leads at fs_ecg: name -> array
    extra: dict = field(default_factory=dict)     # other channels: name -> (array, fs)
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------ EPHNOGRAM
def _eph_sheet():
    p = os.path.join(PATHS['ephnogram'], 'ECGPCGSpreadsheet.csv')
    if not os.path.exists(p):                                   # PhysioNet layout: the sheet sits beside MAT/
        p = os.path.join(os.path.dirname(PATHS['ephnogram']), 'ECGPCGSpreadsheet.csv')
    rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
    return {r['Record Name']: r for r in rows if r.get('Record Name')}


def list_ephnogram():
    sheet = _eph_sheet()
    return [('ephnogram', k) for k in sorted(sheet) if os.path.exists(os.path.join(PATHS['ephnogram'], k + '.mat'))]


def load_ephnogram(key):
    import scipy.io as sio
    name = key[1]
    m = sio.loadmat(os.path.join(PATHS['ephnogram'], name + '.mat'))
    fs = float(np.asarray(m['fs']).ravel()[0])
    r = _eph_sheet()[name]
    meta = dict(scenario=r['Recording Scenario'], ecg_notes=r['ECG Notes'], pcg_notes=r['PCG Notes'],
                age=r['Age (years)'], sex=r['Gender'], duration_min=r['Record Duration (min)'])
    return [Record('ephnogram', r['Subject ID'], name, 'chest', m['PCG'].ravel().astype(np.float32), fs,
                   m['ECG'].ravel().astype(np.float32), fs, 'ECG', meta=meta)]


# ------------------------------------------------------------------ SensSmartTech
def list_senssmart():
    heads = sorted(glob.glob(os.path.join(PATHS['senssmart'], '*_pcg.hea')))
    return [('senssmart', os.path.basename(h)[:-8]) for h in heads]


def load_senssmart(key):
    import wfdb
    base = os.path.join(PATHS['senssmart'], key[1])
    p = wfdb.rdrecord(base + '_pcg'); e = wfdb.rdrecord(base + '_ecg')
    leads = {n: e.p_signal[:, i].astype(np.float32) for i, n in enumerate(e.sig_name)}
    extra = {}
    for mod in ('ppg', 'acc'):
        if os.path.exists(base + '_%s.hea' % mod):
            r = wfdb.rdrecord(base + '_' + mod)
            for i, n in enumerate(r.sig_name):
                extra['%s_%s' % (mod, n)] = (r.p_signal[:, i].astype(np.float32), float(r.fs))
    subj = key[1].split('_')[0]
    primary = 'II' if 'II' in leads else e.sig_name[0]
    return [Record('senssmart', subj, key[1], 'apex', p.p_signal[:, 0].astype(np.float32), float(p.fs),
                   leads[primary], float(e.fs), primary, ecg_alt={k: v for k, v in leads.items() if k != primary},
                   extra=extra)]


# ------------------------------------------------------------------ PhysioNet/CinC 2016 training-a
def list_cinc2016a():
    recs = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(PATHS['cinc2016a'], '*.dat')))]
    return [('cinc2016a', r) for r in recs]


def _cinc_labels():
    lab = {}
    ref = os.path.join(PATHS['cinc2016a'], 'REFERENCE.csv')
    if os.path.exists(ref):
        for row in csv.reader(open(ref)):
            lab[row[0]] = 'abnormal' if row[1].strip() == '1' else 'normal'
    return lab


def _fill_nan(x):
    """Linear interpolation over NaN samples (WFDB invalid values); returns (array, share of samples filled)."""
    x = np.asarray(x, np.float64)
    bad = ~np.isfinite(x)
    if bad.any() and (~bad).sum() >= 2:
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    elif bad.all():
        x[:] = 0.0
    return x.astype(np.float32), float(bad.mean())


def load_cinc2016a(key):
    import wfdb
    r = wfdb.rdrecord(os.path.join(PATHS['cinc2016a'], key[1]))
    i_p, i_e = r.sig_name.index('PCG'), r.sig_name.index('ECG')
    pcg, nan_p = _fill_nan(r.p_signal[:, i_p]); ecg, nan_e = _fill_nan(r.p_signal[:, i_e])
    return [Record('cinc2016a', key[1], key[1], 'unknown', pcg, float(r.fs), ecg, float(r.fs), 'ECG',
                   meta=dict(label=_cinc_labels().get(key[1], ''), subject_map='record id used; true subject map unpublished',
                             nan_share_pcg=round(nan_p, 6), nan_share_ecg=round(nan_e, 6)))]


# ------------------------------------------------------------------ FOSTER
def _foster_inner():
    outer = zipfile.ZipFile(PATHS['foster'])
    return zipfile.ZipFile(outer.open('FOSTER_dataset_CSV.zip'))   # stored member: seekable without extraction


def list_foster():
    return [('foster', n[:-4]) for n in sorted(_foster_inner().namelist()) if n.endswith('.csv')]


def load_foster(key):
    import pandas as pd
    inner = _foster_inner()
    df = pd.read_csv(inner.open(key[1] + '.csv'), dtype=np.float32)
    fs = float(round(1.0 / np.median(np.diff(df['Time'].values[:1000]))))
    sig = {c: df[c].values.astype(np.float32) for c in df.columns if c != 'Time'}
    extra = {c: (sig[c], fs) for c in ('PVDF', 'PZT', 'ACC', 'ERB')}
    return [Record('foster', key[1], key[1], 'chest', sig['PCG'], fs, sig['ECG'], fs, 'II', extra=extra)]


# ------------------------------------------------------------------ BSSLAB (LabScribe MAT)
def read_labscribe(b):
    """Tolerant reader for LabScribe MAT exports: the outer element size field is wrong, so walk elements."""
    off, out = 128, {}
    while off + 8 <= len(b):
        d, n = struct.unpack('<II', b[off:off + 8]); start = off + 8; o = start; parts = []
        if d != 14:
            break
        while o < start + n and o + 8 <= len(b):
            dd, nn = struct.unpack('<II', b[o:o + 8])
            if dd >> 16:
                parts.append((dd & 0xffff, b[o + 4:o + 4 + (dd >> 16)])); o += 8
            else:
                parts.append((dd, b[o + 8:o + 8 + nn])); o += 8 + nn + ((8 - nn % 8) % 8)
        dims = np.frombuffer(parts[1][1], '<i4'); name = parts[2][1].decode()
        dt = {9: '<f8', 2: 'u1', 7: '<f4', 5: '<i4'}[parts[3][0]]
        out[name] = np.frombuffer(parts[3][1], dt).reshape(dims[::-1])   # column-major file -> (channels, samples)
        off = start + n
    return out


BSSLAB_STAGE1 = dict(time=0, pcg=(1, 2), resp=(3, 8), ecg=(4, 5, 6, 7))          # 9-channel files
BSSLAB_STAGE1_8CH = dict(time=0, pcg=(1, 2), resp=(3, 7), ecg=(4, 5, 6))           # subject 4: 8 channels



def list_bsslab():
    z = zipfile.ZipFile(PATHS['bsslab'])
    keys = []
    for n in sorted(z.namelist()):
        m = re.match(r'Data/(\d+)/(\d)-([A-Z]+)\.mat$', n)
        if m and m.group(2) == '1':                      # stage II files carry no heart sounds
            for ch in BSSLAB_STAGE1['pcg']:
                keys.append(('bsslab', n, ch))
    return keys


def load_bsslab(key):
    z = zipfile.ZipFile(PATHS['bsslab'])
    v = read_labscribe(z.read(key[1]))
    X = v['b']
    lay = BSSLAB_STAGE1 if X.shape[0] == 9 else BSSLAB_STAGE1_8CH
    fs = float(round(1.0 / np.median(np.diff(X[0, :2000]))))
    m = re.match(r'Data/(\d+)/(\d)-([A-Z]+)\.mat$', key[1]); subj, sites = m.group(1), m.group(3)
    ecgs = {'ecg_ch%d' % c: X[c].astype(np.float32) for c in lay['ecg']}
    extra = {'resp_ch%d' % c: (X[c].astype(np.float32), fs) for c in lay['resp']}
    other = [c for c in lay['pcg'] if c != key[2]][0]
    extra['pcg_other_ch%d' % other] = (X[other].astype(np.float32), fs)
    return [Record('bsslab', subj, '%s_%s_pcg%d' % (subj, sites, key[2]), '%s_pcgch%d' % (sites, key[2]),
                   X[key[2]].astype(np.float32), fs, ecgs['ecg_ch4'], fs, 'ecg_ch4',
                   ecg_alt={k: a for k, a in ecgs.items() if k != 'ecg_ch4'}, extra=extra,
                   meta=dict(site_pair=sites, n_channels=int(X.shape[0]),
                             note='site of each stethoscope channel to confirm; ECG channel roles to confirm'))]


# ------------------------------------------------------------------ HFrEF (Zenodo 16934966)
def _hfref_zips():
    return {os.path.basename(p): p for p in glob.glob(os.path.join(PATHS['hfref'], '*.zip'))}


def _hfref_labels():
    lab = {}
    for zn, p in _hfref_zips().items():
        z = zipfile.ZipFile(p)
        for n in z.namelist():
            if n.endswith('.csv'):
                for row in csv.reader(io.StringIO(z.read(n).decode('utf-8-sig'))):
                    if len(row) >= 2:
                        lab[row[0].strip()] = float(row[1])
    return lab


def list_hfref():
    keys = []
    for zn, p in sorted(_hfref_zips().items()):
        z = zipfile.ZipFile(p)
        for n in sorted(z.namelist()):
            if n.endswith('_PCG.wav'):
                keys.append(('hfref', zn, n))
    return keys


def _read_wav(b):
    w = wave.open(io.BytesIO(b)); fs = w.getframerate(); n = w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype='<i%d' % w.getsampwidth()).astype(np.float32)
    return x / float(2 ** (8 * w.getsampwidth() - 1)), float(fs)


def load_hfref(key):
    z = zipfile.ZipFile(os.path.join(PATHS['hfref'], key[1]))
    pcg, fp = _read_wav(z.read(key[2])); ecg, fe = _read_wav(z.read(key[2].replace('_PCG.wav', '_ECG.wav')))
    base = os.path.basename(key[2])[:-8]; subj, site = base.split('_')
    split = 'community_test' if 'community' in key[1] else re.search(r'fold_\d', key[2]).group(0)
    labels = _hfref_labels()
    lvef = labels.get(base, labels.get(subj))       # the community-test label file is keyed by subject only
    return [Record('hfref', subj, base, site, pcg, fp, ecg, fe, 'Eko DUO single lead',
                   meta=dict(split=split, lvef=lvef, hfref=(lvef is not None and lvef <= 0.40)))]


LISTERS = dict(ephnogram=list_ephnogram, senssmart=list_senssmart, cinc2016a=list_cinc2016a,
               foster=list_foster, bsslab=list_bsslab, hfref=list_hfref)
LOADERS = dict(ephnogram=load_ephnogram, senssmart=load_senssmart, cinc2016a=load_cinc2016a,
               foster=load_foster, bsslab=load_bsslab, hfref=load_hfref)


def list_all(datasets=DATASETS):
    return [k for d in datasets for k in LISTERS[d]()]


def load(key):
    return LOADERS[key[0]](key)
