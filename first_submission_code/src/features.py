"""
Acoustic feature extraction (464 features per recording).

Part of the reproducibility package for:
  "Synthetic-ECG-gated phonocardiography for paediatric heart-murmur characterisation"

The extractor is unchanged from the analysis source, so the features computed
here are the features the study used.
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq, dct
from scipy.stats import skew, kurtosis, spearmanr
from scipy.ndimage import gaussian_filter1d
import pywt

CONFIG = {
    'target_sr': 4000, 'recording_level_n_fft': 2048, 'n_mfcc_recording': 40,
    'n_mfcc_phase': 13, 'detection_freq_low': 20, 'detection_freq_high': 800,
    'diagnosis_freq_low': 50, 'diagnosis_freq_high': 400, 'envelope_smooth_ms': 50,
    'subphase_boundaries': {'early': (0.0, 0.4), 'mid': (0.2, 0.8), 'late': (0.6, 1.0)},
    'wavelet': 'db4', 'wavelet_levels': 5,
}

def _fast_mfcc(y, sr, n_mfcc=13, n_fft=512, hop=256, n_mels=40):
    if len(y) < n_fft: y = np.pad(y, (0, n_fft - len(y)))
    f, t, Zxx = sig.stft(y.astype(np.float64), fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window='hann')
    power = np.abs(Zxx) ** 2
    fmin, fmax = 0, sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700); mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.clip(np.floor((n_fft + 1) * hz_pts / sr).astype(int), 0, n_fft // 2)
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        if center > left: fbank[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
        if right > center: fbank[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
    mel_spec = np.maximum(fbank @ power, 1e-10)
    log_mel = np.log(mel_spec)
    return dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc]

def _fast_spectral_features(y, sr, n_fft=2048):
    if len(y) < n_fft: return 0, 0, 0, 0, 0, 0
    spectrum = np.abs(rfft(y[:n_fft])) ** 2; freqs = rfftfreq(n_fft, 1.0 / sr)
    spec_sum = spectrum.sum() + 1e-10
    centroid = np.sum(freqs * spectrum) / spec_sum
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spec_sum)
    cumsum = np.cumsum(spectrum); rolloff = freqs[min(np.searchsorted(cumsum, 0.85 * cumsum[-1]), len(freqs) - 1)]
    flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)
    zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y)); rms = np.sqrt(np.mean(y ** 2))
    return centroid, bandwidth, rolloff, flatness, zcr, rms

def compute_envelope(segment, sr, fc_low, fc_high, smooth_ms=50):
    if len(segment) < 64: return np.abs(segment)
    nyq = sr / 2; low = max(fc_low / nyq, 0.01); high = min(fc_high / nyq, 0.99)
    try:
        b, a = sig.butter(4, [low, high], btype='band'); filtered = sig.filtfilt(b, a, segment)
        rectified = np.abs(filtered); fc_env = min(10 / nyq, 0.99)
        b_env, a_env = sig.butter(2, fc_env, btype='low'); envelope = sig.filtfilt(b_env, a_env, rectified)
        return gaussian_filter1d(envelope, max(1, int(smooth_ms * sr / 1000)) / 4)
    except: return np.abs(segment)

def compute_envelope_detection(seg, sr): return compute_envelope(seg, sr, CONFIG['detection_freq_low'], CONFIG['detection_freq_high'])

def compute_envelope_diagnosis(seg, sr): return compute_envelope(seg, sr, CONFIG['diagnosis_freq_low'], CONFIG['diagnosis_freq_high'])

def extract_recording_level_features(pcg, sr):
    features = {}; n_fft = CONFIG['recording_level_n_fft']; n_mfcc = CONFIG['n_mfcc_recording']
    def zeros():
        for s in ['mean','std','min','max']:
            for i in range(n_mfcc): features[f'rec_mfcc_{i}_{s}'] = 0
        for i in range(13): features[f'rec_mfcc_delta_{i}_mean'] = 0; features[f'rec_mfcc_delta_{i}_std'] = 0
        for k in ['rec_spectral_centroid_mean','rec_spectral_centroid_std','rec_spectral_bandwidth_mean','rec_spectral_bandwidth_std','rec_spectral_rolloff_mean','rec_spectral_flatness_mean','rec_zcr_mean','rec_zcr_std','rec_rms_mean','rec_rms_std']: features[k] = 0
    if len(pcg) < n_fft: zeros(); return features
    try:
        mfccs = _fast_mfcc(pcg, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop=n_fft//4, n_mels=64)
        for i in range(n_mfcc):
            features[f'rec_mfcc_{i}_mean'] = np.mean(mfccs[i]); features[f'rec_mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'rec_mfcc_{i}_min'] = np.min(mfccs[i]); features[f'rec_mfcc_{i}_max'] = np.max(mfccs[i])
        delta = np.diff(mfccs[:13], axis=1)
        for i in range(13):
            features[f'rec_mfcc_delta_{i}_mean'] = np.mean(delta[i]) if delta.shape[1] > 0 else 0
            features[f'rec_mfcc_delta_{i}_std'] = np.std(delta[i]) if delta.shape[1] > 0 else 0
        c, bw, ro, fl, zcr, rms = _fast_spectral_features(pcg, sr, n_fft)
        features['rec_spectral_centroid_mean'] = c; features['rec_spectral_centroid_std'] = 0
        features['rec_spectral_bandwidth_mean'] = bw; features['rec_spectral_bandwidth_std'] = 0
        features['rec_spectral_rolloff_mean'] = ro; features['rec_spectral_flatness_mean'] = fl
        features['rec_zcr_mean'] = zcr; features['rec_zcr_std'] = 0
        features['rec_rms_mean'] = rms; features['rec_rms_std'] = 0
    except: zeros()
    return features

def extract_soft_murmur_features(pcg, sr):
    features = {}; keys = ['adaptive_snr_low','adaptive_snr_mid','adaptive_snr_high','spectral_stability','pitch_stability','harmonic_ratio','energy_variance','low_freq_ratio','mid_freq_ratio','high_freq_ratio']
    if len(pcg) < 1024:
        for k in keys: features[k] = 0
        return features
    try:
        nyq = sr / 2; bands = {}
        for name, lo, hi in [('low',25,100),('mid',100,300),('high',300,600)]:
            b, a = sig.butter(4, [lo/nyq, hi/nyq], btype='band'); bands[name] = sig.filtfilt(b, a, pcg)
            features[f'adaptive_snr_{name}'] = np.percentile(np.abs(bands[name]),90) / (np.percentile(np.abs(bands[name]),25) + 1e-10)
        frame_len = 1024; hop = 256; n_frames = max(1, (len(pcg) - frame_len) // hop)
        fe = np.array([np.sum(pcg[i*hop:i*hop+frame_len]**2) for i in range(n_frames)])
        features['spectral_stability'] = 1 / (np.std(fe) / (np.mean(fe) + 1e-10) + 1e-10)
        features['pitch_stability'] = features['spectral_stability']; features['harmonic_ratio'] = 0.5
        features['energy_variance'] = np.std(fe) / (np.mean(fe) + 1e-10)
        total = sum(np.sum(v**2) for v in bands.values()) + 1e-10
        for name in ['low','mid','high']: features[f'{name}_freq_ratio'] = np.sum(bands[name]**2) / total
    except:
        for k in keys: features[k] = 0
    return features

def extract_phase_specific_mfccs(s1, sys, s2, dia, sr):
    features = {}; n = CONFIG['n_mfcc_phase']; pm = {}
    for pn, seg in [('s1',s1),('systole',sys),('s2',s2),('diastole',dia)]:
        if len(seg) < 64:
            for i in range(n): features[f'{pn}_mfcc_{i}'] = 0
            pm[pn] = np.zeros(n); continue
        n_fft = min(512, len(seg) - len(seg) % 2)
        try:
            mm = np.mean(_fast_mfcc(seg, sr, n_mfcc=n, n_fft=n_fft, hop=n_fft//2), axis=1)
            for i in range(n): features[f'{pn}_mfcc_{i}'] = mm[i]
            pm[pn] = mm
        except:
            for i in range(n): features[f'{pn}_mfcc_{i}'] = 0
            pm[pn] = np.zeros(n)
    try:
        features['mfcc_systole_diastole_dist'] = np.linalg.norm(pm['systole'] - pm['diastole'])
        features['mfcc_systole_s1_dist'] = np.linalg.norm(pm['systole'] - pm['s1'])
        features['mfcc_systole_s2_dist'] = np.linalg.norm(pm['systole'] - pm['s2'])
        features['mfcc_diastole_hs_dist'] = np.linalg.norm(pm['diastole'] - (pm['s1'] + pm['s2'])/2)
    except:
        for k in ['mfcc_systole_diastole_dist','mfcc_systole_s1_dist','mfcc_systole_s2_dist','mfcc_diastole_hs_dist']: features[k] = 0
    return features

def extract_diastolic_features(seg, sr):
    features = {}; ks = ['diastole_rms','diastole_peak','diastole_spectral_centroid','diastole_energy_ratio','diastole_spectral_bandwidth','diastole_spectral_rolloff','diastole_zcr']
    if len(seg) < 64:
        for k in ks: features[k] = 0
        return features
    try:
        features['diastole_rms'] = np.sqrt(np.mean(seg**2)); features['diastole_peak'] = np.max(np.abs(seg))
        c, bw, ro, fl, zcr, _ = _fast_spectral_features(seg, sr, min(512, len(seg)))
        features['diastole_spectral_centroid'] = c; features['diastole_spectral_bandwidth'] = bw
        features['diastole_spectral_rolloff'] = ro; features['diastole_zcr'] = zcr
        env = compute_envelope_detection(seg, sr); features['diastole_energy_ratio'] = np.sum(env > np.max(env)*0.2) / len(env)
    except:
        for k in ks: features[k] = 0
    return features

def extract_absolute_amplitude_features(pcg, s1, sys, s2, dia, sr):
    features = {}; rec_rms = np.sqrt(np.mean(pcg**2)) + 1e-10; rec_max = np.max(np.abs(pcg)) + 1e-10
    for name, seg in [('s1',s1),('systole',sys),('s2',s2),('diastole',dia)]:
        features[f'abs_{name}_rms'] = np.sqrt(np.mean(seg**2)) if len(seg) > 0 else 1e-10
    features['rel_systole_intensity'] = features['abs_systole_rms'] / rec_rms
    features['rel_diastole_intensity'] = features['abs_diastole_rms'] / rec_rms
    for name, seg in [('systole',sys),('diastole',dia)]:
        features[f'abs_{name}_peak'] = np.max(np.abs(seg)) if len(seg) > 0 else 1e-10
        features[f'rel_{name}_peak'] = features[f'abs_{name}_peak'] / rec_max
    noise = np.percentile(np.abs(pcg), 10) + 1e-10
    features['systole_snr'] = features['abs_systole_peak'] / noise; features['diastole_snr'] = features['abs_diastole_peak'] / noise
    features['total_murmur_energy'] = features['abs_systole_rms'] + features['abs_diastole_rms']
    return features

def extract_boundary_features(s1, sys, s2, sr):
    features = {}; bs = int(20 * sr / 1000)
    for prefix, a_, b_ in [('s1_systole', s1, sys), ('systole_s2', sys, s2)]:
        if len(a_) > bs and len(b_) > bs:
            tail, head = a_[-bs:], b_[:bs]
            features[f'{prefix}_transition_rms'] = np.sqrt(np.mean(np.concatenate([tail, head])**2))
            c = np.corrcoef(np.abs(tail), np.abs(head))[0,1] if len(tail)==len(head) else 0
            features[f'{prefix}_continuity'] = c if not np.isnan(c) else 0
        else:
            features[f'{prefix}_transition_rms'] = 0; features[f'{prefix}_continuity'] = 0
    return features

def extract_systolic_diagnosis_features(seg, sr):
    features = {}
    if len(seg) < 64:
        for k in ['timing_occupation_ratio','timing_s1_gap','timing_s2_gap','timing_holosystolic_score','shape_monotonicity','shape_convexity','shape_plateau_cv','shape_time_to_peak','shape_skewness','shape_kurtosis']: features[k] = 0
        for i in range(10): features[f'envelope_sample_{i}'] = 0
        return features
    env = compute_envelope_diagnosis(seg, sr); env_n = env / (np.max(env) + 1e-10); above = env_n > 0.1
    if np.sum(above) > 0:
        on = np.argmax(above); off = len(env_n) - np.argmax(above[::-1])
        features['timing_s1_gap'] = on / len(env_n); features['timing_s2_gap'] = (len(env_n) - off) / len(env_n)
        features['timing_occupation_ratio'] = np.sum(above) / len(env_n)
        if features['timing_occupation_ratio'] > 0.85:
            cv = np.std(env_n[above]) / (np.mean(env_n[above]) + 1e-10); features['timing_holosystolic_score'] = max(0, 1 - cv)
        else: features['timing_holosystolic_score'] = features['timing_occupation_ratio'] * 0.5
    else:
        features['timing_s1_gap'] = 0.5; features['timing_s2_gap'] = 0.5; features['timing_occupation_ratio'] = 0; features['timing_holosystolic_score'] = 0
    corr, _ = spearmanr(np.arange(len(env_n)), env_n); features['shape_monotonicity'] = corr if not np.isnan(corr) else 0
    idx = np.linspace(0, len(env_n)-1, 5).astype(int)
    features['shape_convexity'] = np.mean(np.diff(np.diff(env_n[idx])))
    features['shape_plateau_cv'] = np.std(env_n) / (np.mean(env_n) + 1e-10)
    features['shape_time_to_peak'] = np.argmax(env_n) / len(env_n)
    features['shape_skewness'] = float(skew(env_n)); features['shape_kurtosis'] = float(kurtosis(env_n))
    for i, ix in enumerate(np.linspace(0, len(env_n)-1, 10).astype(int)): features[f'envelope_sample_{i}'] = env_n[ix]
    for k in features:
        if np.isnan(features[k]): features[k] = 0
    return features

def extract_subphase_features(seg, sr):
    features = {}; n = len(seg)
    if n < 64:
        for p in ['early','mid','late']: features[f'subphase_{p}_energy'] = 0; features[f'subphase_{p}_ratio'] = 0.33
        features['subphase_energy_centroid'] = 0.5; return features
    energies = {}
    for p, (s, e) in CONFIG['subphase_boundaries'].items():
        sub = seg[int(s*n):int(e*n)]; energies[p] = np.sqrt(np.mean(sub**2)) if len(sub) > 0 else 0
        features[f'subphase_{p}_energy'] = energies[p]
    total = sum(energies.values()) + 1e-10
    for p in CONFIG['subphase_boundaries']: features[f'subphase_{p}_ratio'] = energies[p] / total
    env = compute_envelope_diagnosis(seg, sr)
    features['subphase_energy_centroid'] = np.sum(np.arange(len(env)) * env) / (np.sum(env) * len(env) + 1e-10)
    return features

def extract_cross_phase_features(s1, sys, s2, dia, sr):
    features = {}
    rms = {n: np.sqrt(np.mean(s**2)) if len(s) > 0 else 1e-10 for n, s in [('s1',s1),('systole',sys),('s2',s2),('diastole',dia)]}
    features['cross_murmur_to_s1'] = np.clip(rms['systole']/(rms['s1']+1e-10), 0.01, 100)
    features['cross_murmur_to_s2'] = np.clip(rms['systole']/(rms['s2']+1e-10), 0.01, 100)
    features['cross_murmur_to_hs'] = np.clip(rms['systole']/((rms['s1']+rms['s2'])/2+1e-10), 0.01, 100)
    features['cross_systole_to_diastole'] = np.clip(rms['systole']/(rms['diastole']+1e-10), 0.01, 100)
    features['cross_murmur_prominence'] = rms['systole']/(rms['s1']+rms['s2']+rms['diastole']+1e-10)
    features['cross_diastole_to_systole'] = np.clip(rms['diastole']/(rms['systole']+1e-10), 0.01, 100)
    return features

def extract_wavelet_features(seg, sr):
    features = {}; levels = CONFIG['wavelet_levels']
    if len(seg) < 64:
        for i in range(levels + 1): features[f'wavelet_level_{i}'] = 0
        features['wavelet_murmur_to_hs'] = 1; return features
    try:
        coeffs = pywt.wavedec(seg, CONFIG['wavelet'], level=levels)
        for i, c in enumerate(coeffs): features[f'wavelet_level_{i}'] = np.sum(c**2) / (len(c) + 1e-10)
        if len(coeffs) > 5:
            me = np.sum(coeffs[2]**2) + np.sum(coeffs[3]**2); he = np.sum(coeffs[4]**2) + np.sum(coeffs[5]**2) + 1e-10
            features['wavelet_murmur_to_hs'] = np.clip(me / he, 0.01, 100)
        else: features['wavelet_murmur_to_hs'] = 1
    except:
        for i in range(levels + 1): features[f'wavelet_level_{i}'] = 0
        features['wavelet_murmur_to_hs'] = 1
    return features

def extract_cycle_variability_features(cfl):
    features = {}
    if len(cfl) < 2:
        for k in ['cycle_rms_cv','cycle_rms_range','cycle_spectral_cv','cycle_spectral_range','cycle_timing_cv','cycle_timing_range','cycle_consistency','n_cycles']: features[k] = 0
        features['n_cycles'] = len(cfl); return features
    for name, key in [('rms','abs_systole_rms'),('spectral','systole_mfcc_0'),('timing','timing_occupation_ratio')]:
        vals = [c.get(key, 0) for c in cfl]; vals = [v for v in vals if not np.isnan(v) and v != 0]
        if len(vals) > 1:
            features[f'cycle_{name}_cv'] = np.std(vals) / (np.mean(vals) + 1e-10); features[f'cycle_{name}_range'] = np.max(vals) - np.min(vals)
        else: features[f'cycle_{name}_cv'] = 0; features[f'cycle_{name}_range'] = 0
    cvs = [features.get(f'cycle_{n}_cv', 0) for n in ['rms','spectral','timing']]
    features['cycle_consistency'] = 1 - np.mean(cvs); features['n_cycles'] = len(cfl)
    return features

def extract_all_v4_features(pcg_signal, cycles, sr):
    """cycles: list of dicts with s1/systole/s2/diastole as SAMPLE-INDEX (start,end) tuples."""
    all_features = {}
    all_features.update(extract_recording_level_features(pcg_signal, sr))
    all_features.update(extract_soft_murmur_features(pcg_signal, sr))
    cfl = []
    for cycle in cycles:
        cf = {}
        def get_seg(s, e):
            si, ei = max(0, int(s)), min(len(pcg_signal), int(e)); return pcg_signal[si:ei]
        s1 = get_seg(*cycle['s1']); sys_ = get_seg(*cycle['systole']); s2 = get_seg(*cycle['s2']); dia = get_seg(*cycle['diastole'])
        cf.update(extract_phase_specific_mfccs(s1, sys_, s2, dia, sr))
        cf.update(extract_diastolic_features(dia, sr))
        cf.update(extract_absolute_amplitude_features(pcg_signal, s1, sys_, s2, dia, sr))
        cf.update(extract_boundary_features(s1, sys_, s2, sr))
        cf.update(extract_systolic_diagnosis_features(sys_, sr))
        cf.update(extract_subphase_features(sys_, sr))
        cf.update(extract_cross_phase_features(s1, sys_, s2, dia, sr))
        cf.update(extract_wavelet_features(sys_, sr))
        d = {n: (cycle[n][1] - cycle[n][0]) / sr for n in ['s1','systole','s2','diastole']}; total = sum(d.values())
        cf['duration_systole_ratio'] = d['systole'] / (total + 1e-10); cf['duration_diastole_ratio'] = d['diastole'] / (total + 1e-10)
        cf['duration_systole_ms'] = d['systole'] * 1000; cf['duration_cycle_ms'] = total * 1000; cf['duration_heart_rate'] = 60 / (total + 1e-10)
        cfl.append(cf)
    if cfl:
        for fn in list(cfl[0].keys()):
            vals = [c[fn] for c in cfl if fn in c]; vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
            all_features[f'{fn}_median'] = np.median(vals) if vals else 0
            all_features[f'{fn}_std'] = np.std(vals) if len(vals) > 1 else 0
    all_features.update(extract_cycle_variability_features(cfl))
    return all_features
