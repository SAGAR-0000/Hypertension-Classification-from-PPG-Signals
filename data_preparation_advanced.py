import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import time

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(r'e:\study\iiit\new project\new new\data')
OUTPUT_DIR = Path(r'e:\study\iiit\new project\new new\processed_data')
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLING_RATE = 125
SEGMENT_LENGTH = 10
SEGMENT_SAMPLES = SAMPLING_RATE * SEGMENT_LENGTH

BP_THRESHOLDS = {
    'normotensive': {'sbp': (0, 120), 'dbp': (0, 80)},
    'prehypertensive': {'sbp': (120, 140), 'dbp': (80, 90)},
    'hypertensive': {'sbp': (140, 300), 'dbp': (90, 200)}
}

PPG_INDEX = 0
ABP_INDEX = 1
ECG_INDEX = 2

# === DATA LOADING ===

def load_mat_file(file_path: Path) -> List[np.ndarray]:
    try:
        mat_data = loadmat(str(file_path))
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if not data_keys: return []
        main_key = data_keys[0]
        records = mat_data[main_key]
        record_list = []
        if isinstance(records, np.ndarray):
            for record in records.flat:
                if isinstance(record, np.ndarray) and record.size > 0:
                    record_list.append(record)
        return record_list
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def extract_signals(record: np.ndarray) -> Dict[str, np.ndarray]:
    if record.shape[0] == 3:
        return {'ppg': record[PPG_INDEX, :].flatten(), 'abp': record[ABP_INDEX, :].flatten(), 'ecg': record[ECG_INDEX, :].flatten()}
    elif record.shape[1] == 3:
        return {'ppg': record[:, PPG_INDEX].flatten(), 'abp': record[:, ABP_INDEX].flatten(), 'ecg': record[:, ECG_INDEX].flatten()}
    else:
        raise ValueError(f"Invalid record shape: {record.shape}")

# === BP EXTRACTION ===

def extract_bp_from_abp(abp_signal: np.ndarray, fs: int = SAMPLING_RATE) -> Tuple[float, float]:
    abp_clean = abp_signal[np.isfinite(abp_signal)]
    if len(abp_clean) < fs: return np.nan, np.nan
    
    min_distance = int(0.5 * fs)
    try:
        peaks, _ = signal.find_peaks(abp_clean, distance=min_distance)
        troughs, _ = signal.find_peaks(-abp_clean, distance=min_distance)
        if len(peaks) < 2 or len(troughs) < 2: return np.nan, np.nan
        
        sbp = np.median(abp_clean[peaks])
        dbp = np.median(abp_clean[troughs])
        if sbp < 60 or sbp > 250 or dbp < 30 or dbp > 150 or sbp <= dbp: return np.nan, np.nan
        return sbp, dbp
    except: return np.nan, np.nan

def classify_bp(sbp: float, dbp: float) -> str:
    if np.isnan(sbp) or np.isnan(dbp): return 'invalid'
    if sbp >= 140 or dbp >= 90: return 'hypertensive'
    if (120 <= sbp < 140) or (80 <= dbp < 90): return 'prehypertensive'
    if sbp < 120 and dbp < 80: return 'normotensive'
    return 'invalid'

# === SIGNAL QUALITY & PREPROCESSING ===

def check_signal_quality(ppg_signal: np.ndarray) -> Dict[str, float]:
    clean_signal = ppg_signal[np.isfinite(ppg_signal)]
    if len(clean_signal) == 0: return {'valid': False}
    if np.var(clean_signal) < 1e-6: return {'valid': False}
    
    mean_val = np.mean(clean_signal)
    std_val = np.std(clean_signal)
    outlier_ratio = np.sum(np.abs(clean_signal - mean_val) > 3 * std_val) / len(clean_signal)
    if outlier_ratio > 0.1: return {'valid': False}
    
    return {'valid': True}

def preprocess_ppg(ppg_signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    # Clean
    signal_clean = ppg_signal.copy()
    valid_idx = np.isfinite(signal_clean)
    if np.sum(valid_idx) < len(signal_clean) * 0.8: return ppg_signal # Return raw if too damaged, quality check handles it
    if not np.all(valid_idx):
        x = np.arange(len(signal_clean))
        signal_clean[~valid_idx] = np.interp(x[~valid_idx], x[valid_idx], signal_clean[valid_idx])
    
    mean_val = np.mean(signal_clean)
    std_val = np.std(signal_clean)
    signal_clean = np.clip(signal_clean, mean_val - 3*std_val, mean_val + 3*std_val)
    
    # Filter (0.5-8 Hz)
    nyquist = 0.5 * fs
    sos = signal.butter(4, [0.5/nyquist, 8.0/nyquist], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, signal_clean)
    
    # Normalize
    m = np.mean(filtered)
    s = np.std(filtered)
    if s < 1e-10: return filtered
    return (filtered - m) / s

# === FEATURE EXTRACTION ===

def detect_ppg_peaks(ppg_signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    min_distance = int(0.4 * fs)
    peaks, _ = signal.find_peaks(ppg_signal, distance=min_distance, prominence=0.5)
    return peaks

def extract_morphological_features(ppg_signal: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
    if len(peaks) < 2:
        return {f'morph_{k}': np.nan for k in ['peak_height_mean', 'peak_height_std', 'pulse_width_mean', 'pulse_interval_mean']}
    
    peak_heights = ppg_signal[peaks]
    peak_intervals = np.diff(peaks) / SAMPLING_RATE
    
    pulse_widths = []
    for i in range(len(peaks) - 1):
        if len(ppg_signal[peaks[i]:peaks[i+1]]) > 10:
            half = peak_heights[i] / 2
            pulse_widths.append(np.sum(ppg_signal[peaks[i]:peaks[i+1]] > half) / SAMPLING_RATE)
            
    return {
        'morph_peak_height_mean': np.mean(peak_heights),
        'morph_peak_height_std': np.std(peak_heights),
        'morph_pulse_interval_mean': np.mean(peak_intervals),
        'morph_pulse_interval_std': np.std(peak_intervals),
        'morph_pulse_width_mean': np.mean(pulse_widths) if pulse_widths else np.nan
    }

def extract_physiological_features(peaks: np.ndarray, fs: int = SAMPLING_RATE) -> Dict[str, float]:
    if len(peaks) < 3:
        return {f'physio_{k}': np.nan for k in ['hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn50']}
    
    pp_intervals = np.diff(peaks) / fs * 1000 # ms
    hr = 60000 / pp_intervals
    diffs = np.diff(pp_intervals)
    
    return {
        'physio_hr_mean': np.mean(hr),
        'physio_hr_std': np.std(hr),
        'physio_rmssd': np.sqrt(np.mean(diffs ** 2)),
        'physio_sdnn': np.std(pp_intervals),
        'physio_pnn50': (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100
    }

def extract_statistical_features(ppg_signal: np.ndarray) -> Dict[str, float]:
    return {
        'stat_mean': np.mean(ppg_signal),
        'stat_std': np.std(ppg_signal),
        'stat_kurtosis': kurtosis(ppg_signal),
        'stat_skewness': skew(ppg_signal),
        'stat_min': np.min(ppg_signal),
        'stat_max': np.max(ppg_signal)
    }

# --- NEW FEATURES ---

def extract_frequency_features(ppg_signal: np.ndarray, fs: int = SAMPLING_RATE) -> Dict[str, float]:
    try:
        freqs, psd = signal.welch(ppg_signal, fs, nperseg=min(len(ppg_signal), 1024))
        psd_norm = psd / (np.sum(psd) + 1e-10)
        
        dom_freq = freqs[np.argmax(psd)]
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Bands
        vlf = np.trapz(psd[(freqs>=0)&(freqs<0.04)], freqs[(freqs>=0)&(freqs<0.04)])
        lf = np.trapz(psd[(freqs>=0.04)&(freqs<0.15)], freqs[(freqs>=0.04)&(freqs<0.15)])
        hf = np.trapz(psd[(freqs>=0.15)&(freqs<0.4)], freqs[(freqs>=0.15)&(freqs<0.4)])
        
        return {
            'freq_dominant': dom_freq,
            'freq_entropy': entropy,
            'freq_power_vlf': vlf,
            'freq_power_lf': lf,
            'freq_power_hf': hf,
            'freq_lf_hf_ratio': lf/hf if hf > 0 else 0.0
        }
    except:
        return {f'freq_{k}': np.nan for k in ['dominant', 'entropy', 'power_vlf', 'power_lf', 'power_hf', 'lf_hf_ratio']}

def extract_apg_features(ppg_signal: np.ndarray) -> Dict[str, float]:
    try:
        # APG = 2nd derivative
        vpg = np.gradient(ppg_signal)
        apg = np.gradient(vpg)
        
        features = {
            'apg_mean': np.mean(apg),
            'apg_std': np.std(apg),
            'apg_skewness': skew(apg),
            'apg_kurtosis': kurtosis(apg)
        }
        
        # Simplified Wave Analysis
        peaks, _ = signal.find_peaks(apg, distance=30)
        troughs, _ = signal.find_peaks(-apg, distance=30)
        
        if len(peaks) > 0 and len(troughs) > 0:
            a_mean = np.mean(apg[peaks])
            b_mean = np.mean(apg[troughs])
            features['apg_a_wave_mean'] = a_mean
            features['apg_b_wave_mean'] = b_mean
            features['apg_b_a_ratio'] = b_mean / a_mean if abs(a_mean) > 1e-6 else 0.0
        else:
            features['apg_a_wave_mean'] = np.nan
            features['apg_b_wave_mean'] = np.nan
            features['apg_b_a_ratio'] = np.nan
            
        return features
    except:
        return {f'apg_{k}': np.nan for k in ['mean', 'std', 'skewness', 'kurtosis', 'a_wave_mean', 'b_wave_mean', 'b_a_ratio']}

def extract_all_features(ppg_signal: np.ndarray) -> Dict[str, float]:
    peaks = detect_ppg_peaks(ppg_signal)
    
    feats = {}
    feats.update(extract_morphological_features(ppg_signal, peaks))
    feats.update(extract_physiological_features(peaks))
    feats.update(extract_statistical_features(ppg_signal))
    feats.update(extract_frequency_features(ppg_signal))
    feats.update(extract_apg_features(ppg_signal))
    return feats

# === MAIN LOOP ===

def run_processing():
    mat_files = sorted(DATA_DIR.glob('*.mat'))
    print(f"Found {len(mat_files)} files. Starting Advanced Feature Extraction...")
    
    all_features = []
    stats = {'total': 0, 'processed': 0, 'skipped': 0}
    
    start_time = time.time()
    
    for f_idx, mat_file in enumerate(mat_files):
        print(f"Processing {mat_file.name}...")
        records = load_mat_file(mat_file)
        
        for r_idx, record in enumerate(records):
            stats['total'] += 1
            try:
                sig = extract_signals(record)
                ppg, abp = sig['ppg'], sig['abp']
                
                if len(ppg) < SEGMENT_SAMPLES: continue # Skip short
                
                # Check BP
                sbp, dbp = extract_bp_from_abp(abp)
                bp_class = classify_bp(sbp, dbp)
                if bp_class == 'invalid': continuer
                
                # Segments
                num_segs = len(ppg) // SEGMENT_SAMPLES
                for s_idx in range(num_segs):
                    s_start = s_idx * SEGMENT_SAMPLES
                    s_end = s_start + SEGMENT_SAMPLES
                    
                    segment = ppg[s_start:s_end]
                    
                    if not check_signal_quality(segment)['valid']: continue
                    
                    processed = preprocess_ppg(segment)
                    
                    # Extract Features
                    feats = extract_all_features(processed)
                    
                    # Metadata
                    feats['file_name'] = mat_file.stem
                    feats['record_idx'] = r_idx
                    feats['segment_idx'] = s_idx
                    feats['sbp'] = sbp
                    feats['dbp'] = dbp
                    feats['bp_class'] = bp_class
                    
                    all_features.append(feats)
                    stats['processed'] += 1
                    
            except Exception:
                stats['skipped'] += 1
                continue
        
        # Intermediate Save every file
        if all_features:
            df = pd.DataFrame(all_features)
            # Optimize memory: float32
            cols = df.select_dtypes(include=['float64']).columns
            df[cols] = df[cols].astype(np.float32)
            
            output_file = OUTPUT_DIR / 'ppg_features_advanced.csv'
            df.to_csv(output_file, index=False)
            print(f"  Saved {len(df)} samples to {output_file}")

    print(f"\nDone! Processed {stats['processed']} / {stats['total']} records.")
    print(f"Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_processing()
