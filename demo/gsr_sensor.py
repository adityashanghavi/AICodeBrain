"""
gsr_sensor.py
Processes Galvanic Skin Response (GSR / EDA) data.
Detects tonic baseline, phasic peaks (skin conductance responses),
and computes an arousal score.
"""

import random
import math

SAMPLE_RATE = 32  # Hz (GSR is low-frequency)


# --- Signal simulation ---
def generate_gsr(duration_sec=60, arousal_level='low'):
    """Simulate GSR signal in microsiemens (µS)."""
    n = duration_sec * SAMPLE_RATE
    baseline = {'low': 2.0, 'medium': 5.0, 'high': 10.0}.get(arousal_level, 2.0)
    scr_rate  = {'low': 0.5, 'medium': 2.0, 'high': 5.0}.get(arousal_level, 0.5)  # per minute

    signal = []
    tonic = baseline
    for i in range(n):
        t = i / SAMPLE_RATE
        # Slow tonic drift
        tonic += random.gauss(0, 0.01)
        tonic = max(0.5, tonic)
        # Phasic burst (SCR)
        phasic = 0
        if random.random() < scr_rate / (60 * SAMPLE_RATE):
            phasic = random.uniform(0.5, 3.0)  # µS amplitude
        # SCR decays exponentially
        if signal and signal[-1] > tonic:
            phasic += (signal[-1] - tonic) * math.exp(-0.5 / SAMPLE_RATE)
        signal.append(tonic + phasic + random.gauss(0, 0.05))
    return signal


# --- Tonic / phasic decomposition ---
def moving_average(signal, window_sec=4):
    w = int(window_sec * SAMPLE_RATE)
    tonic = []
    for i in range(len(signal)):
        start = max(0, i - w // 2)
        end = min(len(signal), i + w // 2)
        tonic.append(sum(signal[start:end]) / (end - start))
    phasic = [s - t for s, t in zip(signal, tonic)]
    return tonic, phasic


# --- SCR detection ---
def detect_scr(phasic, threshold=0.05, min_gap_sec=1.0):
    """Find skin conductance response peaks in the phasic component."""
    min_gap = int(min_gap_sec * SAMPLE_RATE)
    peaks = []
    last = -min_gap
    for i in range(1, len(phasic) - 1):
        if (phasic[i] > threshold and
                phasic[i] > phasic[i - 1] and
                phasic[i] > phasic[i + 1] and
                i - last >= min_gap):
            peaks.append({'index': i, 'time': i / SAMPLE_RATE, 'amplitude': phasic[i]})
            last = i
    return peaks


# --- Arousal score (0–100) ---
def arousal_score(signal, peaks):
    mean_scl = sum(signal) / len(signal)
    scr_freq = len(peaks) / (len(signal) / SAMPLE_RATE / 60)  # per minute
    avg_amp = sum(p['amplitude'] for p in peaks) / len(peaks) if peaks else 0
    # Normalize and combine (rough heuristic)
    score = min(100, (mean_scl * 4) + (scr_freq * 8) + (avg_amp * 10))
    return round(score, 1)


# --- Main ---
if __name__ == '__main__':
    random.seed(21)
    for level in ['low', 'medium', 'high']:
        print(f"\n--- Arousal level: {level} ---")
        sig = generate_gsr(duration_sec=60, arousal_level=level)
        tonic, phasic = moving_average(sig)
        peaks = detect_scr(phasic)
        score = arousal_score(sig, peaks)
        mean_scl = sum(sig) / len(sig)
        print(f"  Mean SCL:    {mean_scl:.2f} µS")
        print(f"  SCR count:   {len(peaks)}")
        print(f"  Arousal score: {score}/100")
