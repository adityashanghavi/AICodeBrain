"""
eeg_processor.py
Processes simulated EEG signals.
Computes band power (delta, theta, alpha, beta) and cognitive load index.
"""

import math
import random

SAMPLE_RATE = 256  # Hz
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}


# --- Signal generation ---
def generate_eeg(duration_sec=10, state='relaxed'):
    """Simulate an EEG signal as a sum of sinusoids + noise."""
    n = duration_sec * SAMPLE_RATE
    signal = []
    # Dominant frequencies per mental state
    dominants = {
        'relaxed':  {'alpha': 3.0, 'theta': 0.8, 'beta': 0.3},
        'focused':  {'beta': 3.0,  'alpha': 0.5, 'theta': 0.4},
        'fatigued': {'theta': 3.0, 'delta': 1.5, 'alpha': 0.4},
    }.get(state, {'alpha': 1.0})

    for i in range(n):
        t = i / SAMPLE_RATE
        val = random.gauss(0, 0.5)  # baseline noise
        for band, amp in dominants.items():
            freq = sum(BANDS[band]) / 2  # midpoint frequency
            val += amp * math.sin(2 * math.pi * freq * t)
        signal.append(val)
    return signal


# --- Band power via Welch-like method (simplified) ---
def bandpass_power(signal, low, high, fs=SAMPLE_RATE):
    """Approximate band power using DFT on windows."""
    window = 256
    powers = []
    for start in range(0, len(signal) - window, window // 2):
        chunk = signal[start:start + window]
        # Simple DFT magnitude
        total = 0
        for k in range(len(chunk)):
            freq = k * fs / window
            if low <= freq <= high:
                re = sum(chunk[j] * math.cos(2 * math.pi * k * j / window) for j in range(window))
                im = sum(chunk[j] * math.sin(2 * math.pi * k * j / window) for j in range(window))
                total += (re**2 + im**2) / window**2
        powers.append(total)
    return sum(powers) / len(powers) if powers else 0


def compute_band_powers(signal):
    return {band: bandpass_power(signal, lo, hi) for band, (lo, hi) in BANDS.items()}


# --- Cognitive load index ---
def cognitive_load_index(band_powers):
    """
    CLI = (theta + beta) / alpha
    Higher values indicate greater cognitive load.
    """
    alpha = band_powers.get('alpha', 1e-6)
    if alpha == 0:
        alpha = 1e-6
    return (band_powers['theta'] + band_powers['beta']) / alpha


# --- Main ---
if __name__ == '__main__':
    random.seed(7)
    for state in ['relaxed', 'focused', 'fatigued']:
        print(f"\n--- State: {state} ---")
        sig = generate_eeg(duration_sec=5, state=state)
        powers = compute_band_powers(sig)
        cli = cognitive_load_index(powers)
        for band, power in powers.items():
            print(f"  {band:6s}: {power:.4f}")
        print(f"  Cognitive Load Index: {cli:.3f}")
