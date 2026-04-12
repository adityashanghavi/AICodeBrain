"""
eeg_processor.py
Processes simulated EEG signals.
Computes band power (delta, theta, alpha, beta) and cognitive load index.
Plots band power comparison and signal waveforms.
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 256
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}
BAND_COLORS = {
    'delta': '#9b59b6',
    'theta': '#3498db',
    'alpha': '#2ecc71',
    'beta':  '#e74c3c',
}


def generate_eeg(duration_sec=5, state='relaxed'):
    n = duration_sec * SAMPLE_RATE
    dominants = {
        'relaxed':  {'alpha': 3.0, 'theta': 0.8, 'beta': 0.3},
        'focused':  {'beta': 3.0,  'alpha': 0.5, 'theta': 0.4},
        'fatigued': {'theta': 3.0, 'delta': 1.5, 'alpha': 0.4},
    }.get(state, {'alpha': 1.0})

    signal = []
    for i in range(n):
        t = i / SAMPLE_RATE
        val = random.gauss(0, 0.5)
        for band, amp in dominants.items():
            freq = sum(BANDS[band]) / 2
            val += amp * math.sin(2 * math.pi * freq * t)
        signal.append(val)
    return signal


def bandpass_power(signal, low, high, fs=SAMPLE_RATE):
    window = min(256, len(signal))
    powers = []
    for start in range(0, len(signal) - window, window // 2):
        chunk = signal[start:start + window]
        total = 0
        for k in range(window):
            freq = k * fs / window
            if low <= freq <= high:
                re = sum(chunk[j] * math.cos(2 * math.pi * k * j / window) for j in range(window))
                im = sum(chunk[j] * math.sin(2 * math.pi * k * j / window) for j in range(window))
                total += (re**2 + im**2) / window**2
        powers.append(total)
    return sum(powers) / len(powers) if powers else 0


def compute_band_powers(signal):
    return {band: bandpass_power(signal, lo, hi) for band, (lo, hi) in BANDS.items()}


def cognitive_load_index(bp):
    alpha = max(bp.get('alpha', 1e-6), 1e-6)
    return (bp['theta'] + bp['beta']) / alpha


def plot_results(all_powers, states, signals):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.suptitle('EEG Brain Signal Analysis', fontsize=14, fontweight='bold')

    # --- Plot 1: Band power grouped bar chart ---
    ax = axes[0]
    n_states = len(states)
    n_bands = len(BANDS)
    x = range(n_states)
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, (band, color) in enumerate(BAND_COLORS.items()):
        vals = [all_powers[s][band] for s in states]
        positions = [xi + offsets[i] * width for xi in x]
        ax.bar(positions, vals, width=width, color=color, alpha=0.85, label=band.capitalize())

    ax.set_xticks(list(x))
    ax.set_xticklabels([s.capitalize() for s in states])
    ax.set_title('Band Power by Mental State', fontsize=11)
    ax.set_ylabel('Power (µV²/Hz)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_facecolor('#f8f9fa')

    # --- Plot 2: Raw waveforms (first 2s of each state) ---
    ax2 = axes[1]
    colors_state = {'relaxed': '#2ecc71', 'focused': '#e74c3c', 'fatigued': '#3498db'}
    t = [i / SAMPLE_RATE for i in range(2 * SAMPLE_RATE)]
    for state, sig in signals.items():
        ax2.plot(t, sig[:len(t)], color=colors_state[state], alpha=0.75,
                 linewidth=0.8, label=state.capitalize())

    ax2.set_title('EEG Waveforms (first 2 seconds)', fontsize=11)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (µV)')
    ax2.legend(fontsize=9)
    ax2.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    random.seed(7)
    states = ['relaxed', 'focused', 'fatigued']
    all_powers = {}
    signals = {}

    for state in states:
        print(f"\n--- State: {state} ---")
        sig = generate_eeg(duration_sec=5, state=state)
        signals[state] = sig
        powers = compute_band_powers(sig)
        all_powers[state] = powers
        cli = cognitive_load_index(powers)
        for band, power in powers.items():
            print(f"  {band:6s}: {power:.4f}")
        print(f"  Cognitive Load Index: {cli:.3f}")

    plot_results(all_powers, states, signals)
    print("\nPlot saved.")
