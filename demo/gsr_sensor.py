"""
gsr_sensor.py
Processes Galvanic Skin Response (GSR / EDA) data.
Detects tonic baseline, phasic peaks (SCRs), and computes arousal score.
Plots GSR signal with SCR markers and arousal comparison.
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAMPLE_RATE = 32


def generate_gsr(duration_sec=60, arousal_level='low'):
    n = duration_sec * SAMPLE_RATE
    baseline = {'low': 2.0, 'medium': 5.0, 'high': 10.0}.get(arousal_level, 2.0)
    scr_rate  = {'low': 0.5, 'medium': 2.0, 'high': 5.0}.get(arousal_level, 0.5)

    signal, tonic = [], baseline
    for i in range(n):
        tonic += random.gauss(0, 0.01)
        tonic = max(0.5, tonic)
        phasic = 0
        if random.random() < scr_rate / (60 * SAMPLE_RATE):
            phasic = random.uniform(0.5, 3.0)
        if signal and signal[-1] > tonic:
            phasic += (signal[-1] - tonic) * math.exp(-0.5 / SAMPLE_RATE)
        signal.append(tonic + phasic + random.gauss(0, 0.05))
    return signal


def moving_average(signal, window_sec=4):
    w = int(window_sec * SAMPLE_RATE)
    tonic = []
    for i in range(len(signal)):
        start, end = max(0, i - w // 2), min(len(signal), i + w // 2)
        tonic.append(sum(signal[start:end]) / (end - start))
    phasic = [s - t for s, t in zip(signal, tonic)]
    return tonic, phasic


def detect_scr(phasic, threshold=0.05, min_gap_sec=1.0):
    min_gap = int(min_gap_sec * SAMPLE_RATE)
    peaks, last = [], -min_gap
    for i in range(1, len(phasic) - 1):
        if (phasic[i] > threshold and
                phasic[i] > phasic[i - 1] and
                phasic[i] > phasic[i + 1] and
                i - last >= min_gap):
            peaks.append({'index': i, 'time': i / SAMPLE_RATE, 'amplitude': phasic[i]})
            last = i
    return peaks


def arousal_score(signal, peaks):
    mean_scl = sum(signal) / len(signal)
    scr_freq = len(peaks) / (len(signal) / SAMPLE_RATE / 60)
    avg_amp = sum(p['amplitude'] for p in peaks) / len(peaks) if peaks else 0
    return round(min(100, (mean_scl * 4) + (scr_freq * 8) + (avg_amp * 10)), 1)


def plot_results(results):
    levels = list(results.keys())
    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle('GSR / EDA Sensor Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Top row: one GSR time-series per arousal level
    for col, level in enumerate(levels):
        ax = fig.add_subplot(gs[0, col])
        sig = results[level]['signal']
        tonic = results[level]['tonic']
        peaks = results[level]['peaks']
        t = [i / SAMPLE_RATE for i in range(len(sig))]

        ax.plot(t, sig, color=colors[level], alpha=0.7, linewidth=0.8, label='GSR')
        ax.plot(t, tonic, color='black', linewidth=1.2, linestyle='--', alpha=0.6, label='Tonic')
        if peaks:
            ax.scatter([p['time'] for p in peaks],
                       [sig[p['index']] for p in peaks],
                       color='navy', s=18, zorder=5, label='SCR')
        ax.set_title(f'{level.capitalize()} arousal', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Conductance (µS)', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Bottom: arousal score comparison bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    scores = [results[l]['score'] for l in levels]
    bar_colors = [colors[l] for l in levels]
    bars = ax_bar.bar(levels, scores, color=bar_colors, alpha=0.85, width=0.4)
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(score), ha='center', va='bottom', fontweight='bold')
    ax_bar.set_ylim(0, 115)
    ax_bar.set_title('Arousal Score Comparison (0–100)', fontsize=11)
    ax_bar.set_ylabel('Score')
    ax_bar.set_facecolor('#f8f9fa')

    plt.show()


if __name__ == '__main__':
    random.seed(21)
    results = {}
    for level in ['low', 'medium', 'high']:
        sig = generate_gsr(duration_sec=60, arousal_level=level)
        tonic, phasic = moving_average(sig)
        peaks = detect_scr(phasic)
        score = arousal_score(sig, peaks)
        results[level] = {'signal': sig, 'tonic': tonic, 'peaks': peaks, 'score': score}
        print(f"  {level:6s} | Mean SCL: {sum(sig)/len(sig):.2f} µS | "
              f"SCRs: {len(peaks):3d} | Arousal: {score}/100")

    plot_results(results)
    print("\nPlot saved.")
