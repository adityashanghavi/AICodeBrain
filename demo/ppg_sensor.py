"""
ppg_sensor.py
Photoplethysmography (PPG) signal processing demo.
Simulates pulsatile optical waveforms (systolic peak + dicrotic notch) for three
conditions: resting, post-exercise, and stressed.
Computes heart rate, inter-beat intervals (IBI), pulse amplitude, and SpO2 estimate.
Plots PPG waveforms with beat markers, IBI tachogram, and condition comparisons.
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAMPLE_RATE = 100   # Hz — typical green / IR optical PPG sampling rate

# Condition parameters
CONDITION_PARAMS = {
    'resting':  {'hr_bpm': 62,  'hr_var_ms': 28,  'amplitude': 1.00, 'notch_ratio': 0.18, 'noise': 0.020, 'spo2': 98.5},
    'exercise': {'hr_bpm': 120, 'hr_var_ms': 18,  'amplitude': 1.45, 'notch_ratio': 0.09, 'noise': 0.055, 'spo2': 97.8},
    'stressed': {'hr_bpm': 86,  'hr_var_ms': 48,  'amplitude': 0.80, 'notch_ratio': 0.14, 'noise': 0.035, 'spo2': 97.5},
}

COLORS = {'resting': '#3498db', 'exercise': '#e74c3c', 'stressed': '#f39c12'}


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _gaussian(t, center, amplitude, width):
    return amplitude * math.exp(-((t - center) ** 2) / (2 * width ** 2))


def _ppg_pulse(t, beat_start, period, amplitude, notch_ratio):
    """
    Single PPG pulse composed of two Gaussians:
      - Systolic peak: sharp rise at ~20 % of the beat period
      - Dicrotic notch / secondary peak: smaller feature at ~55 % of the beat period
    """
    sys_t = beat_start + 0.20 * period
    sys_w = 0.07 * period
    dic_t = beat_start + 0.55 * period
    dic_w = 0.05 * period
    systolic = amplitude * _gaussian(t, sys_t, 1.0, sys_w)
    dicrotic = amplitude * notch_ratio * _gaussian(t, dic_t, 1.0, dic_w)
    return systolic + dicrotic


def generate_ppg(duration_sec=15, condition='resting'):
    """Return (signal, beat_starts) — signal in normalised AU, times in seconds."""
    p = CONDITION_PARAMS[condition]
    n = duration_sec * SAMPLE_RATE
    period_base = 60.0 / p['hr_bpm']

    # Generate beat onset times with physiological IBI variability
    beat_starts = []
    t = period_base * 0.3
    while t < duration_sec - period_base:
        beat_starts.append(t)
        period = period_base + random.gauss(0, p['hr_var_ms'] / 1000)
        period = max(0.38, min(1.80, period))
        t += period

    baseline = 0.08   # venous / DC offset
    signal = []
    for i in range(n):
        t_i = i / SAMPLE_RATE
        val = baseline
        for bs in beat_starts:
            period = 60.0 / p['hr_bpm']
            if -0.05 < t_i - bs < period:
                val += _ppg_pulse(t_i, bs, period, p['amplitude'], p['notch_ratio'])
        val += random.gauss(0, p['noise'])
        signal.append(val)

    return signal, beat_starts


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def detect_beats(signal, hr_bpm_estimate, threshold_sigma=0.4):
    """Detect systolic peaks as local maxima above (mean + threshold_sigma × std)."""
    mean_v    = sum(signal) / len(signal)
    std_v     = math.sqrt(sum((x - mean_v) ** 2 for x in signal) / len(signal))
    threshold = mean_v + threshold_sigma * std_v
    min_gap   = int(SAMPLE_RATE * 60.0 / hr_bpm_estimate * 0.5)

    peaks, last = [], -min_gap
    for i in range(1, len(signal) - 1):
        if (signal[i] > threshold and
                signal[i] > signal[i - 1] and
                signal[i] > signal[i + 1] and
                i - last >= min_gap):
            peaks.append({'index': i, 'time': i / SAMPLE_RATE, 'amplitude': signal[i]})
            last = i
    return peaks


def inter_beat_intervals(beats):
    """IBI in milliseconds between consecutive systolic peaks."""
    return [
        round((beats[i + 1]['time'] - beats[i]['time']) * 1000, 1)
        for i in range(len(beats) - 1)
    ]


def heart_rate(beats):
    """Mean heart rate in BPM from detected systolic peaks."""
    if len(beats) < 2:
        return 0.0
    elapsed = beats[-1]['time'] - beats[0]['time']
    return round((len(beats) - 1) / elapsed * 60, 1) if elapsed > 0 else 0.0


def pulse_amplitude(beats, signal):
    """Mean amplitude of detected systolic peaks (AU)."""
    if not beats:
        return 0.0
    return round(sum(signal[b['index']] for b in beats) / len(beats), 3)


def ibi_rmssd(ibis):
    """HRV metric: RMSSD of inter-beat intervals (ms)."""
    if len(ibis) < 2:
        return 0.0
    diffs = [(ibis[i + 1] - ibis[i]) ** 2 for i in range(len(ibis) - 1)]
    return round(math.sqrt(sum(diffs) / len(diffs)), 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, duration_sec=15):
    conditions = list(results.keys())
    t_full     = [i / SAMPLE_RATE for i in range(duration_sec * SAMPLE_RATE)]
    # Waveform panels show the first 8 seconds for readability
    t_view     = [t for t in t_full if t <= 8.0]
    n_view     = len(t_view)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('PPG (Photoplethysmography) Signal Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: PPG waveform traces with systolic-peak markers
    for col, cond in enumerate(conditions):
        ax    = fig.add_subplot(gs[0, col])
        sig   = results[cond]['signal']
        beats = results[cond]['beats']
        ax.plot(t_view, sig[:n_view], color=COLORS[cond], linewidth=0.8, alpha=0.9)
        in_view = [b for b in beats if b['time'] <= 8.0]
        if in_view:
            ax.scatter([b['time'] for b in in_view],
                       [sig[b['index']] for b in in_view],
                       color='navy', s=18, zorder=5, label='systolic peak')
        ax.set_title(f'{cond.capitalize()} PPG', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude (AU)', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Row 1: IBI tachogram (beat-to-beat variability)
    for col, cond in enumerate(conditions):
        ax   = fig.add_subplot(gs[1, col])
        ibis = results[cond]['ibis']
        if ibis:
            ax.bar(range(1, len(ibis) + 1), ibis,
                   color=COLORS[cond], alpha=0.75, width=0.7)
            ax.axhline(sum(ibis) / len(ibis), color='black',
                       linewidth=1.2, linestyle='--', alpha=0.7, label='mean IBI')
        ax.set_title(f'{cond.capitalize()} IBI Tachogram', fontsize=9)
        ax.set_xlabel('Beat number', fontsize=8)
        ax.set_ylabel('IBI (ms)', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Row 2: Summary metric bar charts
    metrics = {
        'Heart Rate (BPM)':     [results[c]['hr_bpm']   for c in conditions],
        'Pulse Amplitude (AU)': [results[c]['amplitude'] for c in conditions],
        'SpO₂ Estimate (%)':  [results[c]['spo2']     for c in conditions],
    }
    for col, (title, vals) in enumerate(metrics.items()):
        ax         = fig.add_subplot(gs[2, col])
        bar_colors = [COLORS[c] for c in conditions]
        bars       = ax.bar(conditions, vals, color=bar_colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=9)
        y_span = max(vals) - min(vals) or 1.0
        ax.set_ylim(min(vals) - y_span * 0.3, max(vals) + y_span * 0.5)
        ax.set_facecolor('#f8f9fa')

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(33)
    DURATION = 15   # seconds per condition

    results = {}
    print("PPG Signal Analysis")
    print("=" * 62)
    print(f"{'Condition':<10} {'HR (BPM)':>10} {'IBI RMSSD':>11} {'Amplitude':>11} {'SpO2 (%)':>10}")
    print("-" * 62)

    for cond in ('resting', 'exercise', 'stressed'):
        p = CONDITION_PARAMS[cond]
        signal, _ = generate_ppg(duration_sec=DURATION, condition=cond)
        beats  = detect_beats(signal, p['hr_bpm'])
        ibis   = inter_beat_intervals(beats)
        hr     = heart_rate(beats)
        amp    = pulse_amplitude(beats, signal)
        hrv    = ibi_rmssd(ibis)
        # SpO2 in real PPG requires two wavelengths (red + IR ratio); here we
        # use the condition's physiological baseline with a small noise term.
        spo2   = round(p['spo2'] + random.gauss(0, 0.15), 1)

        results[cond] = {
            'signal':    signal,
            'beats':     beats,
            'ibis':      ibis,
            'hr_bpm':    hr,
            'amplitude': amp,
            'spo2':      spo2,
        }

        print(f"  {cond:<8} {hr:>10.1f} {hrv:>11.1f} {amp:>11.3f} {spo2:>10.1f}")

    print("=" * 62)
    plot_results(results, duration_sec=DURATION)
    print("\nPlot saved.")
