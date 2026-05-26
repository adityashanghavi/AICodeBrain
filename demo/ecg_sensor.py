"""
ecg_sensor.py
Electrocardiogram (ECG / EKG) signal processing demo.
Simulates PQRST waveforms for three cardiac states: resting, exercise, and stressed.
Computes heart rate, RR intervals, and HRV (RMSSD — root mean square of successive differences).
Plots ECG traces with R-peak markers, RR tachogram, and metric comparisons.
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAMPLE_RATE = 500       # Hz — standard clinical ECG sampling rate

# Cardiac state parameters
STATE_PARAMS = {
    'resting':  {'hr_bpm': 65,  'hr_variation_ms': 35, 'noise': 0.04},
    'exercise': {'hr_bpm': 115, 'hr_variation_ms': 20, 'noise': 0.14},
    'stressed': {'hr_bpm': 88,  'hr_variation_ms': 55, 'noise': 0.08},
}

COLORS = {'resting': '#2ecc71', 'exercise': '#e74c3c', 'stressed': '#f39c12'}


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _gaussian(t, center, amplitude, width):
    return amplitude * math.exp(-((t - center) ** 2) / (2 * width ** 2))


def _pqrst(t, r_time):
    """Synthesize one PQRST complex centred at r_time (all times in seconds)."""
    p  = _gaussian(t, r_time - 0.200,  0.15, 0.018)   # P wave
    q  = _gaussian(t, r_time - 0.030, -0.05, 0.008)   # Q dip
    r  = _gaussian(t, r_time,          1.40, 0.007)   # R spike
    s  = _gaussian(t, r_time + 0.040, -0.18, 0.010)   # S dip
    tw = _gaussian(t, r_time + 0.220,  0.30, 0.040)   # T wave
    return p + q + r + s + tw


def generate_ecg(duration_sec=10, state='resting'):
    """Return (signal, beat_times) — signal in millivolts, beat_times in seconds."""
    params = STATE_PARAMS[state]
    n = duration_sec * SAMPLE_RATE
    rr_base = 60.0 / params['hr_bpm']

    # Generate beat-to-beat times with physiological RR variability
    beat_times = []
    t = rr_base * 0.4
    while t < duration_sec - 0.4:
        beat_times.append(t)
        rr = rr_base + random.gauss(0, params['hr_variation_ms'] / 1000)
        rr = max(0.40, min(1.50, rr))
        t += rr

    signal = []
    for i in range(n):
        t_i = i / SAMPLE_RATE
        val = 0.0
        for r_time in beat_times:
            if abs(t_i - r_time) < 0.50:   # only evaluate near each beat
                val += _pqrst(t_i, r_time)
        val += random.gauss(0, params['noise'])
        signal.append(val)

    return signal, beat_times


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def detect_r_peaks(signal, min_gap_sec=0.35, threshold=0.5):
    """Detect R-peaks: local maxima above threshold with a minimum inter-peak gap."""
    min_gap = int(min_gap_sec * SAMPLE_RATE)
    peaks, last = [], -min_gap
    for i in range(1, len(signal) - 1):
        if (signal[i] > threshold and
                signal[i] > signal[i - 1] and
                signal[i] > signal[i + 1] and
                i - last >= min_gap):
            peaks.append({'index': i, 'time': i / SAMPLE_RATE, 'amplitude': signal[i]})
            last = i
    return peaks


def rr_intervals(peaks):
    """RR intervals in milliseconds between consecutive R-peaks."""
    return [
        round((peaks[i + 1]['time'] - peaks[i]['time']) * 1000, 1)
        for i in range(len(peaks) - 1)
    ]


def heart_rate(peaks):
    """Mean heart rate in BPM derived from detected R-peaks."""
    if len(peaks) < 2:
        return 0.0
    elapsed = peaks[-1]['time'] - peaks[0]['time']
    return round((len(peaks) - 1) / elapsed * 60, 1) if elapsed > 0 else 0.0


def rmssd(rr_ms):
    """HRV metric: Root Mean Square of Successive RR Differences (ms)."""
    if len(rr_ms) < 2:
        return 0.0
    diffs = [(rr_ms[i + 1] - rr_ms[i]) ** 2 for i in range(len(rr_ms) - 1)]
    return round(math.sqrt(sum(diffs) / len(diffs)), 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, duration_sec=10):
    states   = list(results.keys())
    t_full   = [i / SAMPLE_RATE for i in range(duration_sec * SAMPLE_RATE)]
    # Waveform panels show the first 5 seconds for readability
    t_view   = [t for t in t_full if t <= 5.0]
    n_view   = len(t_view)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('ECG / EKG Signal Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: ECG waveform traces (first 5 s) with R-peak markers
    for col, state in enumerate(states):
        ax    = fig.add_subplot(gs[0, col])
        sig   = results[state]['signal']
        peaks = results[state]['peaks']
        ax.plot(t_view, sig[:n_view], color=COLORS[state], linewidth=0.7)
        rp_in_view = [p for p in peaks if p['time'] <= 5.0]
        if rp_in_view:
            ax.scatter([p['time'] for p in rp_in_view],
                       [sig[p['index']] for p in rp_in_view],
                       color='navy', s=20, zorder=5, label='R-peak')
        ax.set_title(f'{state.capitalize()} ECG', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude (mV)', fontsize=8)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Row 1: RR interval tachogram (beat-to-beat variability)
    for col, state in enumerate(states):
        ax    = fig.add_subplot(gs[1, col])
        rr_ms = results[state]['rr_ms']
        if rr_ms:
            ax.bar(range(1, len(rr_ms) + 1), rr_ms,
                   color=COLORS[state], alpha=0.75, width=0.7)
            ax.axhline(sum(rr_ms) / len(rr_ms), color='black',
                       linewidth=1.2, linestyle='--', alpha=0.7, label='mean RR')
        ax.set_title(f'{state.capitalize()} RR Tachogram', fontsize=9)
        ax.set_xlabel('Beat number', fontsize=8)
        ax.set_ylabel('RR interval (ms)', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Row 2: Summary metric bar charts
    metrics = {
        'Heart Rate (BPM)': ([results[s]['hr_bpm']   for s in states], '.1f'),
        'RMSSD — HRV (ms)': ([results[s]['rmssd_ms'] for s in states], '.1f'),
        'Detected R-peaks':  ([results[s]['n_peaks']  for s in states], 'd'),
    }
    for col, (title, (vals, fmt)) in enumerate(metrics.items()):
        ax         = fig.add_subplot(gs[2, col])
        bar_colors = [COLORS[s] for s in states]
        bars       = ax.bar(states, vals, color=bar_colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    format(v, fmt), ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.30 + 0.1)
        ax.set_facecolor('#f8f9fa')

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(17)
    DURATION = 10   # seconds per state

    results = {}
    print("ECG Signal Analysis")
    print("=" * 55)
    print(f"{'State':<10} {'HR (BPM)':>10} {'RMSSD (ms)':>12} {'R-peaks':>9}")
    print("-" * 55)

    for state in ('resting', 'exercise', 'stressed'):
        signal, _ = generate_ecg(duration_sec=DURATION, state=state)
        peaks     = detect_r_peaks(signal)
        rr_ms     = rr_intervals(peaks)
        hr_bpm    = heart_rate(peaks)
        hrv       = rmssd(rr_ms)

        results[state] = {
            'signal':   signal,
            'peaks':    peaks,
            'rr_ms':    rr_ms,
            'hr_bpm':   hr_bpm,
            'rmssd_ms': hrv,
            'n_peaks':  len(peaks),
        }

        print(f"  {state:<8} {hr_bpm:>10.1f} {hrv:>12.1f} {len(peaks):>9}")

    print("=" * 55)
    plot_results(results, duration_sec=DURATION)
    print("\nPlot saved.")
