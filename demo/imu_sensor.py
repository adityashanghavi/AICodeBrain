"""
imu_sensor.py
Inertial Measurement Unit (IMU) sensor processing demo.
Simulates 3-axis accelerometer and gyroscope data for three activity states:
still, walking, and running.
Computes RMS acceleration, jerk, step detection, and step cadence.
Plots multi-axis sensor traces, step markers, and activity-level comparisons.
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SAMPLE_RATE = 100      # Hz — typical consumer/research IMU output rate
GRAVITY     = 9.81     # m/s²

# Per-activity motion parameters
ACTIVITY_PARAMS = {
    'still': {
        'accel_noise': 0.05,   # m/s²  — sensor baseline noise floor
        'step_freq':   0.0,    # Hz    — no periodic motion
        'step_amp':    0.0,    # m/s²  — no step impulse
        'gyro_noise':  0.01,   # rad/s — gyro thermal noise
        'gyro_amp':    0.0,    # rad/s — no rotation
    },
    'walking': {
        'accel_noise': 0.15,
        'step_freq':   1.8,    # ~108 steps/min
        'step_amp':    2.5,
        'gyro_noise':  0.02,
        'gyro_amp':    0.8,
    },
    'running': {
        'accel_noise': 0.25,
        'step_freq':   2.8,    # ~168 steps/min
        'step_amp':    6.0,
        'gyro_noise':  0.04,
        'gyro_amp':    2.0,
    },
}

COLORS = {'still': '#3498db', 'walking': '#2ecc71', 'running': '#e74c3c'}
AXIS_COLORS = {'x': '#e74c3c', 'y': '#f39c12', 'z': '#3498db'}


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_imu(duration_sec=10, activity='still'):
    """Simulate accelerometer (m/s²) and gyroscope (rad/s) samples."""
    p   = ACTIVITY_PARAMS[activity]
    n   = duration_sec * SAMPLE_RATE
    f   = p['step_freq']
    amp = p['step_amp']
    ga  = p['gyro_amp']
    gn  = p['gyro_noise']
    an  = p['accel_noise']

    accel = {'x': [], 'y': [], 'z': []}
    gyro  = {'x': [], 'y': [], 'z': []}

    for i in range(n):
        t = i / SAMPLE_RATE

        # Vertical (z): gravity + foot-strike impulse (half-rectified sine gives
        # one peak per stride cycle, so detected cadence matches step_freq)
        az = GRAVITY + amp * max(0.0, math.sin(2 * math.pi * f * t)) + random.gauss(0, an)

        # Forward (x): smaller oscillation at step frequency, in-phase
        ax = 0.35 * amp * math.sin(2 * math.pi * f * t + 0.4) + random.gauss(0, an)

        # Lateral (y): half-frequency lateral sway
        ay = 0.15 * amp * math.sin(2 * math.pi * (f / 2) * t) + random.gauss(0, an)

        accel['x'].append(ax)
        accel['y'].append(ay)
        accel['z'].append(az)

        # Gyroscope: pitch/roll oscillate at step frequency, yaw drifts slowly
        gyro['x'].append(ga * math.sin(2 * math.pi * f * t) + random.gauss(0, gn))
        gyro['y'].append(0.5 * ga * math.cos(2 * math.pi * f * t) + random.gauss(0, gn))
        gyro['z'].append(0.05 * ga * math.sin(2 * math.pi * 0.1 * t) + random.gauss(0, gn))

    return accel, gyro


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def accel_magnitude(accel):
    """Resultant acceleration magnitude at each sample."""
    return [
        math.sqrt(accel['x'][i]**2 + accel['y'][i]**2 + accel['z'][i]**2)
        for i in range(len(accel['x']))
    ]


def rms(values):
    """Root-mean-square of a signal."""
    return math.sqrt(sum(v**2 for v in values) / len(values))


def compute_jerk(magnitude, fs=SAMPLE_RATE):
    """Jerk = rate of change of acceleration (m/s³).  Indicates impact sharpness."""
    return [(magnitude[i] - magnitude[i - 1]) * fs for i in range(1, len(magnitude))]


def detect_steps(accel, fs=SAMPLE_RATE, min_gap_sec=0.25):
    """
    Detect step events from the vertical (z) acceleration after removing gravity.
    Returns list of dicts with time and peak amplitude.
    """
    vertical = [v - GRAVITY for v in accel['z']]
    min_gap  = int(min_gap_sec * fs)
    threshold = 0.8  # m/s² above de-gravitated baseline

    steps, last = [], -min_gap
    for i in range(1, len(vertical) - 1):
        if (vertical[i] > threshold and
                vertical[i] > vertical[i - 1] and
                vertical[i] > vertical[i + 1] and
                i - last >= min_gap):
            steps.append({'index': i, 'time': i / fs, 'amplitude': vertical[i]})
            last = i
    return steps


def step_cadence(steps, duration_sec):
    """Steps per minute."""
    if not steps or duration_sec == 0:
        return 0.0
    return round(len(steps) / duration_sec * 60, 1)


def gyro_rms(gyro):
    """Combined angular velocity RMS across all axes."""
    combined = [
        math.sqrt(gyro['x'][i]**2 + gyro['y'][i]**2 + gyro['z'][i]**2)
        for i in range(len(gyro['x']))
    ]
    return rms(combined)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, duration_sec=10):
    activities = list(results.keys())
    t_full = [i / SAMPLE_RATE for i in range(duration_sec * SAMPLE_RATE)]
    t_jerk = t_full[1:]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('IMU Sensor Analysis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    # Row 0: 3-axis accelerometer traces (one per activity)
    for col, act in enumerate(activities):
        ax = fig.add_subplot(gs[0, col])
        accel = results[act]['accel']
        for axis in ('x', 'y', 'z'):
            ax.plot(t_full, accel[axis], color=AXIS_COLORS[axis],
                    linewidth=0.6, alpha=0.8, label=axis.upper())
        ax.set_title(f'{act.capitalize()} Accelerometer', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Accel (m/s²)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_facecolor('#f8f9fa')

    # Row 1: Magnitude + step markers (one per activity)
    for col, act in enumerate(activities):
        ax = fig.add_subplot(gs[1, col])
        mag   = results[act]['magnitude']
        steps = results[act]['steps']
        ax.plot(t_full, mag, color=COLORS[act], linewidth=0.8, alpha=0.85, label='|a|')
        if steps:
            sx = [s['time'] for s in steps]
            sy = [mag[s['index']] for s in steps]
            ax.scatter(sx, sy, color='navy', s=18, zorder=5, label='step')
        ax.set_title(f'{act.capitalize()} Magnitude & Steps', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('|a| (m/s²)', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_facecolor('#f8f9fa')

    # Row 2: Summary comparison bars
    metrics = {
        'RMS Accel (m/s²)':    [results[a]['rms_accel']   for a in activities],
        'Peak Jerk (m/s³)':    [results[a]['peak_jerk']   for a in activities],
        'Cadence (steps/min)': [results[a]['cadence']     for a in activities],
    }
    for col, (title, vals) in enumerate(metrics.items()):
        ax = fig.add_subplot(gs[2, col])
        bar_colors = [COLORS[a] for a in activities]
        bars = ax.bar(activities, vals, color=bar_colors, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(title.split('(')[1].rstrip(')') if '(' in title else '', fontsize=8)
        ax.set_ylim(0, max(vals) * 1.25 + 0.1)
        ax.set_facecolor('#f8f9fa')

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(42)
    DURATION = 10  # seconds per activity

    results = {}
    print("IMU Sensor Analysis")
    print("=" * 52)
    print(f"{'Activity':<10} {'RMS (m/s²)':>11} {'Peak Jerk':>11} {'Steps':>7} {'Cadence':>10} {'Gyro RMS':>10}")
    print("-" * 52)

    for activity in ('still', 'walking', 'running'):
        accel, gyro = generate_imu(duration_sec=DURATION, activity=activity)
        mag   = accel_magnitude(accel)
        jerk  = compute_jerk(mag)
        steps = detect_steps(accel)
        cad   = step_cadence(steps, DURATION)
        r     = rms(mag)
        pj    = max(abs(j) for j in jerk)
        gr    = gyro_rms(gyro)

        results[activity] = {
            'accel':      accel,
            'gyro':       gyro,
            'magnitude':  mag,
            'steps':      steps,
            'rms_accel':  round(r, 3),
            'peak_jerk':  round(pj, 3),
            'cadence':    cad,
            'gyro_rms':   round(gr, 4),
        }

        print(f"  {activity:<8} {r:>11.3f} {pj:>11.3f} {len(steps):>7} {cad:>9.1f} {gr:>10.4f}")

    print("=" * 52)
    plot_results(results, duration_sec=DURATION)
    print("\nPlot saved.")
