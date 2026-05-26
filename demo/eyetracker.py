"""
eyetracker.py
Processes raw gaze data from an eye-tracking sensor.
Detects fixations, saccades, and blink events.
Plots gaze scanpath and fixation duration timeline.
"""

import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCREEN_W, SCREEN_H = 1920, 1080
VELOCITY_THRESHOLD = 30


def generate_gaze_stream(n=30, fps=60):
    samples = []
    x, y = SCREEN_W / 2, SCREEN_H / 2
    for _ in range(n * fps):
        if random.random() < 0.02:
            x = random.uniform(100, SCREEN_W - 100)
            y = random.uniform(100, SCREEN_H - 100)
        else:
            x += random.gauss(0, 2)
            y += random.gauss(0, 2)
        x = max(0, min(SCREEN_W, x))
        y = max(0, min(SCREEN_H, y))
        blink = random.random() < 0.003
        samples.append({'x': x, 'y': y, 'blink': blink, 'timestamp': len(samples) / fps})
    return samples


def compute_velocity(s1, s2, screen_dpi=96, dist_cm=60):
    dx = (s2['x'] - s1['x']) / screen_dpi * 2.54
    dy = (s2['y'] - s1['y']) / screen_dpi * 2.54
    dt = s2['timestamp'] - s1['timestamp']
    if dt == 0:
        return 0
    return math.degrees(math.atan2(math.sqrt(dx**2 + dy**2), dist_cm)) / dt


def classify_samples(samples):
    labels = []
    for i, s in enumerate(samples):
        if s['blink']:
            labels.append('blink')
        elif i == 0:
            labels.append('fixation')
        else:
            v = compute_velocity(samples[i - 1], s)
            labels.append('fixation' if v < VELOCITY_THRESHOLD else 'saccade')
    return labels


def extract_fixations(samples, labels, min_duration=0.1):
    fixations, group = [], []
    for s, l in zip(samples, labels):
        if l == 'fixation':
            group.append(s)
        else:
            if group:
                dur = group[-1]['timestamp'] - group[0]['timestamp']
                if dur >= min_duration:
                    cx = sum(g['x'] for g in group) / len(group)
                    cy = sum(g['y'] for g in group) / len(group)
                    fixations.append({'x': cx, 'y': cy, 'duration': dur,
                                      'onset': group[0]['timestamp']})
                group = []
    # Flush any trailing fixation group that never hit a non-fixation sample
    if group:
        dur = group[-1]['timestamp'] - group[0]['timestamp']
        if dur >= min_duration:
            cx = sum(g['x'] for g in group) / len(group)
            cy = sum(g['y'] for g in group) / len(group)
            fixations.append({'x': cx, 'y': cy, 'duration': dur,
                              'onset': group[0]['timestamp']})
    return fixations


def summarize(samples, labels, fixations):
    blink_rate = labels.count('blink') / (samples[-1]['timestamp'] / 60)
    saccade_rate = labels.count('saccade') / samples[-1]['timestamp']
    avg_fix_dur = sum(f['duration'] for f in fixations) / len(fixations) if fixations else 0
    return {
        'total_samples': len(samples),
        'duration_sec': round(samples[-1]['timestamp'], 2),
        'fixation_count': len(fixations),
        'avg_fixation_duration_ms': round(avg_fix_dur * 1000, 1),
        'saccade_rate_hz': round(saccade_rate, 2),
        'blink_rate_per_min': round(blink_rate, 1),
    }


def plot_results(samples, labels, fixations):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Eye Tracker Analysis', fontsize=14, fontweight='bold')

    # --- Plot 1: Scanpath on screen ---
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, SCREEN_W)
    ax.set_ylim(SCREEN_H, 0)  # y-axis flipped (screen coords)

    # Draw saccade lines
    xs = [s['x'] for s in samples[::5]]
    ys = [s['y'] for s in samples[::5]]
    ax.plot(xs, ys, color='#4a90d9', alpha=0.25, linewidth=0.5)

    # Draw fixations as circles (size = duration)
    for f in fixations:
        radius = max(10, f['duration'] * 120)
        circle = mpatches.Circle((f['x'], f['y']), radius,
                                  color='#f5a623', alpha=0.6, zorder=3)
        ax.add_patch(circle)

    ax.set_title('Gaze Scanpath\n(circle size = fixation duration)', fontsize=10)
    ax.set_xlabel('Screen X (px)')
    ax.set_ylabel('Screen Y (px)')

    # --- Plot 2: Fixation duration over time ---
    ax2 = axes[1]
    onsets = [f['onset'] for f in fixations]
    durations_ms = [f['duration'] * 1000 for f in fixations]
    colors = ['#e74c3c' if d > 400 else '#2ecc71' if d < 150 else '#f39c12'
              for d in durations_ms]
    ax2.bar(onsets, durations_ms, width=0.4, color=colors, alpha=0.8)
    ax2.axhline(250, color='white', linestyle='--', linewidth=1, alpha=0.5, label='Avg ~250ms')
    ax2.set_facecolor('#2d2d2d')
    ax2.set_title('Fixation Durations Over Time', fontsize=10)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Duration (ms)')
    ax2.tick_params(colors='#cccccc')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    random.seed(42)
    print("Generating gaze data...")
    samples = generate_gaze_stream(n=30)
    labels = classify_samples(samples)
    fixations = extract_fixations(samples, labels)
    metrics = summarize(samples, labels, fixations)

    print("\n=== Eye Tracker Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    plot_results(samples, labels, fixations)
    print("\nPlot saved.")
