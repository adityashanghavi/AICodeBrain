"""
eyetracker.py
Processes raw gaze data from an eye-tracking sensor.
Detects fixations, saccades, and blink events.
"""

import math
import random

# --- Simulated raw gaze stream ---
def generate_gaze_stream(n=60, fps=60):
    """Simulate n seconds of gaze data at given fps."""
    samples = []
    x, y = 960.0, 540.0  # start at screen center
    for _ in range(n * fps):
        # Drift gaze with occasional jumps (saccades)
        if random.random() < 0.02:
            x = random.uniform(100, 1820)
            y = random.uniform(100, 980)
        else:
            x += random.gauss(0, 2)
            y += random.gauss(0, 2)
        x = max(0, min(1920, x))
        y = max(0, min(1080, y))
        blink = random.random() < 0.003
        samples.append({'x': x, 'y': y, 'blink': blink, 'timestamp': len(samples) / fps})
    return samples


# --- Fixation detection (I-VT algorithm) ---
VELOCITY_THRESHOLD = 30  # degrees/sec

def compute_velocity(s1, s2, screen_dpi=96, dist_cm=60):
    """Approx angular velocity between two gaze samples (degrees/sec)."""
    dx = (s2['x'] - s1['x']) / screen_dpi * 2.54
    dy = (s2['y'] - s1['y']) / screen_dpi * 2.54
    dist_px = math.sqrt(dx**2 + dy**2)
    dt = s2['timestamp'] - s1['timestamp']
    if dt == 0:
        return 0
    angle = math.degrees(math.atan2(dist_px, dist_cm))
    return angle / dt


def classify_samples(samples):
    """Label each sample as fixation, saccade, or blink."""
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
    """Group consecutive fixation samples into fixation events."""
    fixations = []
    group = []
    for s, l in zip(samples, labels):
        if l == 'fixation':
            group.append(s)
        else:
            if group:
                duration = group[-1]['timestamp'] - group[0]['timestamp']
                if duration >= min_duration:
                    cx = sum(g['x'] for g in group) / len(group)
                    cy = sum(g['y'] for g in group) / len(group)
                    fixations.append({'x': cx, 'y': cy, 'duration': duration,
                                      'onset': group[0]['timestamp']})
                group = []
    return fixations


# --- Summary metrics ---
def summarize(samples, labels, fixations):
    n = len(samples)
    blink_rate = labels.count('blink') / (samples[-1]['timestamp'] / 60)  # per minute
    saccade_rate = labels.count('saccade') / samples[-1]['timestamp']
    avg_fix_dur = sum(f['duration'] for f in fixations) / len(fixations) if fixations else 0
    return {
        'total_samples': n,
        'duration_sec': round(samples[-1]['timestamp'], 2),
        'fixation_count': len(fixations),
        'avg_fixation_duration_ms': round(avg_fix_dur * 1000, 1),
        'saccade_rate_hz': round(saccade_rate, 2),
        'blink_rate_per_min': round(blink_rate, 1),
    }


# --- Main ---
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
    print(f"\n  Sample fixations:")
    for f in fixations[:3]:
        print(f"    ({f['x']:.0f}, {f['y']:.0f})  dur={f['duration']*1000:.0f}ms  onset={f['onset']:.2f}s")
