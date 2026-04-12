"""
adaptation_engine.py
Fuses eye tracker, EEG, and GSR signals to compute learner state
and adapt task difficulty in real time.
Plots engagement score and difficulty level over the session.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LEVELS = ['very easy', 'easy', 'medium', 'hard', 'very hard']
LEVEL_IDX = {l: i for i, l in enumerate(LEVELS)}

WEIGHTS = {
    'cognitive_load':    0.35,
    'arousal':           0.25,
    'fixation_duration': 0.25,
    'blink_rate':       -0.15,
}

STATE_COLORS = {
    'disengaged':       '#95a5a6',
    'under-challenged': '#3498db',
    'in-flow':          '#2ecc71',
    'over-challenged':  '#f39c12',
    'overloaded':       '#e74c3c',
}

MESSAGES = {
    'disengaged':       "Increasing difficulty — keeping you engaged.",
    'under-challenged': "Stepping it up a notch.",
    'in-flow':          "You're in the zone. Holding current difficulty.",
    'over-challenged':  "Easing back — finding your optimal challenge.",
    'overloaded':       "Reducing difficulty — lowering cognitive load.",
}


def read_sensors():
    return {
        'cognitive_load':    random.uniform(0, 1),
        'arousal':           random.uniform(0, 1),
        'fixation_duration': random.uniform(0, 1),
        'blink_rate':        random.uniform(0, 1),
    }


def compute_engagement(sensors):
    score = sum(sensors[k] * w for k, w in WEIGHTS.items())
    return max(0.0, min(1.0, score))


def classify_state(engagement):
    if engagement < 0.20:  return 'disengaged'
    if engagement < 0.40:  return 'under-challenged'
    if engagement < 0.65:  return 'in-flow'
    if engagement < 0.85:  return 'over-challenged'
    return 'overloaded'


def adapt_difficulty(current, state):
    idx = LEVEL_IDX[current]
    if state in ('disengaged', 'under-challenged'):
        idx = min(idx + 1, len(LEVELS) - 1)
    elif state in ('over-challenged', 'overloaded'):
        idx = max(idx - 1, 0)
    return LEVELS[idx]


def run_session(steps=20):
    difficulty = 'medium'
    log = []

    print("=== Adaptation Engine — Live Session ===\n")
    print(f"{'Step':<5} {'Engagement':>10} {'State':<18} {'Difficulty'}")
    print("-" * 55)

    for step in range(1, steps + 1):
        sensors = read_sensors()
        engagement = compute_engagement(sensors)
        state = classify_state(engagement)
        difficulty = adapt_difficulty(difficulty, state)
        log.append({'step': step, 'engagement': engagement,
                    'state': state, 'difficulty': difficulty,
                    'sensors': sensors})
        print(f"{step:<5} {engagement:>10.3f} {state:<18} {difficulty}")

    return log


def plot_results(log):
    steps = [e['step'] for e in log]
    engagement = [e['engagement'] for e in log]
    difficulty_idx = [LEVEL_IDX[e['difficulty']] for e in log]
    states = [e['state'] for e in log]
    point_colors = [STATE_COLORS[s] for s in states]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle('Adaptive Learning Engine — Session Report', fontsize=14, fontweight='bold')

    # --- Panel 1: Engagement score ---
    ax = axes[0]
    ax.plot(steps, engagement, color='#2c3e50', linewidth=1.5, zorder=2)
    ax.scatter(steps, engagement, c=point_colors, s=60, zorder=3, edgecolors='white', linewidths=0.5)
    ax.axhspan(0.40, 0.65, color='#2ecc71', alpha=0.1, label='Flow zone')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Engagement (0–1)')
    ax.set_title('Engagement Score', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_facecolor('#f8f9fa')

    # --- Panel 2: Difficulty level ---
    ax2 = axes[1]
    ax2.step(steps, difficulty_idx, where='post', color='#8e44ad', linewidth=2)
    ax2.set_yticks(range(len(LEVELS)))
    ax2.set_yticklabels(LEVELS, fontsize=8)
    ax2.set_ylabel('Difficulty')
    ax2.set_title('Adapted Difficulty Level', fontsize=10)
    ax2.set_facecolor('#f8f9fa')

    # --- Panel 3: Sensor readings ---
    ax3 = axes[2]
    sensor_keys = ['cognitive_load', 'arousal', 'fixation_duration', 'blink_rate']
    sensor_colors = ['#e74c3c', '#f39c12', '#3498db', '#1abc9c']
    for key, color in zip(sensor_keys, sensor_colors):
        vals = [e['sensors'][key] for e in log]
        ax3.plot(steps, vals, color=color, alpha=0.75, linewidth=1.2, label=key.replace('_', ' '))
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Normalised value')
    ax3.set_xlabel('Session step')
    ax3.set_title('Sensor Readings', fontsize=10)
    ax3.legend(fontsize=7, ncol=2)
    ax3.set_facecolor('#f8f9fa')

    # Legend for state colors
    patches = [mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=8,
               title='Learner State', bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


if __name__ == '__main__':
    random.seed(99)
    log = run_session(steps=20)
    plot_results(log)
    print("\nPlot saved.")
