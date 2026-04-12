"""
adaptation_engine.py
Fuses signals from eye tracker, EEG, and GSR sensors to compute
a learner state and adapt task difficulty in real time.
"""

import random

# Difficulty levels
LEVELS = ['very easy', 'easy', 'medium', 'hard', 'very hard']

# --- Simulated sensor readings (would come from sensor modules in production) ---
def read_sensors():
    """Return a snapshot of sensor-derived metrics (0–1 normalized)."""
    return {
        'cognitive_load':   random.uniform(0, 1),   # from EEG CLI
        'arousal':          random.uniform(0, 1),   # from GSR
        'fixation_duration': random.uniform(0, 1),  # from eye tracker (longer = more effort)
        'blink_rate':       random.uniform(0, 1),   # from eye tracker (high = fatigue)
    }


# --- Learner state classification ---
WEIGHTS = {
    'cognitive_load':    0.35,
    'arousal':           0.25,
    'fixation_duration': 0.25,
    'blink_rate':       -0.15,  # high blink rate lowers engagement score
}

def compute_engagement(sensors):
    """Weighted sum → engagement score in [0, 1]."""
    score = sum(sensors[k] * w for k, w in WEIGHTS.items())
    return max(0.0, min(1.0, score))


def classify_state(engagement):
    if engagement < 0.2:
        return 'disengaged'
    elif engagement < 0.4:
        return 'under-challenged'
    elif engagement < 0.65:
        return 'in-flow'
    elif engagement < 0.85:
        return 'over-challenged'
    else:
        return 'overloaded'


# --- Adaptation logic ---
def adapt_difficulty(current_level, state):
    """Adjust difficulty index based on learner state."""
    idx = LEVELS.index(current_level)
    if state in ('disengaged', 'under-challenged'):
        idx = min(idx + 1, len(LEVELS) - 1)
    elif state in ('over-challenged', 'overloaded'):
        idx = max(idx - 1, 0)
    # 'in-flow' → no change
    return LEVELS[idx]


# --- Feedback messages ---
MESSAGES = {
    'disengaged':       "Increasing difficulty — let's keep you engaged.",
    'under-challenged': "Stepping it up a notch.",
    'in-flow':          "You're in the zone. Keeping current difficulty.",
    'over-challenged':  "Easing back — finding your optimal challenge.",
    'overloaded':       "Taking it down — reducing cognitive load.",
}


# --- Simulation loop ---
def run_session(steps=10):
    difficulty = 'medium'
    print("=== Adaptation Engine — Live Session ===\n")
    print(f"{'Step':<5} {'Engagement':>12} {'State':<18} {'Difficulty'}")
    print("-" * 55)

    for step in range(1, steps + 1):
        sensors = read_sensors()
        engagement = compute_engagement(sensors)
        state = classify_state(engagement)
        difficulty = adapt_difficulty(difficulty, state)
        msg = MESSAGES[state]
        print(f"{step:<5} {engagement:>12.3f} {state:<18} {difficulty}")

    print("\n--- Final sensor snapshot ---")
    for k, v in sensors.items():
        print(f"  {k}: {v:.3f}")
    print(f"\n  {msg}")


if __name__ == '__main__':
    random.seed(99)
    run_session(steps=12)
