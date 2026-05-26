# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AICodeBrain is a browser-based AI code companion that lets users paste or upload Python code and have it run, debugged, explained, improved, or tested. It uses Claude as the AI backbone and executes Python server-side via a Node.js proxy.

## Running the App

```bash
# Install dependencies (only dotenv is required)
npm install

# Start the server
npm start
# or: node server.js
```

The app runs at `http://localhost:3000` (or `PORT` env var).

**Required:** Copy `.env.example` to `.env` and set your `ANTHROPIC_API_KEY`.

Python must be available on the system path as `python` (override with `PYTHON` env var).

## Architecture

### Request Flow

```
Browser (index.html)
  └── /api/messages  → server.js proxies to api.anthropic.com (key stays server-side)
  └── /api/run       → server.js writes a temp Python script, executes it via execFile(),
                       captures stdout/stderr/plots, cleans up, returns JSON
```

The entire frontend is a single `index.html` with inline CSS and JS — no build step, no bundler.

### Python Execution (`/api/run`)

`server.js` wraps every submitted script before execution:
- Sets `matplotlib` to the `Agg` (non-interactive) backend
- Monkey-patches `plt.show()` to save figures as `plot_N.png` in a temp dir
- After execution, reads those PNGs and returns them as base64 `data:image/png` strings in the `plots` array

This is how plots appear in the UI without a display server.

### Smart Ask / Multi-Step Claude Calls

When a folder is loaded, "Smart Ask" does a two-step Claude pipeline:
1. **File selection** — sends only filenames to `claude-haiku-4-5-20251001` (fast, cheap) to pick the relevant file
2. **Action routing** — sends a short prompt to Haiku to classify the user query as `run | debug | explain | improve | test | data`
3. **Execution** — for `run`/`data`, calls `/api/run`; for others, calls `/api/messages` with a tailored system prompt using `claude-sonnet-4-6`

### Hardware Insights

After any `run` or `data` action, `detectSensorType()` checks the code against regex patterns for EEG, Eye Tracker, GSR, IMU, and Multi-Sensor Fusion. If a sensor type is detected, a second Claude call (`claude-sonnet-4-20250514`, 1200 tokens) generates structured hardware analysis rendered via a custom markdown-lite parser.

### Demo Scripts (`demo/`)

These are sample biosensor processing scripts that demonstrate the hardware insights feature:
- `eeg_processor.py` — simulates EEG band power analysis (delta/theta/alpha/beta) and cognitive load index
- `adaptation_engine.py` — multi-sensor fusion; fuses EEG cognitive load, GSR arousal, and eye tracker fixation/blink signals to adapt task difficulty in real time
- `gsr_sensor.py` — galvanic skin response simulation (tonic/phasic decomposition, SCR detection, arousal score)
- `eyetracker.py` — gaze/fixation/saccade simulation
- `imu_sensor.py` — inertial measurement unit (accelerometer + gyroscope) step detection
- `ecg_sensor.py` — electrocardiogram simulation; synthesises PQRST complexes via Gaussian sums for three cardiac states (resting/exercise/stressed), detects R-peaks, computes RR intervals and HRV (RMSSD)
- `ppg_sensor.py` — photoplethysmography simulation; models systolic peaks + dicrotic notch, detects beats, computes inter-beat intervals (IBI), pulse amplitude, and SpO2 estimate

All demo scripts are self-contained (use `random` + `math` + `matplotlib`) and run without external data files.

## Key Decisions & Constraints

- **No frontend framework / no bundler.** Everything in one `index.html`. Adding a framework requires also adding a build step and updating `server.js` to serve static assets.
- **API key is server-side only.** The browser never sees `ANTHROPIC_API_KEY`; all Anthropic API calls go through the `/api/messages` proxy in `server.js`.
- **Python execution is sandboxed by temp dir** (`os.tmpdir()`), with a 30-second timeout and 1 MB stdout buffer. Scripts that need external packages must have those packages installed in the system Python environment.
- **Plots are only shown in the UI when the user's query contains the word "plot"** (for the `run` action in Smart Ask). The `data` action always shows plots.
- **History is persisted to `localStorage`** under key `py_runner_history` (max 20 entries). No server-side persistence.
