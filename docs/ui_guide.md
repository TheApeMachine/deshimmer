# Interactive Gradio Web Interface Guide

Deshimmer provides a comprehensive, interactive browser-based interface built with Gradio Blocks in [src/deshimmer/ui/](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/ui/). It allows audio engineers and researchers to visually inspect time-frequency spectrograms, audition seamless loops, run automated multi-objective optimizations, and export broadcast-grade 24-bit PCM masters and forensic difference files.

---

## 1. Launching the Web Interface

Launch the interface with a single command using the project `Makefile`:

```bash
make run
```

Alternatively, you can launch via the CLI entry points:

```bash
deshimmer-ui
# or
python -m deshimmer.ui.app
```

The web server will initialize and report:
```
Running on local URL:  http://localhost:7860
```
Open your browser and navigate to `http://localhost:7860`.

---

## 2. Interface Layout Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ Deshimmer UI                                                           │
├────────────────────────────────────────────────────────────────────────┤
│ [ Audio Input (WAV, FLAC, MP3) ]                                       │
│                                                                        │
│ [ Preview Controls: Start (s) | Duration (s) | Loop Crossfade (Toggle)]│
│ [ Action Buttons:  Preview Audio  |  Set from Audio  |  Optimize      ]│
├──────────────────────────────────┬─────────────────────────────────────┤
│ Audio Players                    │ Spectrogram & Metrics Display       │
│  - Input (Original Dry)          │  - Dynamic 2D Spectrogram (PNG)     │
│  - Output (Repaired Wet)         │  - Real-time LUFS, RMS, True-Peak   │
│  - Removed (Forensic Diff)       │  - Tonal Coherence Metrics          │
├──────────────────────────────────┴─────────────────────────────────────┤
│ Parameter Tabs & Knobs:                                                │
│  [Presets] [Shimmer] [Denoise & Swish] [De-Resonator] [Expander/HPSS]  │
│  [HF Resynth] [Delivery Mastering]                                     │
├────────────────────────────────────────────────────────────────────────┤
│ Export Section:                                                        │
│  [ Render Full Song & Generate Downloads (24-bit PCM WAV + Diffs) ]    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Workflow

### Step 1: Load Audio File
- Drag and drop any stereo or mono audio file (WAV, FLAC, MP3, OGG, AIFF) into the **Input Audio** component.
- The player will load the file and display sample rate and channel configuration.

### Step 2: Configure Preview Loop
Iterating over an entire song in real time is inefficient. Deshimmer is designed for fast, focused iteration:
1. **Preview Start (s)**: Set the timestamp where the artifact is most prominent (e.g. `24.5` seconds into a chorus or quiet vocal entry).
2. **Preview Duration (s)**: Choose a 3 to 8-second window.
3. **Loop Crossfade**: When checked, Deshimmer rotates the slice using an equal-power sinusoidal crossfade at the boundary. When you enable the loop button in the browser player, playback repeats seamlessly without click or pop discontinuities.

### Step 3: Automated Optimization (One-Click Auto-Tune)
Instead of manually tweaking 20+ knobs:
- Click **Set from audio**:
  - Automatically identifies stationary whistles and high-frequency shimmer bands.
  - Updates the frequency sliders (`start_hz`, `end_hz`, `deq_start_hz`, `deq_end_hz`) to target the detected anomalies.
- Click **Optimize**:
  - Select trial count (e.g. 15 to 40 trials) and desired **Aggressiveness** (0.0 = ultra-safe preservation, 1.0 = maximum artifact extraction).
  - The NSGA-II evolutionary optimizer searches the multi-objective Pareto space across quiet and loud passages.
  - The knobs update automatically to the best-scoring compromise, and a detailed optimization report displays the exact trade-offs found.

### Step 4: Critical Auditioning & Forensic Diff Listening
Use the three synchronized audio players to verify repair transparency:
1. **Input Audio**: The unprocessed source slice.
2. **Output Audio**: The repaired signal with current settings.
3. **Removed Audio (Diff)**: The exact difference signal ($x_{\text{input}} - y_{\text{processed}}$).

#### Forensic Verification Checkboxes:
- **Listen to removed audio (Delta Listen)**: Directs the diff directly to your main monitoring output. When correctly configured, you should hear *only* the harsh metallic buzz, whine, or hiss. If you hear clear vocals, snare drums, or guitar attacks, lower your threshold or decrease slope to prevent musical damage.
- **Listen to repair-only diff**: Evaluates the pure artifact subtraction without the influence of the downstream mastering limiter or spatial enhancers.

### Step 5: Render Full Master & Download
Once the preview settings sound optimal:
1. Click **Render full & generate downloads**.
2. Deshimmer processes the entire file from start to finish at full 64-bit precision.
3. Three downloadable files are generated in `./ui_downloads/`:
   - `output-<timestamp>.wav`: Broadcast-ready 24-bit PCM master.
   - `diff-<timestamp>.wav`: Full-length forensic difference audio file.
   - `params-<timestamp>.json`: Complete JSON sidecar recording all parameters used for exact archival reproducibility.

---

## 4. Parameter Tabs Reference

### Presets Tab
Select from 17 factory recipes or load/save custom user presets stored in `ui_presets.json`:
- `01 - Bypass`:mix=0.0 sanity check for end-to-end null tests.
- `02 - Default shimmer`: Conservative baseline for 5.1–7.2 kHz sparkle.
- `06 - Shimmer + light denoise`: Balanced hiss and shimmer suppression.
- `08 - De-resonator (gentle)`: Dynamic EQ notch for narrow harmonic rings.
- `10 - Full stack (conservative)`: Recommended production starting point.
- `16 - Suno/Udio: 3.5k whine`: Tuned specifically for persistent 3.5 kHz whistles in generative audio.
- `17 - Suno/Udio: Metallic crickets`: Full recipe for stationary lines and high-frequency phase swish.

### Shimmer Suppression Tab
- **Band Start / End (Hz)**: Boundary frequencies of the shimmer region (default: 5100 to 7200 Hz).
- **Threshold (dB)**: Sensitivity above local background level (lower = more aggressive).
- **Slope**: Attenuation aggression curve (0.1 to 1.5).
- **Density Lo / Hi**: Frame density gating preventing attenuation during broadband passages.
- **Noise Resynth**: Introduces subtle random-phase noise in noise-like frames to break up artificial crystalline textures.

### Denoise & Swish Tab
- **Denoise Depth**: Strength of minimum-statistics spectral subtraction (0.0 to 1.0).
- **Noise Floor (dB)**: Safety floor limiting maximum noise attenuation (e.g. -24 dB) to prevent musical noise chirps.
- **Min Window (ms)**: Time window for noise floor tracking (default: 1000 ms).
- **Swish Repair**: High-frequency phase flutter stabilizer.

### De-Resonator Tab
- **De-Resonator Depth**: Dynamic notch filter aggression (0.0 to 1.0).
- **Persistence (ms)**: Minimum duration a resonance must sustain before notch attenuation triggers (e.g. 1000 ms).
- **Max Attenuation (dB)**: Maximum allowable notch depth (default: 12 dB).
- **Time-Stabilized Floor**: Protects musical vibrato by tracking the stationary baseline.

### Delivery Mastering Tab
- **Enable Mastering**: Toggles the final loudness normalization and true-peak limiter.
- **Target LUFS**: Integrated loudness target (e.g. -14.0 LUFS for streaming, -10.0 LUFS for club/hot masters).
- **Ceiling (dBTP)**: True-peak ceiling enforced with 4x oversampling (e.g. -1.0 dBTP).
- **Subsonic High-Pass (Hz)**: 20 Hz Butterworth filter eliminating subsonic energy.
