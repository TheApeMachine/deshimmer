# Deshimmer

Deshimmer is an audio repair toolkit and parameter optimization engine for suppressing the spectral artifacts common in AI-generated audio and neural vocoders.

Neural synthesis models (diffusion networks, flow-matching engines, and generative vocoders) frequently introduce audible anomalies in the upper frequency spectrum:
- **Shimmer and "birdies"**: High-frequency spectral spikes (typically between 4 kHz and 12 kHz) that flicker rapidly across time frames, producing an unnatural, metallic, "crystalline" sizzle.
- **Stationary whines**: Steady, pure-tone resonant lines caused by generator clock harmonics, frame-grid boundaries, or neural sampling artifacts.
- **Phase swish**: Chaotic inter-frame and inter-bin phase instability in high frequencies that makes cymbals, reverbs, and sustained notes sound watery and disconnected.
- **Tonal partial wandering**: Pitch jitter and harmonic drift in sustained melodic lines.

Rather than running a destructive black-box neural model to fix another neural model, Deshimmer uses stateful, deterministic digital signal processing (DSP) coupled with multi-objective Bayesian optimization. You can inspect every stage, listen to the removed difference signal in isolation, and verify that the original audio is preserved.

---

## Key Features

- **Modular DSP Pipeline**: Eleven discrete, chainable processing stages operating in the Short-Time Fourier Transform (STFT) domain, with Numba-accelerated inner loops for performance.
- **Tonal Self-Consistency**: High-resolution Quadratic Interpolated FFT (QIFFT) partial tracking that identifies harmonic families and stabilizes wandering partials without retuning legitimate vibrato or glissandi.
- **Two-Layer Auto-Tuning**:
  - *Layer 1 (Deterministic Analysis)*: Quickly scans representative audio regions (quiet, loud, transient, tonal) to identify shimmer bands, noise floor levels, and stationary whines, proposing baseline settings.
  - *Layer 2 (Multi-Objective Optimization)*: Uses Optuna's NSGA-II genetic algorithm across six acoustic objectives (artifact reduction, music preservation, transient leakage, stereo width, spectral tilt, and tonal coherence) to find the best parameter trade-offs.
- **Full-Featured Web UI**: A Gradio-based interface with synchronized dual-player playback (input vs. processed output vs. isolated residual), high-resolution spectrograms, looping crossfades, and preset management.
- **Mastering Chain**: Optional EBU R128 integrated loudness normalization (LUFS) and an oversampled (4x/8x) lookahead true-peak limiter.
- **Pure Python with `uv`**: Packaged with modern standards (`pyproject.toml`) and installed cleanly via `uv`.

---

## Installation

Deshimmer requires **Python 3.10+** and uses [uv](https://docs.astral.sh/uv/) for package and virtual environment management.

### 1. Install `uv`
If you do not have `uv` installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Environment

```bash
git clone https://github.com/theapemachine/deshimmer.git
cd deshimmer

# Create a virtual environment and install deshimmer in editable mode with dev dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Quick Start

### 1. Web Interface (Gradio)

Launch the interactive UI:

```bash
deshimmer-ui
# or: python -m deshimmer
```

Open `http://localhost:7860` in your web browser.

#### Interactive Workflow:
1. **Load Audio**: Drop a WAV, FLAC, or MP3 file into the Input Audio component.
2. **Select Preview**: Set preview start time and duration (e.g. 4–6 seconds around a noisy passage) and enable loop crossfade for smooth auditioning.
3. **Auto-Tune (Optional)**:
   - Click **Set from audio** to detect shimmer bands and stationary resonances automatically.
   - Click **Optimize** to run the NSGA-II multi-objective optimizer. It evaluates candidate settings across multiple regions and updates the sliders with the best-scoring parameters.
4. **Audition**:
   - Compare **Input** vs. **Output**.
   - Check **Listen to removed audio** (or **Listen to repair-only diff**) to hear exactly what is being subtracted. If you hear drums or dry vocal leads in the diff, back off the reduction slope or increase transient flux thresholds.
5. **Render Full Track**: Click **Render full & generate downloads** to process the complete file to 24-bit PCM WAV.

---

### 2. Command-Line Interface (CLI)

Process audio files directly from your terminal:

```bash
# Basic repair with default shimmer suppression (5.1–7.2 kHz)
deshimmer-cli input.wav output.wav

# Target a specific resonance whine with the dynamic de-resonator
deshimmer-cli input.wav output.wav \
  --start-hz 3000 --end-hz 4200 \
  --deres 0.8 --deq-thr-db 4.0 --deq-max-att-db 12.0 \
  --deq-persist-ms 1000

# Combine shimmer repair, light denoising, and EBU R128 delivery mastering (-14 LUFS)
deshimmer-cli input.wav output.wav \
  --thr-db 7.0 --slope 0.75 \
  --denoise 0.30 --dn-floor-db -20 \
  --master --target-lufs -14.0 --ceiling-dbtp -1.0
```

Run `deshimmer-cli --help` to see all available command-line flags.

---

### 3. Python API

Deshimmer can be integrated directly into your Python scripts or batch pipelines:

```python
import soundfile as sf
from deshimmer import process_audio, Params, MasterParams

# Load audio (channels as columns, shape (N, C))
audio, sr = sf.read("input.wav")

# Configure repair settings
params = Params(
    start_hz=4800.0,
    end_hz=8200.0,
    thr_db=7.5,
    slope=0.7,
    swish_repair=0.35,
    tonal_repair=0.5,
)

# Optional delivery mastering
mastering = MasterParams(
    enabled=True,
    target_lufs=-14.0,
    ceiling_dbtp=-1.0,
)

# Process audio
output, info = process_audio(
    audio,
    sr,
    params=params,
    master_params=mastering,
    include_residuals=True,
)

# Access measurement statistics
print(f"Input LUFS: {info['measure_in']['lufs']:.2f}")
print(f"Output LUFS: {info['measure_out']['lufs']:.2f}")
print(f"True peak: {info['measure_out']['true_peak_dbtp']:.2f} dBTP")

# Save processed output
sf.write("output.wav", output, sr, subtype="PCM_24")
```

---

## DSP Pipeline Architecture

Deshimmer processes audio through a pipeline of decoupled STFT stages. The internal structure operates as follows:

```
                  ┌───────────────────────────────┐
                  │       Input Audio (L/R)       │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  L/R Micro-Delay Alignment    │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │   STFT Analysis (Hann, 50%+)  │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │   L/R Spectral Balancing      │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ Psychoacoustic Mask Threshold │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │        Modular Stages         │
                  │  1. Downward Expander         │
                  │  2. Smart Spectral Denoise    │
                  │  3. Adaptive De-Resonator     │
                  │  4. Shimmer Suppressor        │
                  │  5. Dynamic Spectral Carver   │
                  │  6. Texture Noise Resynth     │
                  │  7. Phase Blur                │
                  │  8. Swish Phase Repair        │
                  │  9. Stereo Decorrelation      │
                  │ 10. Gain App & Inpainting     │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  Tonal Self-Consistency       │
                  │  (QIFFT Tracking + Hermite)   │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │        iSTFT Synthesis        │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ Delivery Mastering & Limiter  │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │         Output Audio          │
                  └───────────────────────────────┘
```

### Core Processing Stages

1. **L/R Micro-Delay & Spectral Balancing**: Corrects sub-millisecond phase offsets and channel level imbalances introduced by lossy vocoder decoders before spectral analysis.
2. **Psychoacoustic Masking Constraint**: Calculates simultaneous spectral masking curves and forward temporal decay (using Numba JIT). Suppressions that would fall below hearing thresholds are clamped to prevent audible dropouts or hollow artifacts.
3. **Smart Spectral Denoise**: Implements minimum-statistics noise floor tracking. Rather than using an abrupt noise gate, it computes a soft floor attenuation that preserves natural room ambiance while removing background hiss.
4. **Adaptive De-Resonator**: Distinguishes between musical harmonics and unwanted stationary ringing. A frequency peak must demonstrate sustained temporal persistence across multiple analysis windows before the dynamic notch filter engages.
5. **Shimmer Suppressor with Inpainting**: Compares spectral energy against a moving frequency median. Bins that exceed the threshold in noisy/unvoiced sections are reduced. When *Magnitude Inpainting* is enabled, spikes are capped to the local median rather than cut to zero, preserving natural high-frequency presence.
6. **Swish Phase Repair**: Evaluates the inter-frame and inter-bin phase derivative variance. In regions where phase instability indicates vocoder phase-smearing ("swish"), the complex phasor angles are smoothed along time and frequency axes.
7. **Tonal Self-Consistency (`src/deshimmer/tonal/`)**:
   - `PartialTracker`: Extracts sinusoidal peaks with sub-bin precision using parabolic QIFFT peak interpolation on a 4x zero-padded Hann spectrum.
   - `HarmonicField`: Groups related partial trajectories into harmonic families using alternating medians on log-frequency space.
   - `TonalRepair`: Corrects erratic wandering of individual harmonics relative to their family core using cubic Hermite phase synthesis, leaving independent musical voices untouched.
8. **Delivery Mastering**: High-pass filtering (sub-bass cleanup), EBU R128 integrated loudness normalization, and a 4x/8x oversampled true-peak limiter.

---

## Auto-Tuning Engine

Finding clean settings manually across 20+ DSP parameters can be tedious. Deshimmer's auto-tuning engine automates this in two steps:

### Layer 1: Deterministic Analysis (`analyze`)
Analyzes the audio across representative regions (quiet background, loud sections, transients, and vocal passages):
- Detects the primary shimmer frequency range by calculating spectral residue density.
- Estimates the background noise floor to configure denoiser thresholds.
- Identifies stationary resonance peaks and estimates their persistence.
- Evaluates tonal consistency metrics across harmonic families.

### Layer 2: Multi-Objective Bayesian Optimization (`refine`)
Uses Optuna's **NSGA-II** genetic sampler to optimize parameters sequentially across stages (shimmer, denoise, deres, swish, tonal). It optimizes six explicit objectives simultaneously:

1. **Artifact Reduction**: Maximizes dB reduction in detected shimmer and swish bands.
2. **Music Preservation**: Penalizes changes outside the target band, spectral centroid shifts, and added musical noise.
3. **Diff Leakage**: Penalizes transient energy (percussive attacks) and strong musical harmonics leaking into the difference signal.
4. **Stereo Preservation**: Penalizes widening or narrowing of the inter-channel correlation ratio.
5. **Loudness & Tilt Preservation**: Penalizes unwanted spectral darkening or loudness loss.
6. **Tonal Coherence**: Rewards improved agreement among detected harmonic families.

Candidate trials that cause mathematical instability or boundary pumping are rejected. Current settings are retained if proposed candidate trials score worse than the incoming baseline.

---

## Project Structure

```
deshimmer/
├── pyproject.toml              # Build configuration and dependencies (uv / hatchling)
├── ui_presets.json             # Bundled factory presets
├── src/
│   └── deshimmer/
│       ├── __init__.py         # Public package exports
│       ├── __main__.py         # Package executable entrypoint
│       ├── api.py              # High-level Python API (process_audio, slice_with_context)
│       ├── cli.py              # Command-line interface logic
│       ├── core/               # Core DSP modules
│       │   ├── context.py      # DSPContext data container
│       │   ├── debug.py        # Diagnostic metrics and spectrogram plotting
│       │   ├── hpss.py         # Harmonic-percussive separation and inpainting
│       │   ├── mastering.py    # Peak limiter and LUFS normalization
│       │   ├── measurements.py # True peak, LUFS, and phase instability metrics
│       │   ├── params.py       # Params, MasterParams, and DebugParams dataclasses
│       │   ├── pipeline.py     # DSPPipeline orchestrator and process_stft
│       │   ├── psychoacoustic.py # Psychoacoustic masking & temporal decay
│       │   ├── resynth.py      # High-frequency harmonic resynthesis
│       │   ├── spatial.py      # Delay alignment, spectral balancing, and carving
│       │   ├── utils.py        # DSP conversion helpers
│       │   └── stages/         # Individual STFT processing stages
│       │       ├── base.py
│       │       ├── carver.py
│       │       ├── decorrelation.py
│       │       ├── denoise.py
│       │       ├── deresonator.py
│       │       ├── expander.py
│       │       ├── gain.py
│       │       ├── noise_resynth.py
│       │       ├── phase_blur.py
│       │       ├── shimmer.py
│       │       └── swish.py
│       ├── tonal/              # Sinusoidal tracking and harmonic repair
│       │   ├── tracking.py     # QIFFT partial tracking
│       │   ├── harmonic_field.py # Harmonic family fitting
│       │   └── repair.py       # Tonal synthesis and phase reconstruction
│       ├── autotune/           # Automatic analysis and parameter optimization
│       │   ├── analyzer.py     # Deterministic prior detection and region selection
│       │   ├── optimizer.py    # Optuna NSGA-II multi-objective optimization
│       │   ├── scoring.py      # Acoustic metric scoring functions
│       │   └── types.py        # Region, Weights, and report definitions
│       └── ui/                 # Gradio Web Interface
│           ├── app.py          # UI layout and event bindings
│           ├── audio.py        # Audio formatting, slicing, and spectrograms
│           ├── handlers.py     # Preview execution and full export handlers
│           ├── presets.py      # Preset loading and saving
│           └── state.py        # UI state mapping and conversions
├── tests/                      # Automated test suite
│   ├── test_auto_tune.py
│   ├── test_deshimmer_api.py
│   ├── test_dynamic_spectral_carver.py
│   ├── test_harmonic_field.py
│   ├── test_master.py
│   ├── test_tonal_repair.py
│   ├── test_tonal_tracking.py
│   ├── test_ui_gradio.py
│   └── tonal_fixtures.py
```

---

## Testing

Run the full test suite using `pytest`:

```bash
uv run pytest
```

All 27 regression tests verify that channel alignment, spectral balancing, stage isolation, Optuna scoring, file exports, and UI callbacks execute with 100% numerical fidelity.

---

## License

This project is licensed under the [MIT License](LICENSE).
