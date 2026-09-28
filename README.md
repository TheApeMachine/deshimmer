# Deshimmer

Deshimmer is a surgical audio restoration toolkit and multi-objective parameter optimization engine designed to eliminate high-frequency artifacts common in AI-generated audio (e.g. Suno, Udio, Stable Audio) and aggressive lossy codecs.

Neural vocoders, diffusion networks, and generative audio models frequently introduce characteristic acoustic anomalies:
- **High-Frequency Shimmer ("Birdies")**: Rapidly fluctuating, metallic energy spikes (typically in the 4–12 kHz range) that produce an unnatural digital sizzle.
- **Stationary Resonant Whines**: Persistent narrowband whistles caused by sampling rate clock harmonics, frame-grid boundaries, or vocoder resonance nodes.
- **Phase Swish & Incoherence**: Erratic inter-frame phase jumps that make cymbals, reverbs, and sustained instruments sound watery or comb-filtered.
- **Sinusoidal Harmonic Drift**: Micro-pitch wandering and jitter in sustained melodic lines.

Rather than applying a destructive black-box neural network to repair another neural network, Deshimmer relies on **deterministic, psychoacoustically bounded digital signal processing (DSP)** combined with an **evolutionary NSGA-II multi-objective optimizer**. Every stage is fully transparent, mathematically invertible, and auditable through isolated forensic difference signals.

---

## ⚡ Quickstart

Get the interactive web application up and running with a single command:

```bash
make run
```

This single command automatically bootstraps the virtual environment, verifies dependencies, and launches the Gradio Web UI at **`http://localhost:7860`**.

---

## 📚 Documentation Index

The system is granularly documented across specialized guides in the [`docs/`](docs/) directory:

| Document | Description |
|:---|:---|
| **[Core DSP Pipeline](docs/dsp_pipeline.md)** | Deep dive into STFT framing, psychoacoustic masking curves, spatial alignment, and the 10 discrete processing stages. |
| **[Auto-Tune & NSGA-II Optimizer](docs/autotune_optimizer.md)** | Explanation of dynamic region slicing, the 6 weighted acoustic objectives, synthetic seam scoring, and Pareto trial selection. |
| **[Tonal Self-Consistency](docs/tonal_repair.md)** | Mathematical formulation of QIFFT partial tracking, alternating-median harmonic field regression, and cubic Hermite phase synthesis. |
| **[Processing Contracts & Residuals](docs/processing.md)** | Details on bypass invertibility ($mix=0$ null contract), dual residual separation, and forensic difference verification. |
| **[Interactive Web UI Guide](docs/ui_guide.md)** | Walkthrough of the Gradio interface: waveform/spectrogram displays, loop crossfade playback, parameter tabs, presets, and 24-bit export. |
| **[CLI Reference & Batch Manual](docs/cli.md)** | Complete command-line flag documentation, batch shell scripting recipes, and headless server rendering. |
| **[Python API Reference](docs/api.md)** | Developer documentation for `process_audio`, parameter dataclasses (`Params`, `MasterParams`), and custom pipeline construction. |
| **[Architecture & Extension Guide](docs/architecture.md)** | Codebase layout, design principles, testing philosophy, and a step-by-step tutorial on implementing custom `DSPStage` modules. |

---

## 🛠️ Key Capabilities

- **10-Stage Modular DSP Pipeline**: Operates in the Short-Time Fourier Transform (STFT) domain with Numba JIT-accelerated performance loops. Stages include downward expansion, minimum-statistics noise subtraction, dynamic de-resonators, dual-threshold flux attenuators, HPSS, noise resynthesis, and phase blur.
- **Tonal Self-Consistency Stage**: Extracts sinusoids with sub-bin precision using Quadratic Interpolated FFT (QIFFT), groups trajectories into harmonic families via alternating-median regression, and resolves wandering partials using cubic Hermite phase interpolation.
- **Two-Layer Automated Optimization**:
  - *Layer 1 (Deterministic Scene Analysis)*: Scans dynamic audio regions (quiet, loud, transient, tonal) to identify shimmer frequency bands and stationary whines automatically.
  - *Layer 2 (NSGA-II Multi-Objective Optimization)*: Explores the multi-dimensional parameter space across six competing acoustic objectives (artifact reduction, music preservation, transient leakage, stereo width, spectral balance, and tonal coherence) to find optimal Pareto trade-offs.
- **Forensic Delta Verification**: Audition the exact difference signal ($x_{\text{input}} - y_{\text{processed}}$) in real time. If musical notes or drum hits leak into the removed audio, parameters can be instantly adjusted before rendering.
- **Integrated Studio Mastering**: Subsonic high-pass filtering (20 Hz), ITU-R BS.1770 / EBU R128 loudness normalization (LUFS), and a 4x oversampled true-peak lookahead limiter.
- **Clean Tooling with Make & `uv`**: Fully standardized packaging using `pyproject.toml`, reproducible `uv` dependency management, and a comprehensive `Makefile`.

---

## 🚀 Usage

### 1. Interactive Web Interface (Gradio)

Launch the interface via `make run` or directly via CLI:

```bash
make run
# or: deshimmer-ui
# or: python -m deshimmer.ui.app
```

Navigate to `http://localhost:7860` in your web browser.

#### Standard UI Workflow:
1. **Load Audio**: Drop a WAV, FLAC, MP3, or AIFF file into the Input component.
2. **Set Preview Loop**: Choose a 3–8 second slice around an audible artifact. Enable **Loop Crossfade** for click-free browser looping.
3. **Auto-Tune**:
   - Click **Set from audio** to detect shimmer bands and stationary resonances.
   - Click **Optimize** to run the NSGA-II evolutionary search and populate the sliders with the best-scoring compromise.
4. **Audition**: Switch between **Input**, **Output**, and **Removed Audio (Diff)**.
5. **Export**: Click **Render full & generate downloads** to export a 24-bit PCM WAV master, difference track, and JSON parameter sidecar.

*For full details on all controls, see the [Web UI Guide](docs/ui_guide.md).*

---

### 2. Command-Line Batch Processing (CLI)

Process audio files directly from the terminal or automate whole directories:

```bash
# Basic repair on default 5.1–7.2 kHz shimmer band
deshimmer-cli input.wav output.wav

# Eliminate persistent 3.5 kHz generative whine and write difference WAV
deshimmer-cli generated.mp3 cleaned.wav \
  --start-hz 3000 --end-hz 4200 \
  --deres 0.8 --deq-thr-db 4.0 --deq-max-att-db 12.0 \
  --deq-persist-ms 1000 \
  --write-diff diff.wav

# Full restoration stack with streaming delivery mastering (-14.0 LUFS, -1.0 dBTP)
deshimmer-cli rough_mix.wav master_delivery.wav \
  --thr-db 7.5 --slope 0.7 \
  --denoise 0.35 --dn-floor-db -22.0 \
  --tonal-repair 0.5 \
  --master --target-lufs -14.0 --ceiling-dbtp -1.0
```

*For complete flag listings and shell scripting recipes, see the [CLI Reference](docs/cli.md).*

---

### 3. Python API Integration

Embed Deshimmer directly into Python applications, batch jobs, or ML pipelines:

```python
import soundfile as sf
from deshimmer.api import process_audio
from deshimmer.core import Params, MasterParams

# Load audio (channels as columns, shape (samples, channels))
audio, sr = sf.read("input.wav", dtype="float32")

# Configure DSP parameters and delivery mastering
params = Params(
    start_hz=4800.0,
    end_hz=8200.0,
    thr_db=7.5,
    slope=0.7,
    tonal_repair=0.5,
)
mastering = MasterParams(enabled=True, target_lufs=-14.0, ceiling_dbtp=-1.0)

# Process audio
output, info = process_audio(
    audio,
    sr,
    params=params,
    master_params=mastering,
    include_residuals=True,
)

# Export broadcast-ready 24-bit master
sf.write("repaired_master.wav", output, sr, subtype="PCM_24")
```

*For comprehensive parameter tables and custom pipeline examples, see the [Python API Reference](docs/api.md).*

---

## 🏗️ Architecture Summary

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
                  │   STFT Analysis (Hann, 75%)   │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ Psychoacoustic Mask Threshold │
                  │ (ATH + Forward Temporal Decay)│
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  Modular DSP Pipeline Stages  │
                  │  1. Downward Expander         │
                  │  2. Minimum-Stats Denoise     │
                  │  3. Adaptive De-Resonator     │
                  │  4. Shimmer Suppressor        │
                  │  5. Swish Phase Repair        │
                  │  6. HPSS Transients Filter    │
                  │  7. Noise Resynthesis         │
                  │  8. Phase Blur Dispersion     │
                  │  9. Dynamic Spectral Carver   │
                  │ 10. Gain Application          │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │    Tonal Self-Consistency     │
                  │ (QIFFT Tracking + Hermite)    │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │        iSTFT Synthesis        │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │ Delivery Mastering & Limiter  │
                  │ (High-pass + LUFS + True-Peak)│
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  Final Master (24-bit PCM)    │
                  └───────────────────────────────┘
```

---

## 📂 Project Organization

```
deshimmer/
├── Makefile                    # Project automation commands
├── pyproject.toml              # Build config and dependencies
├── pyrightconfig.json          # Type checker configuration
├── ui_presets.json             # Factory & user-saved parameter presets
├── docs/                       # Granular technical documentation
│   ├── api.md                  # Python API reference
│   ├── architecture.md         # Architecture and stage extension guide
│   ├── autotune_optimizer.md   # NSGA-II optimizer & scoring metrics
│   ├── cli.md                  # Command-line interface manual
│   ├── dsp_pipeline.md         # Core DSP pipeline & psychoacoustic engine
│   ├── processing.md           # Processing contracts & residual verification
│   ├── tonal_repair.md         # Tonal self-consistency & sinusoidal tracking
│   └── ui_guide.md             # Gradio web interface manual
├── src/deshimmer/              # Main package codebase
│   ├── api.py                  # High-level process_audio entry point
│   ├── cli.py                  # Headless batch CLI implementation
│   ├── core/                   # Core DSP processing engine & stages
│   ├── tonal/                  # Sinusoidal tracking & harmonic field
│   ├── autotune/               # Automated scene analysis & NSGA-II optimizer
│   └── ui/                     # Gradio web interface
└── tests/                      # Automated unit and regression test suite
```

---

## ⚙️ Makefile Reference

| Target | Description |
|:---|:---|
| **`make run`** | Bootstraps virtual environment if needed and starts the interactive Gradio UI. |
| **`make test`** | Executes all 27 unit and regression tests via `pytest`. |
| **`make cli`** | Runs the CLI interface. Pass flags via `ARGS="..."` (e.g. `make cli ARGS="--help"`). |
| **`make install`** | Sets up the virtual environment (`.venv`) and installs dependencies via `uv`. |
| **`make clean`** | Removes cache files, bytecode (`__pycache__`), and build artifacts. |
| **`make help`** | Displays the list of available Makefile targets. |

---

## 🧪 Testing & Verification

Run the test suite:

```bash
make test
# or: uv run pytest tests -v
```

The automated test suite contains 27 test cases (with 78 subtests) covering:
- Exact phase invertibility and wet/dry mathematical contracts.
- L/R micro-delay alignment bounds within 3 ms.
- Dynamic spectral carving channel broadcasting.
- Optuna multi-objective metric evaluation and failure resilience.
- Sinusoidal peak interpolation and harmonic family tracking.
- Web UI handlers, looping crossfade logic, and 24-bit PCM WAV exports.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
