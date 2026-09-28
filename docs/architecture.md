# Codebase Architecture & Extension Guide

Deshimmer is built on a clean, modular Python architecture that strictly isolates digital signal processing algorithms from user interfaces, external dependencies, and execution environments. This document explains the internal design, package layout, and how to extend Deshimmer with new DSP stages.

---

## 1. Directory Structure

```
deshimmer/
├── Makefile                    # Standard project automation (run, test, cli, clean)
├── pyproject.toml              # Build backend (hatchling), dependencies, and tools
├── pyrightconfig.json          # Type checker configuration
├── ui_presets.json             # Factory & user-saved parameter presets
├── docs/                       # Granular documentation suite
│   ├── api.md                  # Python API reference
│   ├── architecture.md         # Architecture and stage extension guide
│   ├── autotune_optimizer.md   # NSGA-II optimizer & scoring metrics
│   ├── cli.md                  # Command-line interface manual
│   ├── dsp_pipeline.md         # Core DSP pipeline & psychoacoustic engine
│   ├── processing.md           # Processing contracts & residual verification
│   ├── tonal_repair.md         # Tonal self-consistency & sinusoidal tracking
│   └── ui_guide.md             # Gradio web interface manual
├── src/deshimmer/              # Canonical package root
│   ├── __init__.py             # Public package exports
│   ├── __main__.py             # python -m deshimmer entry point
│   ├── api.py                  # High-level process_audio entry point
│   ├── cli.py                  # Headless batch CLI implementation
│   ├── core/                   # Core DSP processing engine
│   │   ├── context.py          # DSPContext execution state
│   │   ├── debug.py            # Debug spectrogram rendering & metrics
│   │   ├── hpss.py             # Harmonic-percussive separation filters
│   │   ├── mastering.py        # ITU-R BS.1770 normalizer & true-peak limiter
│   │   ├── measurements.py     # LUFS, RMS, phase instability metrics
│   │   ├── params.py           # Params, MasterParams, DebugParams dataclasses
│   │   ├── pipeline.py         # DSPPipeline orchestrator & process_stft
│   │   ├── psychoacoustic.py   # ATH curves & forward temporal masking
│   │   ├── resynth.py          # High-frequency harmonic resynthesis
│   │   ├── spatial.py          # Inter-channel delay & spectral tilt alignment
│   │   ├── stages/             # Modular DSP stages
│   │   │   ├── base.py         # DSPStage abstract base class
│   │   │   ├── carver.py       # Dynamic spectral carver
│   │   │   ├── decorrelation.py# HF decorrelation
│   │   │   ├── denoise.py      # Minimum-statistics noise subtraction
│   │   │   ├── deresonator.py  # Smart dynamic notch filter
│   │   │   ├── expander.py     # High-frequency downward expander
│   │   │   ├── gain.py         # Composite gain application
│   │   │   ├── noise_resynth.py# Controlled random-phase noise injection
│   │   │   ├── phase_blur.py   # Low-depth phase dispersion
│   │   │   ├── shimmer.py      # Dual-threshold spectral flux attenuator
│   │   │   └── swish.py        # High-frequency phase flutter stabilizer
│   │   └── utils.py            # Linear-dB conversions and windowing helpers
│   ├── tonal/                  # Sinusoidal tracking & harmonic field
│   │   ├── harmonic_field.py   # Multi-partial family regression
│   │   ├── repair.py           # Cubic Hermite phase synthesis
│   │   └── tracking.py         # Peak interpolation & track continuity
│   ├── autotune/               # Automated optimization
│   │   ├── analyzer.py         # Scene analysis & dynamic region picker
│   │   ├── optimizer.py        # Optuna NSGA-II search loop & transition seam
│   │   ├── scoring.py          # 6 weighted objective functions
│   │   └── types.py            # Region, Weights, and report definitions
│   └── ui/                     # Gradio web interface
│       ├── app.py              # Gradio Blocks layout & event callbacks
│       ├── audio.py            # Audio slicing, loop crossfade, PNG spectrograms
│       ├── handlers.py         # Preview runner & 24-bit PCM WAV exporter
│       ├── presets.py          # Factory presets & JSON serialization
│       └── state.py            # UI state dictionary mapping
└── tests/                      # Automated test suite (27 unit/regression tests)
    ├── test_auto_tune.py
    ├── test_deshimmer_api.py
    ├── test_dynamic_spectral_carver.py
    ├── test_harmonic_field.py
    ├── test_master.py
    ├── test_tonal_repair.py
    ├── test_tonal_tracking.py
    ├── test_ui_gradio.py
    └── tonal_fixtures.py       # Known sinusoidal scene generators
```

---

## 2. Core Design Principles

1. **Complete Interface Independence**: The core DSP engine (`deshimmer.core`) and sinusoidal repair engine (`deshimmer.tonal`) have zero dependencies on Gradio, CLI argument parsers, or filesystem side effects. They take NumPy arrays and return NumPy arrays.
2. **Algebraic Invertibility & Delta Contracts**: Setting `mix=0.0` or running in delta listen mode guarantees exact mathematical reproduction ($y = x$ or $y = x - y_{\text{processed}}$) without phase smearing or level shifts.
3. **Hardware-Accelerated JIT**: Performance-critical loops (such as true-peak polyphase FIR filtering and forward temporal masking decay) are accelerated via Numba JIT compilers, allowing near real-time rendering of large multi-channel audio buffers.

---

## 3. How to Implement a New DSP Stage

Adding a new processing algorithm into Deshimmer is clean and non-invasive. Follow these four steps:

### Step 1: Subclass `DSPStage`
Create a new file in `src/deshimmer/core/stages/` (e.g. `my_filter.py`) and implement `DSPStage`:

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import DSPStage
from ..context import DSPContext
from ..params import Params

@dataclass
class MyFilterConfig:
    enabled: bool = False
    cutoff_hz: float = 10000.0

class MyFilterStage(DSPStage):
    def __init__(self, config: MyFilterConfig) -> None:
        super().__init__(name="my_filter")
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> MyFilterStage:
        return cls(MyFilterConfig(
            enabled=getattr(p, "my_filter_enabled", False),
            cutoff_hz=getattr(p, "my_filter_cutoff", 10000.0),
        ))

    def is_enabled(self) -> bool:
        return self.config.enabled

    def process(self, ctx: DSPContext) -> None:
        # ctx.freqs contains center frequencies for all FFT bins
        # ctx.mag contains magnitude spectrogram (bins x frames)
        # ctx.g_total contains the running cumulative gain matrix (bins x frames)
        
        mask = ctx.freqs > self.config.cutoff_hz
        # Apply a smooth attenuation shelf
        attenuation = np.ones_like(ctx.g_total)
        attenuation[mask, :] = 0.5
        
        # Accumulate into context gain map
        ctx.g_total *= attenuation
```

### Step 2: Register in `stages/__init__.py`
Export your new stage and config class in [src/deshimmer/core/stages/__init__.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/stages/__init__.py).

### Step 3: Add Parameters to `Params`
Add the configuration knobs to [src/deshimmer/core/params.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/params.py):

```python
@dataclass(frozen=True)
class Params:
    # Existing parameters...
    my_filter_enabled: bool = False
    my_filter_cutoff: float = 10000.0
```

### Step 4: Add to `build_default_pipeline`
Insert your stage at the appropriate sequence in [src/deshimmer/core/pipeline.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/pipeline.py):

```python
def build_default_pipeline(p: Params) -> DSPPipeline:
    return DSPPipeline([
        ExpanderStage.from_params(p),
        DenoiseStage.from_params(p),
        DeResonatorStage.from_params(p),
        MyFilterStage.from_params(p),  # New stage
        ShimmerSuppressionStage.from_params(p),
        # ...
    ])
```

---

## 4. Testing & Verification

Deshimmer includes 27 unit and regression tests in [tests/](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/tests/).

Run the full test suite using Make:
```bash
make test
```

Or invoke pytest directly:
```bash
uv run pytest tests -v
```

All new DSP stages should include unit tests verifying:
1. **Bypass Invertibility**: When disabled, the stage performs zero allocations and leaves `ctx.g_total` strictly unmodified.
2. **Channel Invariance**: Multichannel audio buffers are processed without shape distortion or inter-channel crosstalk.
3. **Numerical Stability**: Processing handles silence (all zeros) and high dynamic range audio without producing NaNs or Infinities.
