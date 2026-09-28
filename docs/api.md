# Python API Reference

The Deshimmer library exposes a clean, modular Python API in [src/deshimmer/api.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/api.py) and [src/deshimmer/core/](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/) designed for seamless integration into custom audio pipelines, automated DAW scripting, and ML audio preprocessing.

---

## 1. Primary Entry Point: `process_audio`

```python
from deshimmer.api import process_audio
from deshimmer.core import Params, MasterParams, DebugParams
```

### Function Signature
```python
def process_audio(
    x: np.ndarray,
    sr: int,
    *,
    params: Optional[Params] = None,
    master_params: Optional[MasterParams] = None,
    debug_params: Optional[DebugParams] = None,
    include_residuals: bool = False,
    pipeline: Optional[DSPPipeline] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
```

### Parameters
- **`x`** (`np.ndarray`): Audio samples as `float32` array. Supports mono 1D `(samples,)` or stereo 2D `(samples, channels)`. Audio outside $[-1.0, 1.0]$ will be clipped by the mastering stage if enabled.
- **`sr`** (`int`): Audio sample rate in Hz (e.g. 44100, 48000, 96000).
- **`params`** (`Optional[Params]`): DSP algorithm parameters. Defaults to standard factory settings if omitted.
- **`master_params`** (`Optional[MasterParams]`): Mastering, loudness normalization, and peak limiter options.
- **`debug_params`** (`Optional[DebugParams]`): Diagnostic collection settings.
- **`include_residuals`** (`bool`): If `True`, computes and includes forensic difference arrays in the returned `info` dictionary.
- **`pipeline`** (`Optional[DSPPipeline]`): Custom DSP execution pipeline. If `None`, the default 10-stage pipeline is constructed.

### Returns
A tuple `(y, info)`:
- **`y`** (`np.ndarray`): Processed audio array matching the shape and dimensions of `x`.
- **`info`** (`dict[str, Any]`): Detailed execution diagnostics:
  - `info["measure_in"]`: Pre-processing measurements (`lufs`, `rms_dbfs`, `true_peak_db`, `phase_instability`).
  - `info["measure_rep"]`: Measurements after spectral and tonal repair.
  - `info["measure_out"]`: Final measurements after delivery mastering.
  - `info["tonal"]`: Diagnostics from the sinusoidal tracking stage (`changed_partials`, `coherence_gain`).
  - `info["residuals"]` (when requested):
    - `residuals["artifact_repair_diff"]`: Difference array from spectral and tonal repair only.
    - `residuals["total_enhancement_diff"]`: Total difference including mastering and spatial enhancement ($x - y$).

---

## 2. Configuration Dataclasses

### `Params` ([src/deshimmer/core/params.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/params.py))
Controls all core DSP stage behaviors:

```python
from dataclasses import dataclass
from deshimmer.core import Params

p = Params(
    # Time-frequency framing
    n_fft=2048,
    hop=512,
    
    # Shimmer suppression
    start_hz=5100.0,
    end_hz=7200.0,
    thr_db=8.0,
    slope=0.6,
    density_lo=0.02,
    density_hi=0.15,
    noise_resynth=0.0,
    mix=1.0,
    
    # Tonal self-consistency
    tonal_repair=0.5,
    tonal_max_shift_cents=25.0,
    
    # Minimum-statistics denoise
    denoise=0.0,
    dn_start_hz=3000.0,
    dn_end_hz=16000.0,
    dn_floor_db=-24.0,
    dn_minwin_ms=1000.0,
    dn_freq_smooth_bins=5,
    
    # Smart de-resonator
    deres=0.0,
    deq_start_hz=180.0,
    deq_end_hz=12000.0,
    deq_thr_db=4.0,
    deq_max_att_db=12.0,
    deq_persist_ms=1000.0,
    deq_time_floor=False,
    
    # Expander & HPSS
    expander=False,
    exp_threshold_db=-45.0,
    hpss=False,
    phase_blur=0.0,
    
    # High-frequency resynthesis
    hf_resynth=False,
    hf_drive=2.0,
    hf_mix=0.35,
)
```

### `MasterParams` ([src/deshimmer/core/params.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/core/params.py))
Configures post-repair delivery mastering:

```python
from deshimmer.core import MasterParams

mp = MasterParams(
    enabled=True,
    target_lufs=-14.0,          # Target integrated loudness
    ceiling_dbtp=-1.0,           # True-peak ceiling
    tp_os=4,                     # Polyphase oversampling factor
    hp_hz=20.0,                  # Subsonic high-pass filter frequency
    norm_max_gain_db=12.0,       # Maximum boost permitted during normalization
    norm_max_atten_db=-24.0,     # Maximum attenuation permitted
    lim_lookahead_ms=5.0,        # Limiter lookahead buffer duration
    lim_release_ms=100.0,        # Limiter gain release time constant
)
```

---

## 3. Custom Pipeline Construction

You can build custom pipelines to isolate stages, reorder processing, or benchmark individual DSP blocks:

```python
import numpy as np
from deshimmer.core import (
    DSPPipeline,
    DSPContext,
    Params,
    DeResonatorStage,
    SwishRepairStage,
    GainApplicationStage,
    process_stft,
)

sr = 44100
audio = np.random.normal(0, 0.1, (sr * 2, 2)).astype(np.float32)
p = Params(n_fft=1024, hop=256, deres=0.8, deq_thr_db=4.0, swish_repair=0.5)

# Construct a 3-stage custom pipeline
pipeline = DSPPipeline([
    DeResonatorStage.from_params(p),
    SwishRepairStage.from_params(p),
    GainApplicationStage.from_params(p),
])

# Process audio through the custom pipeline
repaired = process_stft(audio, sr, p, pipeline=pipeline)
```

---

## 4. Auto-Tune & Optimizer API

Automate parameter tuning programmatically in [src/deshimmer/autotune/](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/autotune/):

```python
from deshimmer.autotune import analyze, pick_regions, refine
from deshimmer.core import Params

# Step 1: Pre-analyze audio to locate artifact bands and representative regions
analysis = analyze(audio, sr)
regions = pick_regions(audio, sr, max_regions=3, region_dur=2.0)

# Step 2: Run NSGA-II optimization
initial_params = Params()
best_params, report = refine(
    audio,
    sr,
    regions=regions,
    initial_params=initial_params,
    aggressiveness=0.5,   # 0.0 (safe) to 1.0 (aggressive)
    n_trials=25,
    timeout_s=30.0,
)

print(f"Optimized shimmer band: {best_params.start_hz:.1f} - {best_params.end_hz:.1f} Hz")
print(f"Optimized threshold: {best_params.thr_db:.1f} dB")
```
