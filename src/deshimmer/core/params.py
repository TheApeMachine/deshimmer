"""Configuration parameters for the Deshimmer DSP engine, mastering, and debugging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Params:
    # --- Original shimmer band ---
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0

    n_fft: int = 2048
    hop: int = 512

    # Noise-likeness gate via spectral flatness
    flat_start: float = 0.25
    flat_end: float = 0.70

    # Birdie suppression (within start_hz..end_hz)
    freq_med_bins: int = 9
    thr_db: float = 8.0
    slope: float = 0.6

    # If MANY bins exceed threshold, treat as real broadband event; do less
    density_lo: float = 0.02
    density_hi: float = 0.15

    # Transient protect (based on band energy jump)
    flux_thr_db: float = 6.0
    flux_range_db: float = 8.0

    # Optional random-phase blend (only in noise-like frames)
    noise_resynth: float = 0.0  # 0..1

    # Wet/dry
    mix: float = 1.0

    # Optional engineering stages, separate from artifact repair for residual audits.
    enhance: bool = True

    # Self-referenced harmonic repair is active in the normal processing path.
    tonal_repair: float = 0.5
    tonal_max_shift_cents: float = 25.0

    # Padding & fade
    pad: bool = True
    fade_ms: float = 5.0

    seed: int = 0

    # ----------------------------------------------------------------------
    # Smart noise-floor management: spectral denoise
    # ----------------------------------------------------------------------
    denoise: float = 0.0             # 0..1 overall strength
    dn_start_hz: float = 120.0
    dn_end_hz: float = 16000.0
    dn_edge_hz: float = 200.0

    dn_floor_db: float = -18.0       # minimum gain floor (dB)
    dn_psd_smooth_ms: float = 50.0   # smooth PSD before min-tracking
    dn_minwin_ms: float = 400.0      # minimum-statistics window for tracking minima
    dn_up_db_per_s: float = 3.0      # how fast noise estimate can rise
    dn_attack_ms: float = 5.0
    dn_release_ms: float = 120.0
    dn_freq_smooth_bins: int = 3     # smooth gain across frequency

    # ----------------------------------------------------------------------
    # Smart dynamic EQ: de-resonator (taste-neutral, subtractive only)
    # ----------------------------------------------------------------------
    deres: float = 0.0               # 0..1 strength
    deq_start_hz: float = 180.0
    deq_end_hz: float = 12000.0
    deq_edge_hz: float = 150.0

    deq_freq_med_bins: int = 31      # median width across frequency for baseline
    deq_thr_db: float = 6.0          # residual threshold above local median
    deq_slope: float = 0.7           # how hard to push down peaks above threshold
    deq_max_att_db: float = 8.0      # cap reduction (safety)

    deq_density_lo: float = 0.03
    deq_density_hi: float = 0.20

    # persistence favors stationary resonances over moving harmonics
    deq_persist_ms: float = 600.0
    deq_persist_thr_db: float = 2.5

    deq_freq_smooth_bins: int = 5
    deq_tonal_boost_db: float = 6.0  # raises threshold when frame is tonal

    # ----------------------------------------------------------------------
    # Optional: time-stabilized baseline for stationary ringing ("whine" lines)
    # ----------------------------------------------------------------------
    deq_time_floor: bool = False
    deq_floor_smooth_ms: float = 80.0       # smooth PSD before floor tracking (ms)
    deq_floor_rise_db_per_s: float = 1.0    # how fast the floor can rise
    deq_floor_thr_db: float = 3.0           # threshold for "high stationary floor"

    # Listen to removed only
    delta_listen: bool = False

    # ----------------------------------------------------------------------
    # Optional: downward expander in an artifact band
    # ----------------------------------------------------------------------
    expander: bool = False
    exp_start_hz: float = 3000.0
    exp_end_hz: float = 8000.0
    exp_threshold_db: float = -45.0
    exp_ratio: float = 2.0
    exp_attack_ms: float = 10.0
    exp_release_ms: float = 150.0

    # ----------------------------------------------------------------------
    # Optional: HPSS-ish harmonic mask inside a band
    # ----------------------------------------------------------------------
    hpss: bool = False
    hpss_start_hz: float = 3000.0
    hpss_end_hz: float = 8000.0
    hpss_time_frames: int = 21
    hpss_freq_bins: int = 17
    hpss_harmonic_only: bool = True
    hpss_protect_percussive: float = 0.0

    # ----------------------------------------------------------------------
    # Repair philosophy: magnitude inpainting vs pure attenuation
    # ----------------------------------------------------------------------
    magnitude_inpaint: bool = True
    deq_inpaint: bool = True

    # Combined-stage safety cap (per bin)
    total_att_cap_db: float = 12.0
    nuclear_mode: bool = False

    # Mid/Side: run detectors on mono; apply scaled repair on Side channel
    ms_process: bool = False
    ms_side_scale: float = 0.35

    # ----------------------------------------------------------------------
    # Optional: phase blur in a selected band
    # ----------------------------------------------------------------------
    phase_blur: float = 0.0
    pb_start_hz: float = 3000.0
    pb_end_hz: float = 8000.0
    pb_harmonic_only: bool = True

    # ----------------------------------------------------------------------
    # Swish repair: adaptive phase coherence smoothing
    # ----------------------------------------------------------------------
    swish_repair: float = 0.0
    swish_start_hz: float = 3500.0
    swish_end_hz: float = 14000.0
    swish_time_amt: float = 0.55
    swish_freq_amt: float = 0.30
    swish_time_win: int = 7
    swish_freq_win: int = 5
    swish_transient_protect: float = 0.85
    swish_harmonic_protect: float = 0.45

    # HF stereo decorrelation (break synthetic L/R phase lock in upper band)
    hf_decorrelate: float = 0.0
    hf_dec_start_hz: float = 4500.0
    hf_dec_end_hz: float = 16000.0

    # ----------------------------------------------------------------------
    # Optional: HF resynthesis
    # ----------------------------------------------------------------------
    hf_resynth: bool = False
    hf_lp_hz: float = 3000.0
    hf_src_lo_hz: float = 1000.0
    hf_src_hi_hz: float = 2000.0
    hf_drive: float = 2.0
    hf_hp_hz: float = 3000.0
    hf_mix: float = 0.35
    hf_tilt_lp_hz: float = 12000.0
    hf_zero_phase: bool = True
    hf_confidence_blend: bool = True

    # ----------------------------------------------------------------------
    # Psychoacoustic forward temporal masking
    # ----------------------------------------------------------------------
    temporal_masking: bool = True
    mask_decay_ms: float = 15.0
    mask_flat_gate: bool = True

    # ----------------------------------------------------------------------
    # Sound Goodizer: Self-adapting tonal balance
    # ----------------------------------------------------------------------
    tonal_balance: float = 0.0          # 0..1 overall strength
    tb_low_db: float = 0.0              # -3.0..+3.0 dB (sub/bass ~40-250 Hz)
    tb_low_mid_db: float = 0.0          # -3.0..+3.0 dB (warmth/boxiness ~200-900 Hz)
    tb_mid_db: float = 0.0              # -3.0..+3.0 dB (presence/body ~700-3000 Hz)
    tb_presence_db: float = 0.0         # -3.0..+3.0 dB (clarity/harshness ~2500-8000 Hz)
    tb_air_db: float = 0.0              # -3.0..+3.0 dB (openness/sheen >7000 Hz)

    # ----------------------------------------------------------------------
    # Sound Goodizer: Transient & microdynamic punch restoration
    # ----------------------------------------------------------------------
    transient_punch: float = 0.0        # 0..1 overall strength
    punch_attack_boost_db: float = 2.0  # 1.0..4.0 dB max attack expansion
    punch_low: float = 0.5              # low-end punch weight (kick impact)
    punch_high: float = 0.5             # high-end punch weight (snare/plucks)

    # ----------------------------------------------------------------------
    # Sound Goodizer: Low-end coherence & mono punch
    # ----------------------------------------------------------------------
    bass_mono: bool = False
    bass_mono_hz: float = 120.0         # collapse side energy below this Hz
    bass_tighten: float = 0.0           # 0..1 attenuate sustained boom overhang

    # ----------------------------------------------------------------------
    # Sound Goodizer: Stereo image translation & mono safety
    # ----------------------------------------------------------------------
    stereo_width: float = 1.0           # 0.5..1.5 width scaling (1.0 = neutral)
    stereo_safe_mono: bool = False      # preserve mono compatibility, suppress anti-phase

    # ----------------------------------------------------------------------
    # Sound Goodizer: Musical harmonic overtone enhancement
    # ----------------------------------------------------------------------
    harmonic_enhance: float = 0.0       # 0..0.5 strength
    harmonic_start_hz: float = 3500.0   # start frequency for subtle harmonic sheen


@dataclass
class MasterParams:
    enabled: bool = False

    dc_remove: bool = True
    hp_hz: float = 20.0
    hp_order: int = 2

    # Loudness normalization
    target_lufs: Optional[float] = -14.0
    target_rms_dbfs: Optional[float] = -16.0
    norm_max_gain_db: float = 12.0
    norm_max_atten_db: float = 24.0

    # Limiter / true-peak
    ceiling_dbtp: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 100.0
    os_factor: int = 4


@dataclass
class DebugParams:
    enabled: bool = False
    debug_dir: Optional[str] = None
    stride: int = 8

    # Spectrogram visuals
    spec_n_fft: int = 2048
    spec_hop: int = 512
    spec_max_frames: int = 2200
    spec_max_hz: float = 20000.0

    save_npz: bool = True
    save_png: bool = True
