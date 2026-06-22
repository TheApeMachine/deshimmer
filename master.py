#!/usr/bin/env python3
"""
deshimmer6.py

Anti-shimmer + smart repair + optional delivery mastering:

Core (STFT domain):
- Target-band "birdie/shimmer" suppression via local-median residuals.
- Smart spectral denoise (noise-floor management) using a minimum-statistics-ish noise PSD tracker
  and Wiener-like gain with time/frequency smoothing.
- Smart de-resonator (dynamic EQ / dynamic notch) that attenuates persistent narrow peaks across a band.

Optional post ("deliver"):
- Loudness normalization toward a target LUFS (ITU-R BS.1770 via pyloudnorm when available).
- True-peak-ish limiting via oversampling + lookahead peak limiter.

Debug mode:
- Writes a debug folder with .json summary, .npz internals, and .png visualizations
  (attenuation maps + spectrogram before/after/diff).

Notes:
- This is designed to be subtractive and conservative. It "fixes" more than it "flavors".
- Still: auto-repair can be wrong on some material. Use --debug to see what it's doing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d, maximum_filter1d
from scipy.signal import butter, sosfiltfilt, sosfilt, resample_poly, lfilter, lfilter_zi, stft, istft


# -----------------------------
# Utility conversions
# -----------------------------
def _db_to_lin(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db) / 20.0)


def _lin_to_db(x: float | np.ndarray, eps: float = 1e-12) -> float | np.ndarray:
    return 20.0 * np.log10(np.asarray(x) + eps)


def _as_2d(x: np.ndarray) -> np.ndarray:
    return x[:, None] if x.ndim == 1 else x


def band_from_center(center_hz: float, width_cents: float) -> Tuple[float, float]:
    half = width_cents * 0.5
    ratio = 2.0 ** (half / 1200.0)
    return center_hz / ratio, center_hz * ratio


def _edge_taper(freqs: np.ndarray, band_idx: np.ndarray, start_hz: float, end_hz: float, edge_hz: float) -> np.ndarray:
    """Cosine taper inside edge_hz of the band edges."""
    w = np.ones(band_idx.size, dtype=np.float32)
    edge = float(max(0.0, edge_hz))
    if edge <= 0.0 or band_idx.size == 0:
        return w
    fb = freqs[band_idx].astype(np.float32)
    lo, hi = float(start_hz), float(end_hz)

    m = fb < lo + edge
    if np.any(m):
        rel = (fb[m] - lo) / edge
        w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1))

    m2 = fb > hi - edge
    if np.any(m2):
        rel = (hi - fb[m2]) / edge
        w[m2] = np.minimum(w[m2], 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1)))

    return w


def _frame_coeff(hop: int, sr: int, ms: float) -> float:
    """Per-frame EMA coefficient for time constant ms."""
    tau = max(1e-4, float(ms) / 1000.0)
    return float(math.exp(-float(hop) / (float(sr) * tau)))


def _butter_sos(sr: int, kind: str, cutoff_hz, order: int = 2):
    nyq = 0.5 * sr
    if kind in ("lowpass", "highpass"):
        wn = float(cutoff_hz) / nyq
        wn = float(np.clip(wn, 1e-6, 0.999999))
        return butter(int(order), wn, btype=kind, output="sos")
    if kind == "bandpass":
        lo, hi = cutoff_hz
        lo = float(np.clip(lo / nyq, 1e-6, 0.999999))
        hi = float(np.clip(hi / nyq, 1e-6, 0.999999))
        if hi <= lo:
            raise ValueError("Invalid bandpass cutoffs")
        return butter(int(order), [lo, hi], btype="bandpass", output="sos")
    raise ValueError("Unknown filter kind")


def _onepole_lp(x: np.ndarray, a: float) -> np.ndarray:
    """One-pole low-pass: y[n] = (1-a)*x[n] + a*y[n-1]  (0<a<1)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    a = float(np.clip(a, 0.0, 0.999999))
    b = np.array([1.0 - a], dtype=np.float32)
    A = np.array([1.0, -a], dtype=np.float32)
    zi = lfilter_zi(b, A) * float(x[0])
    y, _ = lfilter(b, A, x, zi=zi)
    return y.astype(np.float32, copy=False)


# -----------------------------
# Parameters
# -----------------------------
@dataclass
class Params:
    # --- Original shimmer band ---
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0

    n_fft: int = 2048
    hop: int = 512

    # Noise-likeness gate via spectral flatness (geo/arith of power spectrum).
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

    # Padding & fade
    pad: bool = True
    fade_ms: float = 5.0

    seed: int = 0

    # ----------------------------------------------------------------------
    # Smart noise-floor management: spectral denoise (taste-neutral)
    # ----------------------------------------------------------------------
    denoise: float = 0.0             # 0..1 overall strength
    dn_start_hz: float = 120.0
    dn_end_hz: float = 16000.0
    dn_edge_hz: float = 200.0

    dn_floor_db: float = -18.0       # minimum gain floor (dB), avoids dead-silent artifacts
    dn_psd_smooth_ms: float = 50.0   # smooth PSD before min-tracking
    dn_minwin_ms: float = 400.0      # minimum-statistics window for tracking minima
    dn_up_db_per_s: float = 3.0      # how fast noise estimate can rise (dB/s, power-domain approx)
    dn_attack_ms: float = 5.0
    dn_release_ms: float = 120.0
    dn_freq_smooth_bins: int = 3     # smooth gain across frequency (reduce musical noise)

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

    # If too many bins exceed threshold, assume it's legit broadband/tonal content -> back off
    deq_density_lo: float = 0.03
    deq_density_hi: float = 0.20

    # persistence favors stationary resonances over moving harmonics
    deq_persist_ms: float = 600.0
    deq_persist_thr_db: float = 2.5

    deq_freq_smooth_bins: int = 5
    deq_tonal_boost_db: float = 6.0  # raises threshold when frame is tonal (based on flatness)

    # ----------------------------------------------------------------------
    # Optional: time-stabilized baseline for stationary ringing ("whine" lines)
    # ----------------------------------------------------------------------
    deq_time_floor: bool = False
    deq_floor_smooth_ms: float = 80.0       # smooth PSD before floor tracking (ms)
    deq_floor_rise_db_per_s: float = 1.0    # how fast the floor can rise (dB/s, power-domain approx)
    deq_floor_thr_db: float = 3.0           # threshold for "high stationary floor" vs local floor baseline

    # Listen to removed only (useful for dialing in)
    delta_listen: bool = False

    # ----------------------------------------------------------------------
    # Optional: downward expander in an artifact band (targets grit in tails)
    # ----------------------------------------------------------------------
    expander: bool = False
    exp_start_hz: float = 3000.0
    exp_end_hz: float = 8000.0
    exp_threshold_db: float = -45.0  # band level in dB (power)
    exp_ratio: float = 2.0          # 2:1 downward expander
    exp_attack_ms: float = 10.0
    exp_release_ms: float = 150.0

    # ----------------------------------------------------------------------
    # Optional: HPSS-ish harmonic mask inside a band (protect percussive transients)
    # ----------------------------------------------------------------------
    hpss: bool = False
    hpss_start_hz: float = 3000.0
    hpss_end_hz: float = 8000.0
    hpss_time_frames: int = 21     # median over time (odd recommended)
    hpss_freq_bins: int = 17       # median over freq (odd recommended)
    hpss_harmonic_only: bool = True  # apply processing only to harmonic (horizontal) component
    hpss_protect_percussive: float = 0.0  # 0..1 reduce repair depth on percussive bins

    # ----------------------------------------------------------------------
    # Repair philosophy: magnitude inpainting vs pure attenuation
    # ----------------------------------------------------------------------
    magnitude_inpaint: bool = True   # cap spikes to local median instead of only cutting gain
    deq_inpaint: bool = True         # de-resonator uses inpaint when magnitude_inpaint is on

    # Combined-stage safety cap (per bin, product of stage gains)
    total_att_cap_db: float = 12.0
    nuclear_mode: bool = False       # disables combined attenuation cap

    # Mid/Side: run detectors on mono; apply scaled repair on Side channel
    ms_process: bool = False
    ms_side_scale: float = 0.35

    # ----------------------------------------------------------------------
    # Optional: phase blur (random-phase blend) in a selected band (texture masking)
    # ----------------------------------------------------------------------
    phase_blur: float = 0.0
    pb_start_hz: float = 3000.0
    pb_end_hz: float = 8000.0
    pb_harmonic_only: bool = True

    # ----------------------------------------------------------------------
    # Swish repair: adaptive inter-frame / inter-bin phase coherence smoothing
    # Targets moving "swish" from neural decoders (phase incoherence, not birdies).
    # ----------------------------------------------------------------------
    swish_repair: float = 0.0
    swish_start_hz: float = 3500.0
    swish_end_hz: float = 14000.0
    swish_time_amt: float = 0.55       # inter-frame phase smoothing blend
    swish_freq_amt: float = 0.30       # inter-bin phase smoothing blend
    swish_time_win: int = 7            # STFT frames (odd)
    swish_freq_win: int = 5            # frequency bins (odd)
    swish_transient_protect: float = 0.85
    swish_harmonic_protect: float = 0.45

    # HF stereo decorrelation (break synthetic L/R phase lock in upper band)
    hf_decorrelate: float = 0.0
    hf_dec_start_hz: float = 4500.0
    hf_dec_end_hz: float = 16000.0

    # ----------------------------------------------------------------------
    # Optional: "Nuclear" HF resynthesis (remove HF and re-create from low band)
    # ----------------------------------------------------------------------
    hf_resynth: bool = False
    hf_lp_hz: float = 3000.0      # low-pass cutoff for "clean" base
    hf_src_lo_hz: float = 1000.0  # source band to excite
    hf_src_hi_hz: float = 2000.0
    hf_drive: float = 2.0         # tanh drive
    hf_hp_hz: float = 3000.0      # high-pass for generated harmonics
    hf_mix: float = 0.35          # mix generated HF back in
    hf_tilt_lp_hz: float = 12000.0  # soften buzz: low-pass on generated HF
    hf_zero_phase: bool = True      # offline nicety; set False for realtime-ish chunk processing
    hf_confidence_blend: bool = True  # blend resynth only where artifact confidence is high


# Last STFT-frame artifact confidence (F_band, T) for HF resynth blending
_LAST_ARTIFACT_CONF_FRAMES: Optional[np.ndarray] = None


@dataclass
class MasterParams:
    enabled: bool = False

    dc_remove: bool = True
    hp_hz: float = 20.0
    hp_order: int = 2

    # Loudness normalization
    target_lufs: Optional[float] = -14.0  # None disables
    target_rms_dbfs: Optional[float] = -16.0  # fallback if pyloudnorm not available
    norm_max_gain_db: float = 12.0
    norm_max_atten_db: float = 24.0

    # Limiter / true-peak-ish
    ceiling_dbtp: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 100.0
    os_factor: int = 4  # ITU/BS.1770 commonly uses 4x for true-peak estimation


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


# -----------------------------
# Debug collector
# -----------------------------
class DebugCollector:
    def __init__(
        self,
        sr: int,
        freqs: np.ndarray,
        dn_idx: np.ndarray,
        deq_idx: np.ndarray,
        sh_idx: np.ndarray,
        stride: int,
        pad_offset: int,
        duration_s: float,
    ):
        self.sr = int(sr)
        self.freqs = freqs
        self.dn_idx = dn_idx
        self.deq_idx = deq_idx
        self.sh_idx = sh_idx
        self.stride = int(max(1, stride))
        self.pad_offset = int(pad_offset)
        self.duration_s = float(duration_s)

        self.frame_i = 0

        self.t: List[float] = []
        self.w_noise: List[float] = []
        self.w_trans: List[float] = []
        self.flux_db: List[float] = []
        self.band_db: List[float] = []

        self.dn_depth: List[float] = []
        self.deq_depth: List[float] = []
        self.sh_depth: List[float] = []

        self.att_dn_db: List[np.ndarray] = []
        self.att_deq_db: List[np.ndarray] = []
        self.att_sh_db: List[np.ndarray] = []

        self.noise_psd_dn_db: List[np.ndarray] = []  # estimated noise PSD in denoise band (dB)

    def want(self) -> bool:
        return (self.frame_i % self.stride) == 0

    def push_meta(self, s: int, n_fft: int, w_noise: float, w_trans: float, flux_db: float, band_db: float,
                  dn_depth: float, deq_depth: float, sh_depth: float):
        # frame center time relative to *original* (unpadded) signal
        t_center = (float(s) + 0.5 * float(n_fft) - float(self.pad_offset)) / float(self.sr)
        if t_center < 0.0:
            t_center = 0.0
        if t_center > self.duration_s:
            t_center = self.duration_s
        self.t.append(float(t_center))
        self.w_noise.append(float(w_noise))
        self.w_trans.append(float(w_trans))
        self.flux_db.append(float(flux_db))
        self.band_db.append(float(band_db))
        self.dn_depth.append(float(dn_depth))
        self.deq_depth.append(float(deq_depth))
        self.sh_depth.append(float(sh_depth))

    def push_att(self, att_dn_db: Optional[np.ndarray], att_deq_db: Optional[np.ndarray], att_sh_db: Optional[np.ndarray]):
        if att_dn_db is not None:
            self.att_dn_db.append(att_dn_db.astype(np.float32, copy=False))
        if att_deq_db is not None:
            self.att_deq_db.append(att_deq_db.astype(np.float32, copy=False))
        if att_sh_db is not None:
            self.att_sh_db.append(att_sh_db.astype(np.float32, copy=False))

    def push_noise_psd_dn(self, noise_psd_dn_db: Optional[np.ndarray]):
        if noise_psd_dn_db is not None:
            self.noise_psd_dn_db.append(noise_psd_dn_db.astype(np.float32, copy=False))

    def step(self):
        self.frame_i += 1

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["t"] = np.array(self.t, dtype=np.float32)
        out["w_noise"] = np.array(self.w_noise, dtype=np.float32)
        out["w_trans"] = np.array(self.w_trans, dtype=np.float32)
        out["flux_db"] = np.array(self.flux_db, dtype=np.float32)
        out["band_db"] = np.array(self.band_db, dtype=np.float32)
        out["dn_depth"] = np.array(self.dn_depth, dtype=np.float32)
        out["deq_depth"] = np.array(self.deq_depth, dtype=np.float32)
        out["sh_depth"] = np.array(self.sh_depth, dtype=np.float32)

        if len(self.att_dn_db) > 0:
            out["f_dn"] = self.freqs[self.dn_idx].astype(np.float32)
            out["att_dn_db"] = np.stack(self.att_dn_db, axis=1)  # (F, T)
        if len(self.att_deq_db) > 0:
            out["f_deq"] = self.freqs[self.deq_idx].astype(np.float32)
            out["att_deq_db"] = np.stack(self.att_deq_db, axis=1)
        if len(self.att_sh_db) > 0:
            out["f_sh"] = self.freqs[self.sh_idx].astype(np.float32)
            out["att_sh_db"] = np.stack(self.att_sh_db, axis=1)
        if len(self.noise_psd_dn_db) > 0:
            out["noise_psd_dn_db"] = np.stack(self.noise_psd_dn_db, axis=1)

        return out


# -----------------------------
# Measurements (for debug)
# -----------------------------
def measure_true_peak_db(x: np.ndarray, os_factor: int = 4) -> float:
    """Approx true-peak via oversampling + sample peak on oversampled signal."""
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    osf = int(max(1, os_factor))
    if osf > 1:
        x_os = resample_poly(x2, osf, 1, axis=0).astype(np.float32, copy=False)
    else:
        x_os = x2
    peak = float(np.max(np.abs(x_os))) if x_os.size else 0.0
    return float(_lin_to_db(peak))


def measure_rms_dbfs(x: np.ndarray) -> float:
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    rms = float(np.sqrt(np.mean(x2 * x2) + 1e-12))
    return float(_lin_to_db(rms))


def measure_lufs(x: np.ndarray, sr: int) -> Optional[float]:
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(int(sr))
        return float(meter.integrated_loudness(_as_2d(x)))
    except Exception:
        return None


# -----------------------------
# Post: loudness normalize + limiter
# -----------------------------
def _peak_limiter(x: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float, release_ms: float) -> np.ndarray:
    """
    Lookahead peak limiter:
    - Compute forward-looking local max over lookahead window.
    - Compute required gain to keep peaks under ceiling.
    - Apply "attack instant / release slow" smoothing via low-pass+min trick.
    """
    eps = 1e-12
    x2 = _as_2d(x).astype(np.float32, copy=False)

    ceiling = float(_db_to_lin(float(ceiling_db)))
    ceiling = float(np.clip(ceiling, 1e-6, 1.0))

    sc = np.max(np.abs(x2), axis=1).astype(np.float32)

    la = int(float(sr) * (float(lookahead_ms) / 1000.0))
    la = max(0, la)
    size = la + 1
    if size <= 1:
        peak = sc
    else:
        # Forward window: [n, n+la]
        origin = la // 2  # shifts window to the right
        peak = maximum_filter1d(sc, size=size, origin=origin, mode="nearest").astype(np.float32)

    req = np.minimum(1.0, ceiling / (peak + eps)).astype(np.float32)

    rel = max(1.0, float(release_ms)) / 1000.0
    a_rel = math.exp(-1.0 / (float(sr) * rel))

    req_lp = _onepole_lp(req, a_rel)
    g = np.minimum(req, req_lp).astype(np.float32)

    return (x2 * g[:, None]).astype(np.float32, copy=False)


def loudness_normalize(x: np.ndarray, sr: int, mp: MasterParams, debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loudness normalization:
    - Prefer pyloudnorm (BS.1770) if available.
    - Otherwise fallback to RMS.
    - Clamp gain.
    """
    info: Dict[str, Any] = {}
    x2 = _as_2d(np.asarray(x, dtype=np.float32))

    max_up = float(max(0.0, mp.norm_max_gain_db))
    max_dn = float(max(0.0, mp.norm_max_atten_db))

    gain_db: Optional[float] = None
    lufs = None
    if mp.target_lufs is not None:
        lufs = measure_lufs(x2, sr)
        if lufs is not None and math.isfinite(lufs):
            gain_db = float(mp.target_lufs) - float(lufs)
            info["lufs_in"] = float(lufs)
            info["target_lufs"] = float(mp.target_lufs)

    if gain_db is None:
        # RMS fallback (not perceptual, but better than nothing)
        if mp.target_rms_dbfs is None:
            info["norm"] = "disabled"
            return x2, info
        rms_db = measure_rms_dbfs(x2)
        gain_db = float(mp.target_rms_dbfs) - float(rms_db)
        info["rms_dbfs_in"] = float(rms_db)
        info["target_rms_dbfs"] = float(mp.target_rms_dbfs)

    gain_db = float(np.clip(gain_db, -max_dn, +max_up))
    info["gain_db_applied"] = float(gain_db)

    g = float(_db_to_lin(gain_db))
    y = (x2 * g).astype(np.float32, copy=False)
    if debug:
        info["peak_after_norm_dbfs"] = float(_lin_to_db(np.max(np.abs(y)) + 1e-12))
    return y, info


def master_post(x: np.ndarray, sr: int, mp: MasterParams, debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": bool(mp.enabled)}
    if not mp.enabled:
        return x, info

    y = _as_2d(np.asarray(x, dtype=np.float32))

    # DC remove
    if mp.dc_remove:
        y = (y - np.mean(y, axis=0, keepdims=True)).astype(np.float32, copy=False)
    info["dc_remove"] = bool(mp.dc_remove)

    # Subsonic high-pass
    if mp.hp_hz and mp.hp_hz > 0.0:
        nyq = 0.5 * sr
        if mp.hp_hz < nyq - 10.0:
            sos = _butter_sos(sr, "highpass", float(mp.hp_hz), order=int(max(1, mp.hp_order)))
            y = sosfiltfilt(sos, y, axis=0).astype(np.float32, copy=False)
            info["hp_hz"] = float(mp.hp_hz)
            info["hp_order"] = int(mp.hp_order)

    # Loudness normalization
    y, norm_info = loudness_normalize(y, sr, mp, debug=debug)
    info["normalize"] = norm_info

    # Limiter (oversampled for true-peak-ish)
    osf = int(max(1, mp.os_factor))
    info["limiter_os_factor"] = osf
    if osf > 1:
        y_os = resample_poly(y, osf, 1, axis=0).astype(np.float32, copy=False)
        y_os = _peak_limiter(y_os, sr * osf, mp.ceiling_dbtp, mp.lookahead_ms, mp.release_ms)
        y2 = resample_poly(y_os, 1, osf, axis=0).astype(np.float32, copy=False)
        # Align length
        if y2.shape[0] > y.shape[0]:
            y2 = y2[: y.shape[0], :]
        elif y2.shape[0] < y.shape[0]:
            y2 = np.pad(y2, ((0, y.shape[0] - y2.shape[0]), (0, 0)), mode="constant")
        y = y2
    else:
        y = _peak_limiter(y, sr, mp.ceiling_dbtp, mp.lookahead_ms, mp.release_ms)

    info["ceiling_dbtp"] = float(mp.ceiling_dbtp)
    info["lookahead_ms"] = float(mp.lookahead_ms)
    info["release_ms"] = float(mp.release_ms)

    # Safety: prevent sample peak > 0 dBFS
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999999:
        y = (y / peak * 0.999999).astype(np.float32)
        info["post_scale_db"] = float(_lin_to_db(0.999999 / peak))

    return y.squeeze(), info


# -----------------------------
# Core STFT repair
# -----------------------------

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba missing (though we installed it)
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def _numba_exp_env(g_target, n_cols, att, rel):
    # g_target: (T,)
    g_env = np.ones_like(g_target)
    g_sm = 1.0
    # Cast scalars to match array precision (likely float32)
    att_f = float(att)
    rel_f = float(rel)
    for i in range(n_cols):
        curr = g_target[i]
        if curr < g_sm:
            g_sm = att_f * g_sm + (1.0 - att_f) * curr
        else:
            g_sm = rel_f * g_sm + (1.0 - rel_f) * curr
        g_env[i] = g_sm
    return g_env

@jit(nopython=True)
def _numba_dn_noise_est(curr_min, n_blocks, curr_noise_vec, rise):
    # curr_min: (F, n_blocks)
    # curr_noise_vec: (F,) initialized
    output = np.zeros_like(curr_min)
    # Cast rise to float32 to match curr_noise_vec precision
    rise_f = np.float32(rise)
    for b in range(n_blocks):
        M = curr_min[:, b]
        # Ensure operation result is cast or compatible
        # curr_noise_vec * rise_f should be float32 if both are float32
        curr_noise_vec = np.minimum(M, curr_noise_vec * rise_f)
        output[:, b] = curr_noise_vec
    return output

@jit(nopython=True)
def _numba_dn_gain_smooth(target_grid, n_cols, att, rel):
    # target_grid: (F, T)
    output = np.empty_like(target_grid)
    F = target_grid.shape[0]
    s_val = np.ones(F, dtype=np.float32)
    
    att_f = np.float32(att)
    rel_f = np.float32(rel)
    
    for i in range(n_cols):
        t_col = target_grid[:, i]
        # Explicit inner loop for Numba efficiency
        for f in range(F):
            t = t_col[f]
            s = s_val[f]
            if t < s:
                s_val[f] = att_f * s + (1.0 - att_f) * t
            else:
                s_val[f] = rel_f * s + (1.0 - rel_f) * t
        output[:, i] = s_val
    return output

@jit(nopython=True)
def _numba_deq_persist(pos, n_cols, a_p):
    # pos: (F, T)
    output = np.zeros_like(pos)
    F = pos.shape[0]
    state_p = np.zeros(F, dtype=np.float32)
    a_p_f = np.float32(a_p)
    
    for i in range(n_cols):
        col = pos[:, i]
        for f in range(F):
            state_p[f] = a_p_f * state_p[f] + (1.0 - a_p_f) * col[f]
            output[f, i] = state_p[f]
    return output

def _hpss_harmonic_mask(log_mag: np.ndarray, time_frames: int, freq_bins: int, eps: float = 1e-12) -> np.ndarray:
    """Median-filter HPSS harmonic mask in [0, 1] with shape matching log_mag."""
    tf = int(max(1, time_frames))
    ff = int(max(1, freq_bins))
    H = median_filter(log_mag, size=(1, tf), mode="nearest")
    P = median_filter(log_mag, size=(ff, 1), mode="nearest")
    Hlin = np.exp(H)
    Plin = np.exp(P)
    return (Hlin * Hlin) / (Hlin * Hlin + Plin * Plin + eps)


def _hpss_depth_scale(
    Mh: np.ndarray,
    *,
    harmonic_only: bool,
    protect_percussive: float,
) -> np.ndarray:
    """Scale repair depth maps; returns multiplier in [0, 1]."""
    scale = np.ones_like(Mh, dtype=np.float32)
    if harmonic_only:
        scale *= Mh.astype(np.float32, copy=False)
    prot = float(np.clip(protect_percussive, 0.0, 1.0))
    if prot > 0.0:
        Mp = (1.0 - Mh).astype(np.float32, copy=False)
        scale *= (1.0 - prot * Mp)
    return np.clip(scale, 0.0, 1.0).astype(np.float32, copy=False)


def _hpss_mask_for_band(
    Mag_mono: np.ndarray,
    band_idx: np.ndarray,
    *,
    time_frames: int,
    freq_bins: int,
    eps: float,
) -> np.ndarray:
    if band_idx.size == 0:
        return np.ones((0, Mag_mono.shape[1]), dtype=np.float32)
    log_mag = np.log(Mag_mono[band_idx, :] + eps)
    return _hpss_harmonic_mask(log_mag, time_frames, freq_bins, eps=eps)


def _apply_magnitude_inpaint(
    Z_band: np.ndarray,
    *,
    target_mag: np.ndarray,
    repair_depth: np.ndarray,
    confidence: np.ndarray,
    eps: float,
    ch_scales: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blend toward target magnitude keeping phase. Returns (Z_new, g_eff mono).
    repair_depth/confidence: (F, T); ch_scales optional (n_ch,) per-channel depth scale.
    """
    orig = Z_band
    orig_mag = np.abs(orig).astype(np.float32, copy=False)
    orig_phase = orig / (orig_mag + eps)
    tm = np.minimum(orig_mag[..., 0] if orig_mag.ndim == 3 else orig_mag, target_mag).astype(np.float32)
    if orig_mag.ndim == 3:
        tm = tm[:, :, None]
    rd = (repair_depth * confidence).astype(np.float32)
    if rd.ndim == 2 and orig_mag.ndim == 3:
        rd = rd[:, :, None]
    if ch_scales is not None and orig_mag.ndim == 3:
        rd = rd * ch_scales.astype(np.float32)[None, None, :]
    new_mag = (1.0 - rd) * orig_mag + rd * tm
    g_eff = (new_mag[..., 0] / (orig_mag[..., 0] + eps)).astype(np.float32)
    return (new_mag * orig_phase).astype(np.complex64, copy=False), g_eff


def _phasor_smooth_complex(
    Z: np.ndarray,
    weights: np.ndarray,
    size: int,
    axis: int,
    eps: float,
) -> np.ndarray:
    """Weighted circular-mean smoothing of unit phasors along one axis."""
    mag = np.abs(Z).astype(np.float32)
    w = (weights.astype(np.float32) * mag).astype(np.float32)
    unit = Z / (mag + eps)
    k = max(3, int(size) | 1)

    def _filt(a: np.ndarray) -> np.ndarray:
        return uniform_filter1d(a.astype(np.float64), size=k, axis=axis, mode="nearest").astype(np.float32)

    wr = _filt(w * np.real(unit))
    wi = _filt(w * np.imag(unit))
    ww = _filt(w) + eps
    u = (wr + 1j * wi) / ww
    u /= (np.abs(u) + eps)
    return u.astype(np.complex64, copy=False)


def _phase_instability_map(Z_mono: np.ndarray, eps: float) -> np.ndarray:
    """
    (F, T) map in [0, 1]: high where inter-frame / inter-bin phase jumps are erratic.
    Used to target moving 'swish' without dulling stable harmonics.
    """
    unit = Z_mono / (np.abs(Z_mono) + eps)
    n_f, n_t = unit.shape
    instab_t = np.zeros((n_f, n_t), dtype=np.float32)
    if n_t >= 2:
        dt = unit[:, 1:] * np.conj(unit[:, :-1])
        dphi_t = np.abs(np.angle(dt)).astype(np.float32)
        instab_t[:, 0] = dphi_t[:, 0]
        if n_t >= 3:
            instab_t[:, 1:-1] = 0.5 * (dphi_t[:, :-1] + dphi_t[:, 1:])
        instab_t[:, -1] = dphi_t[:, -1]

    instab_f = np.zeros((n_f, n_t), dtype=np.float32)
    if n_f >= 2:
        df = unit[1:, :] * np.conj(unit[:-1, :])
        dphi_f = np.abs(np.angle(df)).astype(np.float32)
        instab_f[0, :] = dphi_f[0, :]
        if n_f >= 3:
            instab_f[1:-1, :] = 0.5 * (dphi_f[:-1, :] + dphi_f[1:, :])
        instab_f[-1, :] = dphi_f[-1, :]

    raw = instab_t + 0.65 * instab_f
    p25 = float(np.percentile(raw, 25.0))
    p75 = float(np.percentile(raw, 75.0))
    span = max(p75 - p25, 0.04)
    return np.clip((raw - p25) / span, 0.0, 1.0).astype(np.float32, copy=False)


def measure_phase_instability(
    x: np.ndarray,
    sr: int,
    lo_hz: float,
    hi_hz: float,
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """
    Magnitude-weighted mean phase instability in [0, 1] for a frequency band.
    Higher values indicate more moving 'swish' / erratic phase texture.
    """
    xm = np.asarray(x, dtype=np.float32)
    if xm.ndim == 2:
        xm = np.mean(xm, axis=1).astype(np.float32, copy=False)
    elif xm.ndim != 1:
        raise ValueError("audio must be 1D or 2D")
    if xm.size < n_fft:
        return 0.0

    eps = 1e-12
    _, _, Z = stft(
        xm[None, :],
        fs=int(sr),
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float32)
    lo = float(max(0.0, lo_hz))
    hi = float(min(float(freqs[-1]), hi_hz))
    if hi <= lo:
        return 0.0
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if idx.size < 4:
        return 0.0

    zm = Z[0, idx, :].astype(np.complex64, copy=False)
    instab = _phase_instability_map(zm, eps=eps)
    mag = np.abs(zm).astype(np.float32, copy=False)
    return float(np.sum(instab * mag) / (np.sum(mag) + eps))


def _apply_swish_phase_repair(
    Z: np.ndarray,
    band_idx: np.ndarray,
    *,
    strength: float,
    time_amt: float,
    freq_amt: float,
    time_win: int,
    freq_win: int,
    w_nontrans: np.ndarray,
    transient_protect: float,
    harmonic_protect: float,
    Mag_mono: np.ndarray,
    hpss_time_frames: int,
    hpss_freq_bins: int,
    eps: float,
) -> None:
    """In-place adaptive phase smoothing for moving swish texture."""
    if band_idx.size == 0 or strength <= 1e-6:
        return

    zm = np.mean(Z[band_idx, :, :], axis=2)
    instab = _phase_instability_map(zm, eps=eps)

    log_mag = np.log(Mag_mono[band_idx, :] + eps)
    t_win = max(3, int(hpss_time_frames) | 1)
    f_win = max(3, int(hpss_freq_bins) | 1)
    h_lin = np.exp(median_filter(log_mag, size=(1, t_win), mode="nearest"))
    p_lin = np.exp(median_filter(log_mag, size=(f_win, 1), mode="nearest"))
    mh = (h_lin ** 2 / (h_lin ** 2 + p_lin ** 2 + eps)).astype(np.float32, copy=False)

    w_trans = 1.0 - w_nontrans.astype(np.float32, copy=False)
    w_prot = (1.0 - float(np.clip(transient_protect, 0.0, 1.0)) * w_trans).astype(np.float32, copy=False)
    depth = (
        float(np.clip(strength, 0.0, 1.0))
        * instab
        * w_prot[None, :]
        * (1.0 - float(np.clip(harmonic_protect, 0.0, 1.0)) * mh)
    ).astype(np.float32, copy=False)

    weights = Mag_mono[band_idx, :].astype(np.float32, copy=False)
    t_amt = float(np.clip(time_amt, 0.0, 1.0))
    f_amt = float(np.clip(freq_amt, 0.0, 1.0))

    for ch in range(Z.shape[2]):
        zc = Z[band_idx, :, ch]
        mag = np.abs(zc).astype(np.float32, copy=False)
        unit = zc / (mag + eps)

        if t_amt > 1e-6:
            unit_t = _phasor_smooth_complex(zc, weights, time_win, axis=1, eps=eps)
            blend = (t_amt * depth).astype(np.float32, copy=False)
            unit = (1.0 - blend) * unit + blend * unit_t

        if f_amt > 1e-6:
            unit_f = _phasor_smooth_complex(unit * mag, weights, freq_win, axis=0, eps=eps)
            blend = (f_amt * depth).astype(np.float32, copy=False)
            unit = (1.0 - blend) * unit + blend * unit_f

        unit /= (np.abs(unit) + eps)
        Z[band_idx, :, ch] = (mag * unit).astype(np.complex64, copy=False)


def _apply_hf_decorrelation(
    Z: np.ndarray,
    band_idx: np.ndarray,
    *,
    amount: float,
    rng: np.random.Generator,
    eps: float,
) -> None:
    """Break synthetic L/R phase lock in HF by partial Side-channel phase scatter."""
    if band_idx.size == 0 or amount <= 1e-6 or Z.shape[2] < 2:
        return

    left = Z[band_idx, :, 0]
    right = Z[band_idx, :, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    coh = np.real(left * np.conj(right)) / (np.abs(left) * np.abs(right) + eps)
    coh = np.clip(coh, -1.0, 1.0).astype(np.float32, copy=False)
    coh_excess = np.clip((coh - 0.25) / 0.65, 0.0, 1.0).astype(np.float32, copy=False)

    side_mag = np.abs(side)
    side_phi = np.angle(side)
    mid_phi = np.angle(mid + eps)
    phi_rand = rng.uniform(0.0, 2.0 * np.pi, size=side_phi.shape).astype(np.float32)
    phi_target = 0.55 * phi_rand + 0.45 * mid_phi
    side_new = side_mag * np.exp(1j * phi_target).astype(np.complex64)

    depth = (float(np.clip(amount, 0.0, 1.0)) * coh_excess).astype(np.float32, copy=False)
    side_blend = (1.0 - depth) * side + depth * side_new

    Z[band_idx, :, 0] = (mid + side_blend).astype(np.complex64, copy=False)
    Z[band_idx, :, 1] = (mid - side_blend).astype(np.complex64, copy=False)


def _artifact_confidence_map(
    Mag_mono: np.ndarray,
    band_idx: np.ndarray,
    *,
    freq_med_bins: int,
    thr_db: float,
    eps: float,
) -> np.ndarray:
    """High where narrow log-magnitude residual exceeds threshold."""
    if band_idx.size == 0:
        return np.zeros((0, Mag_mono.shape[1]), dtype=np.float32)
    mag = Mag_mono[band_idx, :]
    L = np.log(mag + eps)
    L_med = median_filter(L, size=(int(max(3, freq_med_bins)), 1), mode="nearest")
    resid = (L - L_med) * (20.0 / np.log(10.0))
    conf = np.clip((resid - float(thr_db)) / max(1e-6, float(thr_db)), 0.0, 1.0)
    return conf.astype(np.float32, copy=False)

def process_stft(x: np.ndarray, sr: int, p: Params, dbg: Optional[DebugCollector] = None) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    
    # Transpose to (n_ch, n_samples) for scipy.signal
    x_t = x.T
    n_ch, n_samples = x_t.shape
    
    n_fft = int(p.n_fft)
    hop = int(p.hop)
    
    # Get STFT (scipy style: input (ch, time) -> output (ch, freq, time) with axis=-1)
    f, t_sec, Z = stft(x_t, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', padded=True, axis=-1)
    
    # Transpose to (n_freq, n_time, n_ch) for convenient processing
    Z = Z.transpose(1, 2, 0).astype(np.complex64)
    
    freqs = f.astype(np.float32)
    nyq = f[-1] if len(f) > 0 else 0.0
    n_cols = Z.shape[1]
    eps = 1e-12
    rng = np.random.default_rng(int(getattr(p, "seed", 0)))

    # --- Indices helper ---
    def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
        lo = float(max(0.0, lo_hz))
        hi = float(min(nyq, hi_hz))
        if hi <= lo:
            return np.array([], dtype=np.int64)
        return np.where((freqs >= lo) & (freqs <= hi))[0]

    # --- Pre-calc indices ---
    sh_idx = _idx(p.start_hz, p.end_hz)
    dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
    deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)
    exp_idx = _idx(getattr(p, "exp_start_hz", 3000.0), getattr(p, "exp_end_hz", 8000.0))
    hpss_idx = _idx(getattr(p, "hpss_start_hz", 3000.0), getattr(p, "hpss_end_hz", 8000.0))
    pb_idx = _idx(getattr(p, "pb_start_hz", 3000.0), getattr(p, "pb_end_hz", 8000.0))
    sw_idx = _idx(getattr(p, "swish_start_hz", 3500.0), getattr(p, "swish_end_hz", 14000.0))
    hdc_idx = _idx(getattr(p, "hf_dec_start_hz", 4500.0), getattr(p, "hf_dec_end_hz", 16000.0))


    # --- Tapers ---
    w_sh = _edge_taper(freqs, sh_idx, p.start_hz, p.end_hz, p.edge_hz)[:, None]
    w_dn = _edge_taper(freqs, dn_idx, p.dn_start_hz, p.dn_end_hz, p.dn_edge_hz)[:, None]
    w_deq = _edge_taper(freqs, deq_idx, p.deq_start_hz, p.deq_end_hz, p.deq_edge_hz)[:, None]

    # --- Base Magnitude / PSD ---
    Mag = np.abs(Z) # (F, T, Ch)
    Mag_mono = np.mean(Mag, axis=2) # (F, T)
    PSD = Mag_mono ** 2 + eps

    # --- Global Features ---
    def calc_flatness(P_region):
        log_P = np.log(P_region + eps)
        return np.exp(np.mean(log_P, axis=0)) / (np.mean(P_region, axis=0) + eps)
        
    flat_full = calc_flatness(PSD)
    band_db_full = 10.0 * np.log10(np.mean(PSD, axis=0) + eps)
    
    w_noise_full = np.clip((flat_full - p.flat_start) / max(1e-6, p.flat_end - p.flat_start), 0.0, 1.0)
    
    flux = np.zeros_like(band_db_full)
    flux[1:] = np.maximum(0.0, band_db_full[1:] - band_db_full[:-1])
    w_trans = np.clip((flux - p.flux_thr_db) / max(1e-6, p.flux_range_db), 0.0, 1.0)
    w_nontrans = 1.0 - w_trans

    # Mid/Side channel depth scales (stereo only)
    ms_enabled = bool(getattr(p, "ms_process", False)) and n_ch >= 2
    ch_scales = np.ones(n_ch, dtype=np.float32)
    if ms_enabled:
        ch_scales[0] = 1.0
        ch_scales[1] = float(np.clip(getattr(p, "ms_side_scale", 0.35), 0.0, 1.0))
        if n_ch > 2:
            ch_scales[2:] = float(np.clip(getattr(p, "ms_side_scale", 0.35), 0.0, 1.0))

    Z_orig = Z.copy()
    g_total = np.ones((freqs.size, n_cols), dtype=np.float32)
    cap_db = 48.0 if bool(getattr(p, "nuclear_mode", False)) else float(getattr(p, "total_att_cap_db", 12.0))
    g_min = float(10.0 ** (-max(0.0, cap_db) / 20.0))

    hpss_on = bool(getattr(p, "hpss", False))
    hpss_tf = int(getattr(p, "hpss_time_frames", 21))
    hpss_ff = int(getattr(p, "hpss_freq_bins", 17))
    hpss_harm = bool(getattr(p, "hpss_harmonic_only", True))
    hpss_prot = float(getattr(p, "hpss_protect_percussive", 0.0))
    use_inpaint = bool(getattr(p, "magnitude_inpaint", True))
    use_deq_inpaint = bool(getattr(p, "deq_inpaint", True)) and use_inpaint

    def _depth_scale_for(band_idx: np.ndarray) -> np.ndarray:
        if not hpss_on or band_idx.size == 0:
            return np.ones((band_idx.size, n_cols), dtype=np.float32)
        Mh = _hpss_mask_for_band(Mag_mono, band_idx, time_frames=hpss_tf, freq_bins=hpss_ff, eps=eps)
        return _hpss_depth_scale(Mh, harmonic_only=hpss_harm, protect_percussive=hpss_prot)

    artifact_conf_parts: list[np.ndarray] = []

    # --- (0) Expander ---
    if bool(getattr(p, "expander", False)) and exp_idx.size:
        band_p_exp = np.mean(PSD[exp_idx, :], axis=0)
        band_db_exp = 10.0 * np.log10(band_p_exp + eps)
        thr = float(getattr(p, "exp_threshold_db", -45.0))
        ratio = float(max(1.0, getattr(p, "exp_ratio", 2.0)))
        
        red_db = np.zeros_like(band_db_exp)
        mask = band_db_exp < thr
        red_db[mask] = (thr - band_db_exp[mask]) * (ratio - 1.0)
        g_target = 10.0 ** (-red_db / 20.0)
        
        att = _frame_coeff(hop, sr, getattr(p, "exp_attack_ms", 10.0))
        rel = _frame_coeff(hop, sr, getattr(p, "exp_release_ms", 150.0))
        if HAS_NUMBA:
            g_target = g_target.astype(np.float32)
            g_env = _numba_exp_env(g_target, n_cols, att, rel)
        else:
            g_env = np.ones_like(g_target)
            g_sm = 1.0
            for i in range(n_cols):
                curr = g_target[i]
                if curr < g_sm: g_sm = att * g_sm + (1.0 - att) * curr
                else: g_sm = rel * g_sm + (1.0 - rel) * curr
                g_env[i] = g_sm
            
        g_total[exp_idx, :] *= g_env.astype(np.float32, copy=False)

    # --- Debug: attenuation maps (filled when stages run) ---
    g_eff_dn_map: Optional[np.ndarray] = None
    g_eff_deq_map: Optional[np.ndarray] = None
    g_eff_sh_map: Optional[np.ndarray] = None
    noise_psd_dn_db_map: Optional[np.ndarray] = None
    dn_depth_map: Optional[np.ndarray] = None
    deq_depth_map: Optional[np.ndarray] = None
    sh_depth_map: Optional[np.ndarray] = None

    # --- (A) Smart Denoise ---
    dn_str = float(np.clip(p.denoise, 0.0, 1.0))
    if dn_str > 1e-6 and dn_idx.size:
        psd_dn = PSD[dn_idx, :]
        a_psd = _frame_coeff(hop, sr, p.dn_psd_smooth_ms)
        psd_sm_map = lfilter([1.0-a_psd], [1.0, -a_psd], psd_dn, axis=1) # (F_dn, T)
        
        minwin = max(4, int(sr * p.dn_minwin_ms / 1000.0 / hop))
        n_blocks = int(np.ceil(n_cols / minwin))
        pad_t = n_blocks * minwin - n_cols
        p_padded = np.pad(psd_sm_map, ((0,0), (0, pad_t)), constant_values=np.inf)
        p_reshaped = p_padded.reshape(psd_dn.shape[0], n_blocks, minwin)
        curr_min = np.min(p_reshaped, axis=2)
        
        rise = 10.0 ** ((p.dn_up_db_per_s * (minwin*hop/sr))/10.0)
        curr_noise_vec = psd_dn[:, 0]
        if HAS_NUMBA:
            curr_min = curr_min.astype(np.float32)
            curr_noise_vec = curr_noise_vec.astype(np.float32)
            noise_est_blocks = _numba_dn_noise_est(curr_min, n_blocks, curr_noise_vec, rise)
        else:
            noise_est_blocks = np.zeros_like(curr_min)
            for b in range(n_blocks):
                M = curr_min[:, b]
                curr_noise_vec = np.minimum(M, curr_noise_vec * rise)
                noise_est_blocks[:, b] = curr_noise_vec
        
        noise_map = np.repeat(noise_est_blocks, minwin, axis=1)[:, :n_cols]
        SNR = psd_sm_map / (noise_map + eps)
        k = 1.0 + 3.0 * dn_str
        g_inst = SNR / (SNR + k)
        floor = 10.0**(p.dn_floor_db/20.0)
        g_inst = np.clip(floor + (1.0 - floor) * g_inst, floor, 1.0)
        
        att = _frame_coeff(hop, sr, p.dn_attack_ms)
        rel = _frame_coeff(hop, sr, p.dn_release_ms)
        if HAS_NUMBA:
             g_inst = g_inst.astype(np.float32)
             dn_g = _numba_dn_gain_smooth(g_inst, n_cols, att, rel)
        else:
             dn_g = np.ones_like(g_inst)
             s_val = np.ones(psd_dn.shape[0])
             for i in range(n_cols):
                  target = g_inst[:, i]
                  down = target < s_val
                  s_val[down] = att * s_val[down] + (1.0 - att) * target[down]
                  s_val[~down] = rel * s_val[~down] + (1.0 - rel) * target[~down]
                  dn_g[:, i] = s_val
             
        if p.dn_freq_smooth_bins > 1:
            dn_g = uniform_filter1d(dn_g, size=int(p.dn_freq_smooth_bins), axis=0, mode='nearest')
            
        hpss_dn = _depth_scale_for(dn_idx)
        depth_base = dn_str * (0.5 + 0.5 * w_noise_full) * w_nontrans
        depth_map = depth_base[None, :] * w_dn * hpss_dn
        g_eff = 1.0 - depth_map * (1.0 - dn_g)
        g_total[dn_idx, :] *= g_eff.astype(np.float32, copy=False)
        g_eff_dn_map = g_eff.astype(np.float32, copy=False)
        dn_depth_map = depth_base.astype(np.float32, copy=False)
        noise_psd_dn_db_map = (10.0 * np.log10(noise_map + eps)).astype(np.float32, copy=False)

    # --- (B) De-resonator ---
    deres_str = float(np.clip(p.deres, 0.0, 1.0))
    if deres_str > 1e-6 and deq_idx.size:
        L = np.log(Mag_mono[deq_idx, :] + eps)
        L_med = median_filter(L, size=(int(p.deq_freq_med_bins), 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))
        
        thr_eff = p.deq_thr_db + p.deq_tonal_boost_db * (1.0 - w_noise_full)
        pos = np.maximum(0.0, resid - thr_eff[None, :])

        if bool(getattr(p, "deq_time_floor", False)):
            psd_deq = PSD[deq_idx, :].astype(np.float32, copy=False)
            a_floor = _frame_coeff(hop, sr, float(getattr(p, "deq_floor_smooth_ms", 80.0)))
            psd_floor_sm = lfilter([1.0 - a_floor], [1.0, -a_floor], psd_deq, axis=1)

            rise_lin = float(
                10.0 ** ((float(getattr(p, "deq_floor_rise_db_per_s", 1.0)) * (float(hop) / float(sr))) / 10.0)
            )
            floor_env = np.empty_like(psd_floor_sm)
            floor_env[:, 0] = psd_floor_sm[:, 0]
            for ti in range(1, n_cols):
                prev = floor_env[:, ti - 1]
                floor_env[:, ti] = np.minimum(prev * rise_lin, psd_floor_sm[:, ti])

            floor_win = max(4, int(round(sr * 1.0 / hop)))
            floor_map = minimum_filter1d(floor_env, size=floor_win, axis=1, mode="nearest")
            floor_db = 10.0 * np.log10(floor_map + eps)
            local_floor_db = median_filter(
                floor_db,
                size=(int(p.deq_freq_med_bins), 1),
                mode="nearest",
            )
            floor_thr = float(getattr(p, "deq_floor_thr_db", 3.0))
            floor_excess = np.maximum(0.0, floor_db - local_floor_db - floor_thr)
            pos = np.maximum(pos, floor_excess.astype(np.float32, copy=False))
        
        mask = pos > 0.0
        dens = np.mean(mask, axis=0)
        if bool(getattr(p, "deq_time_floor", False)):
            w_narrow = np.ones(n_cols, dtype=np.float32)
        else:
            w_narrow = 1.0 - np.clip((dens - p.deq_density_lo)/(max(1e-6, p.deq_density_hi - p.deq_density_lo)), 0.0, 1.0)
        
        a_p = _frame_coeff(hop, sr, p.deq_persist_ms)
        if HAS_NUMBA:
            pos = pos.astype(np.float32)
            persist_map = _numba_deq_persist(pos, n_cols, a_p)
        else:
            persist_map = np.zeros_like(pos)
            state_p = np.zeros(pos.shape[0])
            for i in range(n_cols):
                state_p = a_p * state_p + (1.0 - a_p) * pos[:, i]
                persist_map[:, i] = state_p
            
        gate = np.clip(persist_map / max(1e-6, p.deq_persist_thr_db), 0.0, 1.0)
        att_db = np.minimum(p.deq_slope * pos * gate, p.deq_max_att_db)
        gain = 10.0 ** (-att_db / 20.0)
        
        if p.deq_freq_smooth_bins > 1:
            gain = uniform_filter1d(gain, size=int(p.deq_freq_smooth_bins), axis=0, mode='nearest')
            
        hpss_deq = _depth_scale_for(deq_idx)
        depth_base = deres_str * w_nontrans * w_narrow
        depth_map = depth_base[None, :] * w_deq * hpss_deq
        conf_deq = np.clip(gate, 0.0, 1.0).astype(np.float32, copy=False)
        artifact_conf_parts.append(conf_deq)

        if use_deq_inpaint:
            thr_lin = float(p.deq_thr_db) * (np.log(10.0) / 20.0)
            target_mag = np.exp(L_med + thr_lin).astype(np.float32)
            repair_depth = depth_map * conf_deq
            Z_deq, g_eff = _apply_magnitude_inpaint(
                Z_orig[deq_idx, :, :],
                target_mag=target_mag,
                repair_depth=repair_depth,
                confidence=np.ones_like(conf_deq),
                eps=eps,
                ch_scales=ch_scales if ms_enabled else None,
            )
            g_total[deq_idx, :] *= g_eff
        else:
            g_eff = 1.0 - depth_map * (1.0 - gain)
            g_total[deq_idx, :] *= g_eff.astype(np.float32, copy=False)
        g_eff_deq_map = g_eff.astype(np.float32, copy=False)
        deq_depth_map = depth_base.astype(np.float32, copy=False)

    # --- (C) Shimmer ---
    if sh_idx.size >= 8:
        mag_sh = Mag_mono[sh_idx, :]
        flat_sh = calc_flatness(mag_sh**2)
        w_noise_sh = np.clip((flat_sh - p.flat_start)/max(1e-6, p.flat_end - p.flat_start), 0.0, 1.0)
        
        L = np.log(mag_sh + eps)
        L_med = median_filter(L, size=(int(p.freq_med_bins), 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))
        
        mask = resid > p.thr_db
        dens = np.mean(mask, axis=0)
        w_narrow = 1.0 - np.clip((dens - p.density_lo)/max(1e-6, p.density_hi - p.density_lo), 0.0, 1.0)

        att_db = np.zeros_like(resid)
        m2 = resid > p.thr_db
        att_db[m2] = p.slope * (resid[m2] - p.thr_db)
        gain = 10.0 ** (-att_db / 20.0)

        hpss_sh = _depth_scale_for(sh_idx)
        depth_base = w_noise_sh * w_nontrans * w_narrow
        depth_map = depth_base[None, :] * w_sh * hpss_sh
        conf_sh = np.clip((resid - float(p.thr_db)) / max(1e-6, float(p.thr_db)), 0.0, 1.0).astype(np.float32)
        artifact_conf_parts.append(conf_sh)

        if use_inpaint:
            thr_lin = float(p.thr_db) * (np.log(10.0) / 20.0)
            target_mag = np.exp(L_med + thr_lin).astype(np.float32)
            repair_depth = depth_map * conf_sh
            _Z_sh, g_eff = _apply_magnitude_inpaint(
                Z_orig[sh_idx, :, :],
                target_mag=target_mag,
                repair_depth=repair_depth,
                confidence=np.ones_like(conf_sh),
                eps=eps,
                ch_scales=ch_scales if ms_enabled else None,
            )
            g_total[sh_idx, :] *= g_eff
        else:
            g_eff = 1.0 - depth_map * (1.0 - gain)
            g_total[sh_idx, :] *= g_eff.astype(np.float32, copy=False)
        g_eff_sh_map = g_eff.astype(np.float32, copy=False)
        sh_depth_map = depth_base.astype(np.float32, copy=False)

    # --- Apply combined gain with cap ---
    proc_idx = np.unique(np.concatenate([
        arr for arr in [exp_idx, dn_idx, deq_idx, sh_idx] if arr.size > 0
    ])).astype(np.int64) if (exp_idx.size or dn_idx.size or deq_idx.size or sh_idx.size) else np.array([], dtype=np.int64)
    if proc_idx.size:
        g_total[proc_idx, :] = np.maximum(g_total[proc_idx, :], g_min)
        Z = Z_orig.copy()
        g_apply = g_total[proc_idx, :].astype(np.float32, copy=False)
        if ms_enabled and n_ch >= 2:
            for ci in range(n_ch):
                g_ch = 1.0 - (1.0 - g_apply) * float(ch_scales[ci])
                Z[proc_idx, :, ci] = Z_orig[proc_idx, :, ci] * g_ch
        else:
            Z[proc_idx, :, :] = Z_orig[proc_idx, :, :] * g_apply[:, :, None]
    else:
        Z = Z_orig.copy()

    # Per-frame artifact confidence (max across freq bins and stages) for HF resynth.
    # Band maps differ in F (deq_idx vs sh_idx), so merge on time only.
    global _LAST_ARTIFACT_CONF_FRAMES
    if artifact_conf_parts:
        conf_per_frame = np.zeros(n_cols, dtype=np.float32)
        for part in artifact_conf_parts:
            conf_per_frame = np.maximum(conf_per_frame, np.max(part, axis=0))
        _LAST_ARTIFACT_CONF_FRAMES = conf_per_frame.reshape(1, -1)
    else:
        _LAST_ARTIFACT_CONF_FRAMES = None

    # --- Shimmer noise resynth (post-gain) ---
    if sh_idx.size >= 8 and float(p.noise_resynth) > 0.0:
        nr = float(p.noise_resynth)
        flat_sh = calc_flatness(Mag_mono[sh_idx, :] ** 2)
        w_noise_sh = np.clip((flat_sh - p.flat_start)/max(1e-6, p.flat_end - p.flat_start), 0.0, 1.0)
        L = np.log(Mag_mono[sh_idx, :] + eps)
        L_med = median_filter(L, size=(int(p.freq_med_bins), 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))
        mask = resid > p.thr_db
        dens = np.mean(mask, axis=0)
        w_narrow = 1.0 - np.clip((dens - p.density_lo)/max(1e-6, p.density_hi - p.density_lo), 0.0, 1.0)
        hpss_sh = _depth_scale_for(sh_idx)
        depth_map = (nr * w_noise_sh * w_nontrans * w_narrow)[None, :] * w_sh * hpss_sh
        phases = rng.uniform(0.0, 2.0 * np.pi, size=Z[sh_idx].shape[:2]).astype(np.float32)
        zph = np.cos(phases) + 1j * np.sin(phases)
        zph = zph[:, :, None]
        d = depth_map[:, :, None]
        if ms_enabled:
            d = d * ch_scales.astype(np.float32)[None, None, :]
        Z[sh_idx] = (1.0 - d) * Z[sh_idx] + d * (np.abs(Z[sh_idx]) * zph)

    # --- (D) HPSS & Phase Blur ---
    pb_amt = float(p.phase_blur)
    if pb_amt > 1e-6 and pb_idx.size:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=Z[pb_idx].shape[:2]).astype(np.float32)
        zph = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)
        zph = zph[:, :, None]
        mix_v = pb_amt
        if bool(getattr(p, "pb_harmonic_only", True)):
             mag_pb = Mag_mono[pb_idx, :]
             L = np.log(mag_pb + eps)
             H = median_filter(L, size=(1, int(p.hpss_time_frames)), mode='nearest')
             P = median_filter(L, size=(int(p.hpss_freq_bins), 1), mode='nearest')
             Hlin = np.exp(H); Plin = np.exp(P)
             Mh = Hlin**2 / (Hlin**2 + Plin**2 + eps)
             mix_v = pb_amt * Mh
        
        m = mix_v if np.isscalar(mix_v) else mix_v[:, :, None]
        Z[pb_idx] = (1.0 - m) * Z[pb_idx] + m * (np.abs(Z[pb_idx]) * zph)

    # --- (E) Swish repair: adaptive phase coherence smoothing ---
    swish_amt = float(getattr(p, "swish_repair", 0.0))
    if swish_amt > 1e-6 and sw_idx.size:
        _apply_swish_phase_repair(
            Z,
            sw_idx,
            strength=swish_amt,
            time_amt=float(getattr(p, "swish_time_amt", 0.55)),
            freq_amt=float(getattr(p, "swish_freq_amt", 0.30)),
            time_win=int(getattr(p, "swish_time_win", 7)),
            freq_win=int(getattr(p, "swish_freq_win", 5)),
            w_nontrans=w_nontrans,
            transient_protect=float(getattr(p, "swish_transient_protect", 0.85)),
            harmonic_protect=float(getattr(p, "swish_harmonic_protect", 0.45)),
            Mag_mono=Mag_mono,
            hpss_time_frames=hpss_tf,
            hpss_freq_bins=hpss_ff,
            eps=eps,
        )

    # --- (F) HF stereo decorrelation ---
    hdc_amt = float(getattr(p, "hf_decorrelate", 0.0))
    if hdc_amt > 1e-6 and hdc_idx.size and n_ch >= 2:
        _apply_hf_decorrelation(Z, hdc_idx, amount=hdc_amt, rng=rng, eps=eps)

    # --- Debug collection (strided frame samples) ---
    if dbg is not None:
        flux_full = np.zeros(n_cols, dtype=np.float32)
        flux_full[1:] = flux.astype(np.float32, copy=False)

        def _att_db_from_gain(g_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if g_map is None:
                return None
            return (-20.0 * np.log10(np.clip(g_map, 1e-6, 1.0))).astype(np.float32, copy=False)

        att_dn_db_map = _att_db_from_gain(g_eff_dn_map)
        att_deq_db_map = _att_db_from_gain(g_eff_deq_map)
        att_sh_db_map = _att_db_from_gain(g_eff_sh_map)

        for fi in range(n_cols):
            if dbg.want():
                dbg.push_meta(
                    s=fi * hop,
                    n_fft=n_fft,
                    w_noise=float(w_noise_full[fi]),
                    w_trans=float(w_trans[fi]),
                    flux_db=float(flux_full[fi]),
                    band_db=float(band_db_full[fi]),
                    dn_depth=float(dn_depth_map[fi]) if dn_depth_map is not None else 0.0,
                    deq_depth=float(deq_depth_map[fi]) if deq_depth_map is not None else 0.0,
                    sh_depth=float(sh_depth_map[fi]) if sh_depth_map is not None else 0.0,
                )
                dbg.push_att(
                    att_dn_db_map[:, fi] if att_dn_db_map is not None else None,
                    att_deq_db_map[:, fi] if att_deq_db_map is not None else None,
                    att_sh_db_map[:, fi] if att_sh_db_map is not None else None,
                )
                dbg.push_noise_psd_dn(
                    noise_psd_dn_db_map[:, fi] if noise_psd_dn_db_map is not None else None
                )
            dbg.step()

    # --- Reconstruct ---
    Z_out = Z.transpose(2, 0, 1)
    # istft will remove padding and merge overlaps
    _, y_rec_t = istft(Z_out, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', time_axis=-1, freq_axis=-2)
    
    y_rec = y_rec_t.T
    if y_rec.shape[0] > x.shape[0]:
        y_rec = y_rec[:x.shape[0], :]
    elif y_rec.shape[0] < x.shape[0]:
        y_rec = np.pad(y_rec, ((0, x.shape[0]-y_rec.shape[0]), (0,0)))
        
    mix_p = float(p.mix)
    y_final = mix_p * y_rec + (1.0 - mix_p) * x
    
    if p.delta_listen:
        return (x - y_final).squeeze()
    return y_final.squeeze()


def hf_resynth_post(x: np.ndarray, sr: int, p: Params, *, artifact_conf: Optional[np.ndarray] = None) -> np.ndarray:
    """Optional HF resynthesis ("nuclear"): remove HF and rebuild from low band."""
    if not bool(getattr(p, "hf_resynth", False)):
        return x
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    nyq = 0.5 * float(sr)

    lp_hz = float(np.clip(getattr(p, "hf_lp_hz", 3000.0), 20.0, nyq - 100.0))
    src_lo = float(np.clip(getattr(p, "hf_src_lo_hz", 1000.0), 20.0, nyq - 100.0))
    src_hi = float(np.clip(getattr(p, "hf_src_hi_hz", 2000.0), src_lo + 10.0, nyq - 100.0))
    hp_hz = float(np.clip(getattr(p, "hf_hp_hz", 3000.0), 20.0, nyq - 100.0))
    drive = float(max(0.1, getattr(p, "hf_drive", 2.0)))
    mix = float(np.clip(getattr(p, "hf_mix", 0.35), 0.0, 1.0))
    conf_blend = bool(getattr(p, "hf_confidence_blend", True))

    zero_phase = bool(getattr(p, "hf_zero_phase", True))
    tilt_lp_hz = float(np.clip(getattr(p, "hf_tilt_lp_hz", 12000.0), hp_hz + 50.0, nyq - 50.0))

    def lr4_sos(kind: str, hz: float):
        s = _butter_sos(sr, kind, float(hz), order=2)
        return np.concatenate([s, s], axis=0)

    def _sos_apply(sos, sig):
        y = sosfilt(sos, sig, axis=0)
        return y[0] if isinstance(y, tuple) else y

    sos_lp = lr4_sos("lowpass", lp_hz)
    sos_hp = lr4_sos("highpass", hp_hz)

    if zero_phase:
        base = sosfiltfilt(sos_lp, x2, axis=0).astype(np.float32, copy=False)
    else:
        base = _sos_apply(sos_lp, x2).astype(np.float32, copy=False)

    sos_bp = _butter_sos(sr, "bandpass", (src_lo, src_hi), order=2)
    if zero_phase:
        src = sosfiltfilt(sos_bp, x2, axis=0).astype(np.float32, copy=False)
    else:
        src = _sos_apply(sos_bp, x2).astype(np.float32, copy=False)
    gen = np.tanh(src * drive).astype(np.float32, copy=False)

    if zero_phase:
        gen_hf = sosfiltfilt(sos_hp, gen, axis=0).astype(np.float32, copy=False)
    else:
        gen_hf = _sos_apply(sos_hp, gen).astype(np.float32, copy=False)

    if tilt_lp_hz < nyq - 50.0:
        sos_tilt = _butter_sos(sr, "lowpass", tilt_lp_hz, order=1)
        if zero_phase:
            gen_hf = sosfiltfilt(sos_tilt, gen_hf, axis=0).astype(np.float32, copy=False)
        else:
            gen_hf = _sos_apply(sos_tilt, gen_hf).astype(np.float32, copy=False)

    y_full = base + mix * gen_hf

    if not conf_blend:
        return y_full.squeeze()

    conf_frames = artifact_conf
    if conf_frames is None:
        conf_frames = _LAST_ARTIFACT_CONF_FRAMES
    if conf_frames is None or conf_frames.size == 0:
        return y_full.squeeze()

    hop = int(p.hop)
    n_samp = x2.shape[0]
    conf_t = np.mean(conf_frames, axis=0).astype(np.float32)
    t_frames = np.arange(conf_t.size, dtype=np.float32) * (hop / float(sr))
    t_samp = np.arange(n_samp, dtype=np.float32) / float(sr)
    conf_s = np.interp(t_samp, t_frames, conf_t, left=float(conf_t[0] if conf_t.size else 0.0),
                       right=float(conf_t[-1] if conf_t.size else 0.0)).astype(np.float32)
    conf_s = np.clip(conf_s, 0.0, 1.0)[:, None]
    y_orig_hf = x2 - base
    y = base + conf_s * (mix * gen_hf) + (1.0 - conf_s) * y_orig_hf
    return y.squeeze()


# -----------------------------
# Debug rendering
# -----------------------------
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _compute_mag_spectrogram_db(x: np.ndarray, sr: int, n_fft: int, hop: int, max_frames: int, max_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple magnitude spectrogram in dB (mono).
    Returns (S_db [F,T], freqs [F], times [T])
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    n_fft = int(max(256, n_fft))
    hop = int(max(1, hop))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_mask = freqs <= float(min(0.5 * sr, max_hz))
    freqs = freqs[f_mask]

    # Adapt hop to keep frames manageable
    n = x.shape[0]
    est_frames = int(math.ceil(n / hop))
    if est_frames > max_frames:
        hop = int(math.ceil(n / max_frames))
        hop = max(1, hop)

    win = np.hanning(n_fft).astype(np.float32)
    eps = 1e-12

    frames = []
    times = []
    for s in range(0, n, hop):
        frame = x[s:s + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]), mode="constant")
        frame = frame * win
        spec = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(spec).astype(np.float32) + eps
        mag = mag[f_mask]
        frames.append(20.0 * np.log10(mag))
        times.append((s + 0.5 * n_fft) / sr)

    S_db = np.stack(frames, axis=1) if frames else np.zeros((freqs.size, 0), dtype=np.float32)
    t = np.array(times, dtype=np.float32)
    return S_db.astype(np.float32), freqs.astype(np.float32), t


def _plot_and_save_png(fig_path: str, fig) -> None:
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def render_debug(
    dbg_dir: str,
    dbg_data: Dict[str, Any],
    x_in: np.ndarray,
    y_repaired: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    dp: DebugParams,
    summary: Dict[str, Any],
) -> None:
    dbg_dir = _ensure_dir(dbg_dir)

    # Save internals
    if dp.save_npz:
        npz_path = os.path.join(dbg_dir, "debug_data.npz")
        np.savez_compressed(npz_path, **{k: v for k, v in dbg_data.items() if isinstance(v, np.ndarray)})

    # Save summary
    _write_json(os.path.join(dbg_dir, "summary.json"), summary)

    # Make plots if possible
    if not dp.save_png:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _write_json(os.path.join(dbg_dir, "plot_error.json"), {"error": str(e)})
        return

    # ---- Spectrogram before/after/diff ----
    S0, f0, t0 = _compute_mag_spectrogram_db(x_in, sr, dp.spec_n_fft, dp.spec_hop, dp.spec_max_frames, dp.spec_max_hz)
    S1, f1, t1 = _compute_mag_spectrogram_db(y_out, sr, dp.spec_n_fft, dp.spec_hop, dp.spec_max_frames, dp.spec_max_hz)

    # align time bins if slightly different due to hop adaptation
    T = min(S0.shape[1], S1.shape[1])
    S0 = S0[:, :T]
    S1 = S1[:, :T]
    t = t0[:T] if t0.size else t1[:T]
    f = f0 if f0.size else f1

    # input spectrogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(S0, origin="lower", aspect="auto",
                   extent=[float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 0.0,
                           float(f[0]) if f.size else 0.0, float(f[-1]) if f.size else 0.0])
    ax.set_title("Spectrogram (input) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_input.png"), fig)

    # output spectrogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(S1, origin="lower", aspect="auto",
                   extent=[float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 0.0,
                           float(f[0]) if f.size else 0.0, float(f[-1]) if f.size else 0.0])
    ax.set_title("Spectrogram (output) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_output.png"), fig)

    # diff spectrogram
    Sd = (S1 - S0).astype(np.float32)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Sd, origin="lower", aspect="auto",
                   extent=[float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 0.0,
                           float(f[0]) if f.size else 0.0, float(f[-1]) if f.size else 0.0])
    ax.set_title("Spectrogram (output - input) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Δ dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_diff.png"), fig)

    # ---- Attenuation maps (STFT-domain processing) ----
    t_frames = dbg_data.get("t", None)

    def plot_att_map(name: str, f_key: str, a_key: str):
        if f_key not in dbg_data or a_key not in dbg_data or t_frames is None:
            return
        F = dbg_data[f_key]
        A = dbg_data[a_key]  # (F, T)
        if A.size == 0:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111)
        extent = [float(t_frames[0]), float(t_frames[-1]), float(F[0]), float(F[-1])]
        im = ax.imshow(A, origin="lower", aspect="auto", extent=extent)
        ax.set_title(f"Attenuation map: {name} [dB]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Attenuation (dB)")
        _plot_and_save_png(os.path.join(dbg_dir, f"atten_{name}.png"), fig)

    plot_att_map("denoise", "f_dn", "att_dn_db")
    plot_att_map("deres", "f_deq", "att_deq_db")
    plot_att_map("shimmer", "f_sh", "att_sh_db")

    # ---- Time-series metrics ----
    if t_frames is not None and t_frames.size > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t_frames, dbg_data.get("w_noise", np.zeros_like(t_frames)), label="noise-likeness (0..1)")
        ax.plot(t_frames, dbg_data.get("w_trans", np.zeros_like(t_frames)), label="transient gate (0..1)")
        ax.plot(t_frames, dbg_data.get("dn_depth", np.zeros_like(t_frames)), label="denoise depth")
        ax.plot(t_frames, dbg_data.get("deq_depth", np.zeros_like(t_frames)), label="deres depth")
        ax.plot(t_frames, dbg_data.get("sh_depth", np.zeros_like(t_frames)), label="shimmer depth")
        ax.set_title("Frame metrics / gates")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8)
        _plot_and_save_png(os.path.join(dbg_dir, "frame_metrics.png"), fig)

        # Mean attenuation vs time for each stage
        fig = plt.figure()
        ax = fig.add_subplot(111)

        def mean_att(a_key: str) -> Optional[np.ndarray]:
            if a_key not in dbg_data:
                return None
            A = dbg_data[a_key]
            if A.size == 0:
                return None
            return np.mean(A, axis=0)

        m_dn = mean_att("att_dn_db")
        m_deq = mean_att("att_deq_db")
        m_sh = mean_att("att_sh_db")
        if m_dn is not None:
            ax.plot(t_frames, m_dn, label="mean denoise att (dB)")
        if m_deq is not None:
            ax.plot(t_frames, m_deq, label="mean deres att (dB)")
        if m_sh is not None:
            ax.plot(t_frames, m_sh, label="mean shimmer att (dB)")
        ax.set_title("Mean attenuation over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dB")
        ax.legend(loc="upper right", fontsize=8)
        _plot_and_save_png(os.path.join(dbg_dir, "mean_attenuation.png"), fig)

    # ---- Noise PSD estimate (optional) ----
    if "noise_psd_dn_db" in dbg_data and t_frames is not None:
        N = dbg_data["noise_psd_dn_db"]  # (F, T)
        F = dbg_data.get("f_dn", None)
        if F is not None and N.size:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t_frames, np.median(N, axis=0))
            ax.set_title("Estimated noise floor (median over denoise band) [dB]")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dB (power)")
            _plot_and_save_png(os.path.join(dbg_dir, "noise_floor_estimate.png"), fig)

    # Write a tiny readme
    with open(os.path.join(dbg_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Debug outputs:\n"
            "- summary.json: key measurements + settings\n"
            "- debug_data.npz: internal arrays (attenuation maps, gates)\n"
            "- spectrogram_*.png: before/after/diff spectrograms\n"
            "- atten_*.png: per-stage attenuation maps\n"
            "- frame_metrics.png: gates/depth over time\n"
            "- mean_attenuation.png: average attenuation over time\n"
        )


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Repair AI/codec-ish shimmer + smart denoise + smart de-resonator + optional loudness/true-peak delivery + debug visuals.")

    ap.add_argument("input")
    ap.add_argument("output")

    # --- De-shimmer band ---
    ap.add_argument("--start-hz", type=float, default=5100.0)
    ap.add_argument("--end-hz", type=float, default=7200.0)
    ap.add_argument("--center-hz", type=float, default=None)
    ap.add_argument("--width-cents", type=float, default=None)
    ap.add_argument("--edge-hz", type=float, default=200.0)

    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)

    ap.add_argument("--flat-start", type=float, default=0.25)
    ap.add_argument("--flat-end", type=float, default=0.70)

    ap.add_argument("--freq-med-bins", type=int, default=9)
    ap.add_argument("--thr-db", type=float, default=8.0)
    ap.add_argument("--slope", type=float, default=0.6)
    ap.add_argument("--density-lo", type=float, default=0.02)
    ap.add_argument("--density-hi", type=float, default=0.15)

    ap.add_argument("--flux-thr-db", type=float, default=6.0)
    ap.add_argument("--flux-range-db", type=float, default=8.0)

    ap.add_argument("--noise-resynth", type=float, default=0.0)
    ap.add_argument("--mix", type=float, default=1.0)

    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--fade-ms", type=float, default=5.0)

    # --- Smart denoise ---
    ap.add_argument("--denoise", type=float, default=0.0, help="0..1 spectral noise floor reduction")
    ap.add_argument("--dn-start-hz", type=float, default=120.0)
    ap.add_argument("--dn-end-hz", type=float, default=16000.0)
    ap.add_argument("--dn-edge-hz", type=float, default=200.0)
    ap.add_argument("--dn-floor-db", type=float, default=-18.0)
    ap.add_argument("--dn-psd-smooth-ms", type=float, default=50.0)
    ap.add_argument("--dn-minwin-ms", type=float, default=400.0)
    ap.add_argument("--dn-up-db-per-s", type=float, default=3.0)
    ap.add_argument("--dn-attack-ms", type=float, default=5.0)
    ap.add_argument("--dn-release-ms", type=float, default=120.0)
    ap.add_argument("--dn-freq-smooth-bins", type=int, default=3)

    # --- Smart de-resonator ---
    ap.add_argument("--deres", type=float, default=0.0, help="0..1 de-resonator (dynamic EQ)")
    ap.add_argument("--deq-start-hz", type=float, default=180.0)
    ap.add_argument("--deq-end-hz", type=float, default=12000.0)
    ap.add_argument("--deq-edge-hz", type=float, default=150.0)
    ap.add_argument("--deq-freq-med-bins", type=int, default=31)
    ap.add_argument("--deq-thr-db", type=float, default=6.0)
    ap.add_argument("--deq-slope", type=float, default=0.7)
    ap.add_argument("--deq-max-att-db", type=float, default=8.0)
    ap.add_argument("--deq-density-lo", type=float, default=0.03)
    ap.add_argument("--deq-density-hi", type=float, default=0.20)
    ap.add_argument("--deq-persist-ms", type=float, default=600.0)
    ap.add_argument("--deq-persist-thr-db", type=float, default=2.5)
    ap.add_argument("--deq-freq-smooth-bins", type=int, default=5)
    ap.add_argument("--deq-tonal-boost-db", type=float, default=6.0)
    ap.add_argument("--deq-time-floor", action="store_true", help="Enable time-stabilized per-bin floor to target stationary ringing lines.")
    ap.add_argument("--deq-floor-smooth-ms", type=float, default=80.0)
    ap.add_argument("--deq-floor-rise-db-per-s", type=float, default=1.0)
    ap.add_argument("--deq-floor-thr-db", type=float, default=3.0)

    ap.add_argument("--delta-listen", action="store_true", help="Output removed-only (input - output). Useful for dialing parameters.")

    # --- Downward expander ---
    ap.add_argument("--expander", action="store_true", help="Enable downward expander in a band (helps correlated grit in tails).")
    ap.add_argument("--exp-start-hz", type=float, default=3000.0)
    ap.add_argument("--exp-end-hz", type=float, default=8000.0)
    ap.add_argument("--exp-threshold-db", type=float, default=-45.0)
    ap.add_argument("--exp-ratio", type=float, default=2.0)
    ap.add_argument("--exp-attack-ms", type=float, default=10.0)
    ap.add_argument("--exp-release-ms", type=float, default=150.0)

    # --- HPSS-ish ---
    ap.add_argument("--hpss", action="store_true", help="Enable HPSS-ish harmonic mask inside a band.")
    ap.add_argument("--hpss-start-hz", type=float, default=3000.0)
    ap.add_argument("--hpss-end-hz", type=float, default=8000.0)
    ap.add_argument("--hpss-time-frames", type=int, default=21)
    ap.add_argument("--hpss-freq-bins", type=int, default=17)
    ap.add_argument("--hpss-no-harmonic-only", action="store_true", help="Do not restrict processing to harmonic component.")

    # --- Phase blur ---
    ap.add_argument("--phase-blur", type=float, default=0.0, help="0..1 random-phase blend in pb band (texture masking).")
    ap.add_argument("--pb-start-hz", type=float, default=3000.0)
    ap.add_argument("--pb-end-hz", type=float, default=8000.0)
    ap.add_argument("--pb-no-harmonic-only", action="store_true")

    # --- HF resynthesis (nuclear) ---
    ap.add_argument("--hf-resynth", action="store_true", help="Replace HF with harmonics generated from a low band.")
    ap.add_argument("--hf-lp-hz", type=float, default=3000.0)
    ap.add_argument("--hf-src-lo-hz", type=float, default=1000.0)
    ap.add_argument("--hf-src-hi-hz", type=float, default=2000.0)
    ap.add_argument("--hf-drive", type=float, default=2.0)
    ap.add_argument("--hf-hp-hz", type=float, default=3000.0)
    ap.add_argument("--hf-mix", type=float, default=0.35)
    ap.add_argument("--hf-tilt-lp-hz", type=float, default=12000.0)
    ap.add_argument("--hf-zero-phase", action="store_true", help="Use zero-phase filters (offline). Disable for realtime-ish chunk processing by setting --no-hf-zero-phase.")
    ap.add_argument("--no-hf-zero-phase", action="store_true", help="Disable zero-phase filtering for HF resynth.")

    # --- Delivery mastering ---
    ap.add_argument("--master", action="store_true", help="Enable loudness normalization + true-peak-ish limiting post stage.")
    ap.add_argument("--hp-hz", type=float, default=20.0)
    ap.add_argument("--target-lufs", type=float, default=-14.0, help="Target integrated loudness in LUFS (requires pyloudnorm). Set to 999 to disable.")
    ap.add_argument("--target-rms-dbfs", type=float, default=-16.0, help="Fallback target RMS in dBFS if pyloudnorm missing or LUFS disabled.")
    ap.add_argument("--norm-max-gain-db", type=float, default=12.0)
    ap.add_argument("--norm-max-atten-db", type=float, default=24.0)
    ap.add_argument("--ceiling-dbtp", type=float, default=-1.0)
    ap.add_argument("--lim-lookahead-ms", type=float, default=5.0)
    ap.add_argument("--lim-release-ms", type=float, default=100.0)
    ap.add_argument("--tp-os", type=int, default=4, help="Oversampling factor for true-peak-ish limiting/measurement (4x is common).")

    # Output
    ap.add_argument("--subtype", type=str, default="PCM_24", help="SoundFile subtype, e.g. PCM_16, PCM_24, FLOAT.")

    # Debug / diagnostics
    ap.add_argument("--debug", action="store_true", help="Write debug folder with plots and internal arrays.")
    ap.add_argument("--debug-dir", type=str, default=None)
    ap.add_argument("--debug-stride", type=int, default=8)
    ap.add_argument("--debug-spec-nfft", type=int, default=2048)
    ap.add_argument("--debug-spec-hop", type=int, default=512)
    ap.add_argument("--debug-spec-max-frames", type=int, default=2200)
    ap.add_argument("--debug-spec-max-hz", type=float, default=20000.0)

    ap.add_argument("--write-diff", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=True)
    x = x.astype(np.float32, copy=False)

    # Compute band from center if provided
    start_hz = float(args.start_hz)
    end_hz = float(args.end_hz)
    if args.center_hz is not None and args.width_cents is not None:
        start_hz, end_hz = band_from_center(float(args.center_hz), float(args.width_cents))

    p = Params(
        start_hz=start_hz,
        end_hz=end_hz,
        edge_hz=float(args.edge_hz),
        n_fft=int(args.n_fft),
        hop=int(args.hop),

        flat_start=float(args.flat_start),
        flat_end=float(args.flat_end),

        freq_med_bins=int(args.freq_med_bins),
        thr_db=float(args.thr_db),
        slope=float(args.slope),
        density_lo=float(args.density_lo),
        density_hi=float(args.density_hi),

        flux_thr_db=float(args.flux_thr_db),
        flux_range_db=float(args.flux_range_db),

        noise_resynth=float(args.noise_resynth),
        mix=float(args.mix),

        pad=(not args.no_pad),
        fade_ms=float(args.fade_ms),

        seed=int(args.seed),

        denoise=float(args.denoise),
        dn_start_hz=float(args.dn_start_hz),
        dn_end_hz=float(args.dn_end_hz),
        dn_edge_hz=float(args.dn_edge_hz),
        dn_floor_db=float(args.dn_floor_db),
        dn_psd_smooth_ms=float(args.dn_psd_smooth_ms),
        dn_minwin_ms=float(args.dn_minwin_ms),
        dn_up_db_per_s=float(args.dn_up_db_per_s),
        dn_attack_ms=float(args.dn_attack_ms),
        dn_release_ms=float(args.dn_release_ms),
        dn_freq_smooth_bins=int(args.dn_freq_smooth_bins),

        deres=float(args.deres),
        deq_start_hz=float(args.deq_start_hz),
        deq_end_hz=float(args.deq_end_hz),
        deq_edge_hz=float(args.deq_edge_hz),
        deq_freq_med_bins=int(args.deq_freq_med_bins),
        deq_thr_db=float(args.deq_thr_db),
        deq_slope=float(args.deq_slope),
        deq_max_att_db=float(args.deq_max_att_db),
        deq_density_lo=float(args.deq_density_lo),
        deq_density_hi=float(args.deq_density_hi),
        deq_persist_ms=float(args.deq_persist_ms),
        deq_persist_thr_db=float(args.deq_persist_thr_db),
        deq_freq_smooth_bins=int(args.deq_freq_smooth_bins),
        deq_tonal_boost_db=float(args.deq_tonal_boost_db),
        deq_time_floor=bool(args.deq_time_floor),
        deq_floor_smooth_ms=float(args.deq_floor_smooth_ms),
        deq_floor_rise_db_per_s=float(args.deq_floor_rise_db_per_s),
        deq_floor_thr_db=float(args.deq_floor_thr_db),

        delta_listen=bool(args.delta_listen),

        expander=bool(args.expander),
        exp_start_hz=float(args.exp_start_hz),
        exp_end_hz=float(args.exp_end_hz),
        exp_threshold_db=float(args.exp_threshold_db),
        exp_ratio=float(args.exp_ratio),
        exp_attack_ms=float(args.exp_attack_ms),
        exp_release_ms=float(args.exp_release_ms),

        hpss=bool(args.hpss),
        hpss_start_hz=float(args.hpss_start_hz),
        hpss_end_hz=float(args.hpss_end_hz),
        hpss_time_frames=int(args.hpss_time_frames),
        hpss_freq_bins=int(args.hpss_freq_bins),
        hpss_harmonic_only=(not bool(args.hpss_no_harmonic_only)),

        phase_blur=float(args.phase_blur),
        pb_start_hz=float(args.pb_start_hz),
        pb_end_hz=float(args.pb_end_hz),
        pb_harmonic_only=(not bool(args.pb_no_harmonic_only)),

        hf_resynth=bool(args.hf_resynth),
        hf_lp_hz=float(args.hf_lp_hz),
        hf_src_lo_hz=float(args.hf_src_lo_hz),
        hf_src_hi_hz=float(args.hf_src_hi_hz),
        hf_drive=float(args.hf_drive),
        hf_hp_hz=float(args.hf_hp_hz),
        hf_mix=float(args.hf_mix),
        hf_tilt_lp_hz=float(args.hf_tilt_lp_hz),
        hf_zero_phase=(not bool(args.no_hf_zero_phase)),
    )

    target_lufs = None if float(args.target_lufs) >= 998.0 else float(args.target_lufs)
    mp = MasterParams(
        enabled=bool(args.master),
        hp_hz=float(args.hp_hz),
        target_lufs=target_lufs,
        target_rms_dbfs=float(args.target_rms_dbfs) if args.target_rms_dbfs is not None else None,
        norm_max_gain_db=float(args.norm_max_gain_db),
        norm_max_atten_db=float(args.norm_max_atten_db),
        ceiling_dbtp=float(args.ceiling_dbtp),
        lookahead_ms=float(args.lim_lookahead_ms),
        release_ms=float(args.lim_release_ms),
        os_factor=int(args.tp_os),
    )

    dp = DebugParams(
        enabled=bool(args.debug),
        debug_dir=args.debug_dir,
        stride=int(args.debug_stride),
        spec_n_fft=int(args.debug_spec_nfft),
        spec_hop=int(args.debug_spec_hop),
        spec_max_frames=int(args.debug_spec_max_frames),
        spec_max_hz=float(args.debug_spec_max_hz),
    )

    # ---- Measurements (input) ----
    summary: Dict[str, Any] = {}
    summary["input_path"] = args.input
    summary["output_path"] = args.output
    summary["sr"] = int(sr)
    summary["channels"] = int(x.shape[1])
    summary["duration_s"] = float(x.shape[0] / sr)
    summary["params"] = asdict(p)
    summary["master_params"] = asdict(mp)
    summary["debug_params"] = asdict(dp)

    meas_in = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(x)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(x, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(x)),
        "lufs": measure_lufs(x, sr),
    }
    summary["measure_in"] = meas_in

    # ---- Setup debug collector (needs FFT freqs/idx) ----
    dbg_collector = None
    dbg_data: Dict[str, Any] = {}
    if dp.enabled:
        n_fft = int(p.n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        nyq = float(freqs[-1])

        def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
            lo = float(max(0.0, lo_hz))
            hi = float(min(nyq, hi_hz))
            if hi <= lo:
                return np.array([], dtype=np.int64)
            return np.where((freqs >= lo) & (freqs <= hi))[0]

        sh_idx = _idx(p.start_hz, p.end_hz)
        dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
        deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)

        pad_offset = n_fft if p.pad else 0
        dbg_collector = DebugCollector(
            sr=sr,
            freqs=freqs,
            dn_idx=dn_idx,
            deq_idx=deq_idx,
            sh_idx=sh_idx,
            stride=dp.stride,
            pad_offset=pad_offset,
            duration_s=float(x.shape[0] / sr),
        )

    # ---- STFT repair ----
    y_repaired = process_stft(x, sr, p, dbg=dbg_collector)
    # Optional "nuclear" HF resynthesis after STFT repair (off by default)
    y_repaired = hf_resynth_post(y_repaired, sr, p, artifact_conf=_LAST_ARTIFACT_CONF_FRAMES)
    y_rep_2d = _as_2d(y_repaired)

    meas_rep = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(y_rep_2d)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(y_rep_2d, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(y_rep_2d)),
        "lufs": measure_lufs(y_rep_2d, sr),
    }
    summary["measure_after_repair"] = meas_rep

    # ---- Optional master/deliver ----
    y_out, master_info = master_post(y_repaired, sr, mp, debug=dp.enabled)
    y_out_2d = _as_2d(y_out)

    meas_out = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(y_out_2d)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(y_out_2d, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(y_out_2d)),
        "lufs": measure_lufs(y_out_2d, sr),
    }
    summary["measure_out"] = meas_out
    summary["master_info"] = master_info

    # ---- Write output ----
    subtype = str(args.subtype)
    sf.write(args.output, y_out_2d.astype(np.float32, copy=False), sr, subtype=subtype)

    # Optional diff
    if args.write_diff:
        diff = (x[: y_out_2d.shape[0], :] - y_out_2d).astype(np.float32)
        sf.write(args.write_diff, diff, sr, subtype=subtype)

    # ---- Finalize debug ----
    if dp.enabled and dbg_collector is not None:
        dbg_data = dbg_collector.finalize()

        # "Top frequencies" summary from attenuation maps
        def top_freqs(f: np.ndarray, A: np.ndarray, n: int = 8) -> List[Dict[str, float]]:
            # A is (F, T) attenuation dB
            if A.size == 0:
                return []
            mean_att = np.mean(A, axis=1)
            idx = np.argsort(mean_att)[::-1][:n]
            out = []
            for i in idx:
                out.append({"hz": float(f[i]), "mean_att_db": float(mean_att[i])})
            return out

        tops: Dict[str, Any] = {}
        if "att_sh_db" in dbg_data:
            tops["shimmer_top_bins"] = top_freqs(dbg_data["f_sh"], dbg_data["att_sh_db"])
        if "att_deq_db" in dbg_data:
            tops["deres_top_bins"] = top_freqs(dbg_data["f_deq"], dbg_data["att_deq_db"])
        if "att_dn_db" in dbg_data:
            tops["denoise_top_bins"] = top_freqs(dbg_data["f_dn"], dbg_data["att_dn_db"])

        summary["top_problem_bins"] = tops

        # Pick debug dir
        dbg_dir = dp.debug_dir
        if dbg_dir is None or dbg_dir.strip() == "":
            base = os.path.splitext(os.path.basename(args.output))[0]
            dbg_dir = os.path.join(os.path.dirname(args.output) or ".", f"{base}_debug")
        render_debug(
            dbg_dir=dbg_dir,
            dbg_data=dbg_data,
            x_in=x,
            y_repaired=_as_2d(y_repaired),
            y_out=_as_2d(y_out),
            sr=sr,
            dp=dp,
            summary=summary,
        )

        # Also print a short summary to stdout
        print("=== Debug summary ===")
        print(json.dumps({
            "measure_in": meas_in,
            "measure_after_repair": meas_rep,
            "measure_out": meas_out,
            "debug_dir": dbg_dir,
        }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
