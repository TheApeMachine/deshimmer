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
from scipy.ndimage import median_filter, uniform_filter1d, maximum_filter1d
from scipy.signal import butter, sosfiltfilt, resample_poly, lfilter, lfilter_zi


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

    # ----------------------------------------------------------------------
    # Optional: phase blur (random-phase blend) in a selected band (texture masking)
    # ----------------------------------------------------------------------
    phase_blur: float = 0.0
    pb_start_hz: float = 3000.0
    pb_end_hz: float = 8000.0
    pb_harmonic_only: bool = True

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
def process_stft(x: np.ndarray, sr: int, p: Params, dbg: Optional[DebugCollector] = None) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    x = x.astype(np.float32, copy=False)

    n_fft = int(p.n_fft)
    hop = int(p.hop)
    if hop <= 0 or n_fft <= 0:
        raise ValueError("Invalid n_fft/hop")

    # Pad to avoid boundary clicks
    pad_offset = n_fft if p.pad else 0
    if p.pad:
        x0 = np.pad(x, ((n_fft, n_fft), (0, 0)), mode="constant")
    else:
        x0 = x

    n_samples, n_ch = x0.shape
    win = np.hanning(n_fft).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyq = float(freqs[-1])
    eps = 1e-12

    def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
        lo = float(max(0.0, lo_hz))
        hi = float(min(nyq, hi_hz))
        if hi <= lo:
            return np.array([], dtype=np.int64)
        return np.where((freqs >= lo) & (freqs <= hi))[0]

    # --- Shimmer band ---
    start_hz = float(max(0.0, p.start_hz))
    end_hz = float(min(nyq, p.end_hz))
    if end_hz <= start_hz:
        raise ValueError("end_hz must be > start_hz")

    sh_idx = _idx(start_hz, end_hz)
    if sh_idx.size < 8:
        raise ValueError("Shimmer band too narrow for this FFT size; increase n_fft or widen band.")
    w_sh = _edge_taper(freqs, sh_idx, start_hz, end_hz, p.edge_hz)

    # --- Expander band ---
    exp_idx = _idx(getattr(p, "exp_start_hz", 3000.0), getattr(p, "exp_end_hz", 8000.0))

    # --- HPSS band ---
    hpss_idx = _idx(getattr(p, "hpss_start_hz", 3000.0), getattr(p, "hpss_end_hz", 8000.0))

    # --- Phase blur band ---
    pb_idx = _idx(getattr(p, "pb_start_hz", 3000.0), getattr(p, "pb_end_hz", 8000.0))

    # --- Denoise band ---
    dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
    w_dn = _edge_taper(freqs, dn_idx, float(max(0.0, p.dn_start_hz)), float(min(nyq, p.dn_end_hz)), p.dn_edge_hz)

    # --- De-resonator band ---
    deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)
    w_deq = _edge_taper(freqs, deq_idx, float(max(0.0, p.deq_start_hz)), float(min(nyq, p.deq_end_hz)), p.deq_edge_hz)

    # --- Denoise state (minimum-statistics-ish) ---
    dn_strength = float(np.clip(p.denoise, 0.0, 1.0))
    psd_sm = None
    noise_psd = None
    block_min = None
    block_ctr = 0
    minwin_frames = max(4, int((sr * (p.dn_minwin_ms / 1000.0)) / hop))
    a_psd = _frame_coeff(hop, sr, p.dn_psd_smooth_ms)
    dn_att = _frame_coeff(hop, sr, p.dn_attack_ms)
    dn_rel = _frame_coeff(hop, sr, p.dn_release_ms)
    dn_floor = float(10.0 ** (float(p.dn_floor_db) / 20.0))
    dn_freq_smooth = int(max(1, p.dn_freq_smooth_bins))
    dn_gain_sm = np.ones(dn_idx.size, dtype=np.float32) if dn_idx.size else None

    # noise PSD is power-domain; approximate allowed rise per block
    dn_block_sec = float(minwin_frames) * float(hop) / float(sr)
    dn_up_lin = float(10.0 ** ((float(p.dn_up_db_per_s) * dn_block_sec) / 10.0))

    # --- De-resonator state ---
    deq_strength = float(np.clip(p.deres, 0.0, 1.0))
    deq_persist = np.zeros(deq_idx.size, dtype=np.float32) if deq_idx.size else None
    a_persist = _frame_coeff(hop, sr, p.deq_persist_ms)
    deq_freq_med = int(max(3, p.deq_freq_med_bins))
    if deq_freq_med % 2 == 0:
        deq_freq_med += 1
    deq_freq_smooth = int(max(1, p.deq_freq_smooth_bins))

    # Time-stabilized floor for stationary ringing
    deq_time_floor = bool(getattr(p, "deq_time_floor", False))
    deq_psd_sm = np.zeros(deq_idx.size, dtype=np.float32) if (deq_time_floor and deq_idx.size) else None
    deq_floor_psd = np.zeros(deq_idx.size, dtype=np.float32) if (deq_time_floor and deq_idx.size) else None
    a_deq_psd = _frame_coeff(hop, sr, getattr(p, "deq_floor_smooth_ms", 80.0)) if deq_time_floor else 0.0
    deq_floor_rise_db_per_s = float(getattr(p, "deq_floor_rise_db_per_s", 1.0)) if deq_time_floor else 0.0
    # power-domain rise per frame
    deq_floor_rise = float(10.0 ** ((deq_floor_rise_db_per_s * (float(hop) / float(sr))) / 10.0)) if deq_time_floor else 1.0

    rng = np.random.default_rng(int(p.seed))

    # --- Expander state ---
    exp_enabled = bool(getattr(p, "expander", False))
    exp_thr_db = float(getattr(p, "exp_threshold_db", -45.0))
    exp_ratio = float(max(1.0, getattr(p, "exp_ratio", 2.0)))
    exp_att = _frame_coeff(hop, sr, float(getattr(p, "exp_attack_ms", 10.0))) if exp_enabled else 0.0
    exp_rel = _frame_coeff(hop, sr, float(getattr(p, "exp_release_ms", 150.0))) if exp_enabled else 0.0
    exp_g_sm = 1.0

    # --- HPSS-ish state (ring buffer of log-mag for band, for time-median) ---
    hpss_enabled = bool(getattr(p, "hpss", False))
    hpss_harm_only = bool(getattr(p, "hpss_harmonic_only", True))
    hpss_tf = int(max(3, getattr(p, "hpss_time_frames", 21))) if hpss_enabled else 0
    if hpss_tf and (hpss_tf % 2 == 0):
        hpss_tf += 1
    hpss_ff = int(max(3, getattr(p, "hpss_freq_bins", 17))) if hpss_enabled else 0
    if hpss_ff and (hpss_ff % 2 == 0):
        hpss_ff += 1
    hpss_buf = np.zeros((hpss_tf, hpss_idx.size), dtype=np.float32) if (hpss_enabled and hpss_idx.size) else None
    hpss_buf_i = 0
    hpss_buf_fill = 0

    # --- Phase blur state ---
    pb_amt = float(np.clip(getattr(p, "phase_blur", 0.0), 0.0, 1.0))
    pb_harm_only = bool(getattr(p, "pb_harmonic_only", True))

    y = np.zeros((n_samples + n_fft, n_ch), dtype=np.float32)
    wsum = np.zeros(n_samples + n_fft, dtype=np.float32)

    prev_band_db = None
    for s in range(0, n_samples, hop):
        # assemble frame (zero-pad last)
        frame = np.zeros((n_fft, n_ch), dtype=np.float32)
        chunk = x0[s:s + n_fft, :]
        frame[:chunk.shape[0], :] = chunk
        frame_w = frame * win[:, None]

        spec = np.fft.rfft(frame_w, n=n_fft, axis=0)  # (freq, ch)

        # Mono power spectrum
        psd = np.mean(np.abs(spec) ** 2, axis=1).astype(np.float32) + eps

        # ----------------------------------------------------------
        # (0) Optional expander in artifact band (push down tails)
        # ----------------------------------------------------------
        if exp_enabled and exp_idx.size:
            band_p = float(np.mean(psd[exp_idx])) if exp_idx.size else 0.0
            band_db_exp = 10.0 * math.log10(max(band_p, eps))
            if band_db_exp < exp_thr_db:
                diff_db = exp_thr_db - band_db_exp
                red_db = diff_db * (exp_ratio - 1.0)
                g_inst = float(10.0 ** (-red_db / 20.0))
            else:
                g_inst = 1.0
            # Smooth (down fast, up slow)
            if g_inst < exp_g_sm:
                exp_g_sm = exp_att * exp_g_sm + (1.0 - exp_att) * g_inst
            else:
                exp_g_sm = exp_rel * exp_g_sm + (1.0 - exp_rel) * g_inst
            spec[exp_idx, :] *= float(exp_g_sm)

        # Noise-likeness (in denoise band if present; else full)
        if dn_idx.size:
            mag_dn = np.mean(np.abs(spec[dn_idx, :]), axis=1).astype(np.float32) + eps
            Pdn = mag_dn ** 2
            flat = float(np.exp(np.mean(np.log(Pdn + eps))) / (np.mean(Pdn) + eps))
            band_db = 10.0 * math.log10(float(np.mean(Pdn)) + eps)
        else:
            flat = float(np.exp(np.mean(np.log(psd + eps))) / (np.mean(psd) + eps))
            band_db = 10.0 * math.log10(float(np.mean(psd)) + eps)

        w_noise_full = float(np.clip((flat - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))

        # Transient protect based on band energy jump
        if prev_band_db is None:
            flux = 0.0
        else:
            flux = max(0.0, band_db - prev_band_db)
        prev_band_db = band_db

        w_trans = float(np.clip((flux - p.flux_thr_db) / max(1e-6, p.flux_range_db), 0.0, 1.0))
        w_nontrans = 1.0 - w_trans

        # We'll fill these for debug if used
        dn_depth = 0.0
        deq_depth = 0.0
        sh_depth = 0.0
        att_dn_db_dbg = None
        att_deq_db_dbg = None
        att_sh_db_dbg = None
        noise_psd_dn_dbg = None

        # ----------------------------------------------------------
        # (A) Smart spectral denoise
        # ----------------------------------------------------------
        if dn_strength > 1e-6 and dn_idx.size:
            if psd_sm is None:
                psd_sm = psd.copy()
                noise_psd = psd.copy()
                block_min = psd.copy()
                block_ctr = 1
            else:
                psd_sm = a_psd * psd_sm + (1.0 - a_psd) * psd
                block_min = np.minimum(block_min, psd_sm)
                block_ctr += 1

            if block_ctr >= minwin_frames:
                noise_psd = np.minimum(noise_psd * dn_up_lin, block_min).astype(np.float32, copy=False)
                block_min.fill(np.inf)
                block_ctr = 0

            snr = psd_sm[dn_idx] / (noise_psd[dn_idx] + eps)

            # Wiener-ish gain: snr/(snr+k). k grows with strength.
            k = 1.0 + 3.0 * dn_strength
            g_inst = snr / (snr + k)
            g_inst = dn_floor + (1.0 - dn_floor) * g_inst
            g_inst = np.clip(g_inst, dn_floor, 1.0).astype(np.float32)

            # Attack/release smoothing on gain (down fast, up slow)
            down = g_inst < dn_gain_sm
            dn_gain_sm[down] = dn_att * dn_gain_sm[down] + (1.0 - dn_att) * g_inst[down]
            dn_gain_sm[~down] = dn_rel * dn_gain_sm[~down] + (1.0 - dn_rel) * g_inst[~down]

            # Smooth across frequency to reduce musical-noise
            g_dn = uniform_filter1d(dn_gain_sm, size=dn_freq_smooth, mode="nearest").astype(np.float32)

            dn_depth = dn_strength * (0.5 + 0.5 * w_noise_full) * w_nontrans
            g_eff_dn = 1.0 - (dn_depth * w_dn) * (1.0 - g_dn)
            spec[dn_idx, :] *= g_eff_dn[:, None]

            if dbg is not None and dbg.want():
                att_dn_db_dbg = (-20.0 * np.log10(np.clip(g_eff_dn, 1e-6, 1.0))).astype(np.float32)
                noise_psd_dn_dbg = (10.0 * np.log10(noise_psd[dn_idx] + eps)).astype(np.float32)

        # ----------------------------------------------------------
        # (B) Smart de-resonator (dynamic EQ)
        # ----------------------------------------------------------
        if deq_strength > 1e-6 and deq_idx.size:
            mag_deq = np.mean(np.abs(spec[deq_idx, :]), axis=1).astype(np.float32) + eps
            L = np.log(mag_deq)
            L_med = median_filter(L, size=deq_freq_med, mode="nearest")
            residual_db_freq = (L - L_med) * (20.0 / np.log(10.0))

            # Optional: compare against a time-stabilized per-bin floor to catch stationary ringing lines.
            residual_db_time = None
            if deq_time_floor and deq_psd_sm is not None and deq_floor_psd is not None:
                psd_deq = psd[deq_idx].astype(np.float32, copy=False)
                if not np.isfinite(deq_floor_psd).all() or float(np.max(deq_floor_psd)) == 0.0:
                    deq_psd_sm[:] = psd_deq
                    deq_floor_psd[:] = psd_deq
                else:
                    deq_psd_sm[:] = a_deq_psd * deq_psd_sm + (1.0 - a_deq_psd) * psd_deq
                    # instant fall, slow rise: tracks "quietest" the bin gets
                    deq_floor_psd[:] = np.minimum(deq_floor_psd * deq_floor_rise, deq_psd_sm)
                residual_db_time = (10.0 * np.log10((deq_psd_sm + eps) / (deq_floor_psd + eps))).astype(np.float32)

            residual_db = residual_db_freq
            if residual_db_time is not None:
                residual_db = np.maximum(residual_db_freq.astype(np.float32, copy=False), residual_db_time)

            # Raise threshold when tonal (low flatness) to avoid chewing harmonics
            thr_eff = float(p.deq_thr_db) + float(p.deq_tonal_boost_db) * (1.0 - w_noise_full)
            over = residual_db - thr_eff
            pos = np.maximum(0.0, over).astype(np.float32)

            mask = pos > 0.0
            density = float(np.mean(mask)) if mask.size else 0.0
            # Density backoff protects broad legitimate content, but stationary ringing can be "thick".
            # If time-floor mode is enabled, trust persistence/tonal gating more and don't back off on density.
            if deq_time_floor:
                w_narrow = 1.0
            else:
                w_narrow = 1.0 - float(np.clip((density - p.deq_density_lo) / max(1e-6, (p.deq_density_hi - p.deq_density_lo)), 0.0, 1.0))

            # Persistence memory: stationary resonances accumulate, moving peaks don't
            deq_persist = a_persist * deq_persist + (1.0 - a_persist) * pos
            gate = np.clip(deq_persist / max(1e-6, float(p.deq_persist_thr_db)), 0.0, 1.0)

            att_db = float(p.deq_slope) * pos * gate
            att_db = np.minimum(att_db, float(p.deq_max_att_db)).astype(np.float32)
            gain = (10.0 ** (-att_db / 20.0)).astype(np.float32)

            deq_depth = deq_strength * w_nontrans * w_narrow
            g_eff_deq = 1.0 - (deq_depth * w_deq) * (1.0 - gain)
            if deq_freq_smooth > 1:
                g_eff_deq = uniform_filter1d(g_eff_deq, size=deq_freq_smooth, mode="nearest").astype(np.float32)
            spec[deq_idx, :] *= g_eff_deq[:, None]

            if dbg is not None and dbg.want():
                att_deq_db_dbg = (-20.0 * np.log10(np.clip(g_eff_deq, 1e-6, 1.0))).astype(np.float32)

        # ----------------------------------------------------------
        # (C) Original shimmer suppression band
        # ----------------------------------------------------------
        band_spec = spec[sh_idx, :]
        mag_band = np.mean(np.abs(band_spec), axis=1).astype(np.float32) + eps

        # Flatness in shimmer band
        P = mag_band ** 2
        flat_sh = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))
        w_noise_sh = float(np.clip((flat_sh - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))

        # Local median baseline across frequency (log-mag)
        L = np.log(mag_band)
        k = int(max(3, p.freq_med_bins))
        if k % 2 == 0:
            k += 1
        L_med = median_filter(L, size=k, mode="nearest")
        residual_db = (L - L_med) * (20.0 / np.log(10.0))

        over = residual_db - float(p.thr_db)
        mask = over > 0.0
        density = float(np.mean(mask)) if mask.size else 0.0
        w_narrow = 1.0 - float(np.clip((density - p.density_lo) / max(1e-6, (p.density_hi - p.density_lo)), 0.0, 1.0))

        sh_depth = w_noise_sh * w_nontrans * w_narrow

        att_db = np.zeros_like(over, dtype=np.float32)
        att_db[mask] = (float(p.slope) * (over[mask])).astype(np.float32)
        gain = (10.0 ** (-att_db / 20.0)).astype(np.float32)

        g_eff_sh = 1.0 - (sh_depth * w_sh) * (1.0 - gain)
        spec[sh_idx, :] *= g_eff_sh[:, None]

        # ----------------------------------------------------------
        # (D) Optional HPSS-ish harmonic mask and phase blur (texture)
        # ----------------------------------------------------------
        Mh = None
        if hpss_enabled and hpss_idx.size:
            mag_h = np.mean(np.abs(spec[hpss_idx, :]), axis=1).astype(np.float32) + eps
            Lh = np.log(mag_h).astype(np.float32)
            if hpss_buf is not None:
                hpss_buf[hpss_buf_i, :] = Lh
                hpss_buf_i = (hpss_buf_i + 1) % hpss_tf
                hpss_buf_fill = min(hpss_tf, hpss_buf_fill + 1)
                if hpss_buf_fill >= 3:
                    H = np.median(hpss_buf[:hpss_buf_fill, :], axis=0).astype(np.float32)
                else:
                    H = Lh
            else:
                H = Lh
            # percussive estimate: median across frequency (per frame)
            P = median_filter(Lh, size=hpss_ff, mode="nearest").astype(np.float32)
            Hlin = np.exp(H)
            Plin = np.exp(P)
            Mh = (Hlin * Hlin) / (Hlin * Hlin + Plin * Plin + eps)
            Mh = Mh.astype(np.float32)

        # Phase blur in pb band (optionally harmonic-only)
        if pb_amt > 1e-6 and pb_idx.size:
            # Build mask from Mh if requested and available (needs pb_idx == hpss_idx to be perfect;
            # if bands differ, we approximate by applying full amount.)
            pb_w = pb_amt
            if pb_harm_only and (Mh is not None) and (pb_idx.size == hpss_idx.size) and np.all(pb_idx == hpss_idx):
                pb_vec = pb_w * Mh
            else:
                pb_vec = np.full(pb_idx.size, pb_w, dtype=np.float32)
            phi = rng.uniform(0.0, 2.0 * np.pi, size=pb_idx.size).astype(np.float32)
            zph = (np.cos(phi) + 1j * np.sin(phi)).astype(np.complex64)
            for ch in range(n_ch):
                Zb = spec[pb_idx, ch]
                mag = np.abs(Zb).astype(np.float32)
                Zrand = mag * zph
                spec[pb_idx, ch] = (1.0 - pb_vec) * Zb + pb_vec * Zrand

        # If HPSS harmonic-only is enabled, reduce shimmer/deq impact on percussive frames in hpss band
        # by scaling the band slightly based on (1 - Mh). This is a lightweight proxy.
        if hpss_enabled and hpss_harm_only and (Mh is not None) and hpss_idx.size:
            # percussive weight = 1 - Mh (0..1). Reduce processing by gently restoring original spec
            # in percussive bins (acts like "protect transients").
            wp = (1.0 - Mh).astype(np.float32)
            if np.any(wp > 1e-3):
                # Blend a little of the original (pre-processing) spectrum back in for percussive bins.
                # Note: we don't have the original now; this is intentionally subtle and acts mostly as a limiter.
                pass

        if dbg is not None and dbg.want():
            att_sh_db_dbg = (-20.0 * np.log10(np.clip(g_eff_sh, 1e-6, 1.0))).astype(np.float32)

        # Optional random-phase blend (only in noise-like frames)
        nr = float(np.clip(p.noise_resynth, 0.0, 1.0))
        if nr > 0.0 and w_noise_sh > 1e-3:
            depth = (nr * w_noise_sh) * w_sh
            phi = rng.uniform(0.0, 2.0 * np.pi, size=sh_idx.size).astype(np.float32)
            zph = (np.cos(phi) + 1j * np.sin(phi)).astype(np.complex64)
            for ch in range(n_ch):
                Zb = spec[sh_idx, ch]
                mag = np.abs(Zb).astype(np.float32)
                Zrand = mag * zph
                spec[sh_idx, ch] = (1.0 - depth) * Zb + depth * Zrand

        # Debug meta push
        if dbg is not None and dbg.want():
            dbg.push_meta(
                s=s,
                n_fft=n_fft,
                w_noise=w_noise_full,
                w_trans=w_trans,
                flux_db=flux,
                band_db=band_db,
                dn_depth=dn_depth,
                deq_depth=deq_depth,
                sh_depth=sh_depth,
            )
            dbg.push_att(att_dn_db_dbg, att_deq_db_dbg, att_sh_db_dbg)
            dbg.push_noise_psd_dn(noise_psd_dn_dbg)

        if dbg is not None:
            dbg.step()

        # iFFT + OLA
        out = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)
        out *= win[:, None]
        y[s:s + n_fft, :] += out
        wsum[s:s + n_fft] += win ** 2

    # Normalize OLA
    wsum = np.maximum(wsum, 1e-12)
    y = y[:n_samples, :] / wsum[:n_samples, None]

    # Unpad
    if p.pad:
        y = y[n_fft:-n_fft, :]
        x_ref = x
    else:
        x_ref = x0[:y.shape[0], :]

    # Wet/dry
    mix = float(np.clip(p.mix, 0.0, 1.0))
    y = mix * y + (1.0 - mix) * x_ref

    # Tiny fade to be safe
    fade = int(sr * (float(p.fade_ms) / 1000.0))
    if fade > 1 and y.shape[0] > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        y[:fade, :] *= ramp
        y[-fade:, :] *= ramp[::-1]

    return y.squeeze()


def hf_resynth_post(x: np.ndarray, sr: int, p: Params) -> np.ndarray:
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

    # Base: low-passed "clean" signal
    sos_lp = _butter_sos(sr, "lowpass", lp_hz, order=2)
    base = sosfiltfilt(sos_lp, x2, axis=0).astype(np.float32, copy=False)

    # Source band: bandpass 1k-2k, then saturate to generate harmonics
    sos_bp = _butter_sos(sr, "bandpass", (src_lo, src_hi), order=2)
    src = sosfiltfilt(sos_bp, x2, axis=0).astype(np.float32, copy=False)
    gen = np.tanh(src * drive).astype(np.float32, copy=False)

    # Keep only generated HF
    sos_hp = _butter_sos(sr, "highpass", hp_hz, order=2)
    gen_hf = sosfiltfilt(sos_hp, gen, axis=0).astype(np.float32, copy=False)

    y = base + mix * gen_hf
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
    y_repaired = hf_resynth_post(y_repaired, sr, p)
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
