"""
auto_tune.py

Automatic settings discovery for the deshimmer pipeline.

Two layers:
  1. analyze(...)  -- cheap, deterministic: derive priors directly from the
                      input signal and produce a ready-to-use Params object.
                      Uses median-residual scans, per-bin minimum-statistics
                      noise estimation, persistent-peak detection, and
                      spectral flatness / flux distributions.
  2. refine(...)   -- staged Bayesian optimization (Optuna TPE) over a small
                      set of preview regions, evaluating a composite score
                      that rewards artifact reduction while penalising content
                      damage, musical noise, and transient/harmonic leakage
                      into the diff signal.

The module is UI-agnostic. ui_gradio.py wires both layers into the Gradio app.
"""

from __future__ import annotations

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportPrivateUsage=false

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import stft as _scipy_stft

import master as _m
from deshimmer_api import process_audio


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
@dataclass
class Region:
    t0: float
    dur: float
    label: str = "auto"


@dataclass
class AnalysisReport:
    detected_band: tuple[float, float]
    band_strength_db: float                     # peak residual energy in detected band
    noise_floor_db: float                       # estimated dBFS-ish broadband noise floor
    noise_dynamic_range_db: float               # median - floor, in dB
    resonance_count: int
    resonance_freqs: list[float]
    transient_density_per_s: float
    flatness_p25: float
    flatness_p75: float
    lufs: Optional[float]
    regions: list[Region] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["**Auto analysis report**", ""]
        b0, b1 = self.detected_band
        lines.append(f"- Shimmer band detected: **{b0:.0f} – {b1:.0f} Hz** (strength {self.band_strength_db:+.1f} dB above local median)")
        lines.append(f"- Noise floor: **{self.noise_floor_db:+.1f} dBFS** (dynamic range {self.noise_dynamic_range_db:.1f} dB)")
        if self.resonance_count > 0:
            freq_s = ", ".join(f"{f:.0f} Hz" for f in self.resonance_freqs[:6])
            lines.append(f"- Persistent resonances: **{self.resonance_count}** (top: {freq_s})")
        else:
            lines.append("- Persistent resonances: none detected")
        lines.append(f"- Transient density: {self.transient_density_per_s:.2f}/s")
        lines.append(f"- Spectral flatness range: {self.flatness_p25:.2f} – {self.flatness_p75:.2f}")
        if self.lufs is not None and math.isfinite(self.lufs):
            lines.append(f"- Integrated loudness: {self.lufs:+.1f} LUFS")
        if self.regions:
            reg_s = ", ".join(f"{r.label} @ {r.t0:.1f}s ({r.dur:.1f}s)" for r in self.regions)
            lines.append(f"- Regions used: {reg_s}")
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _to_mono_float(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return np.mean(a, axis=1).astype(np.float32, copy=False)
    raise ValueError("audio must be 1D or 2D")


def _slice(x: np.ndarray, sr: int, t0: float, dur: float) -> np.ndarray:
    if x.ndim == 1:
        n = x.shape[0]
        s0 = int(round(max(0.0, t0) * sr))
        s1 = int(round(max(0.0, t0 + dur) * sr))
        s0 = max(0, min(n, s0))
        s1 = max(s0, min(n, s1))
        return x[s0:s1]
    n = x.shape[0]
    s0 = int(round(max(0.0, t0) * sr))
    s1 = int(round(max(0.0, t0 + dur) * sr))
    s0 = max(0, min(n, s0))
    s1 = max(s0, min(n, s1))
    return x[s0:s1, :]


def _stft_mag(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Mono magnitude STFT. Returns (mag[F, T], freqs[F])."""
    xm = _to_mono_float(x)
    if xm.size < n_fft:
        xm = np.pad(xm, (0, n_fft - xm.size))
    f, _, Z = _scipy_stft(
        xm, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, boundary="zeros", padded=True,
    )
    return np.abs(Z).astype(np.float32) + _EPS, f.astype(np.float32)


def _residual_above_median(L: np.ndarray, k: int) -> np.ndarray:
    """Log-magnitude residual above local frequency median, in dB."""
    if k % 2 == 0:
        k += 1
    L_med = median_filter(L, size=(k, 1), mode="nearest")
    return (L - L_med) * (20.0 / math.log(10.0))


def _flatness(P: np.ndarray) -> np.ndarray:
    """Spectral flatness per frame on a power spectrum."""
    log_P = np.log(P + _EPS)
    return np.exp(np.mean(log_P, axis=0)) / (np.mean(P, axis=0) + _EPS)


def _frame_db(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.mean(mag, axis=0) + _EPS)


def _hpss_components(mag: np.ndarray, t_win: int = 17, f_win: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Median-filter HPSS. Returns (harmonic_mag, percussive_mag)."""
    L = np.log(mag + _EPS)
    H = np.exp(median_filter(L, size=(1, t_win), mode="nearest"))
    P = np.exp(median_filter(L, size=(f_win, 1), mode="nearest"))
    Hp, Pp = H * H, P * P
    Mh = Hp / (Hp + Pp + _EPS)
    Mp = Pp / (Hp + Pp + _EPS)
    return mag * Mh, mag * Mp


def _band_idx(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((freqs >= lo) & (freqs <= hi))[0]


# ---------------------------------------------------------------------------
# Region picking
# ---------------------------------------------------------------------------
def pick_regions(
    x: np.ndarray,
    sr: int,
    *,
    n: int = 3,
    dur: float = 5.0,
    locked: Optional[Region] = None,
) -> list[Region]:
    """
    Pick ~n short windows that together stress-test the settings:
      - quiet : low broadband energy (artifacts most exposed)
      - loud  : high broadband energy (content preservation matters)
      - trans : highest spectral flux (transient leakage check)

    If locked is provided, it is included verbatim and we pick (n-1) more,
    avoiding overlap.
    """
    xm = _to_mono_float(x)
    n_samples = xm.shape[0]
    total_s = n_samples / float(sr) if sr > 0 else 0.0
    dur = float(max(0.5, min(dur, max(0.5, total_s))))
    if total_s <= 0.0:
        return []

    out: list[Region] = []
    if locked is not None:
        t0 = float(np.clip(locked.t0, 0.0, max(0.0, total_s - dur)))
        d = float(np.clip(locked.dur, 0.5, max(0.5, total_s - t0)))
        out.append(Region(t0=t0, dur=d, label="user"))
        n = max(0, n - 1)
    if n <= 0:
        return out

    # Short files: just stride evenly.
    if total_s < dur * (len(out) + n) + 0.5:
        slots = max(1, len(out) + n)
        labels = (["quiet", "loud", "trans"] * 3)[:slots]
        spans = np.linspace(0.0, max(0.0, total_s - dur), slots)
        for i, t0 in enumerate(spans):
            if i < len(out):
                continue
            out.append(Region(t0=float(t0), dur=dur, label=labels[i]))
        return out[: max(slots, 1)]

    # Compute cheap features on a coarse STFT.
    n_fft = 1024
    hop = 1024
    mag, freqs = _stft_mag(xm, sr, n_fft=n_fft, hop=hop)
    P = mag * mag
    band_db = _frame_db(mag)
    nyq = float(freqs[-1]) if freqs.size else float(sr) * 0.5
    midhf = _band_idx(freqs, 2000.0, min(12000.0, nyq))
    midhf_db = _frame_db(mag[midhf, :]) if midhf.size else band_db
    flux = np.zeros_like(band_db)
    flux[1:] = np.maximum(0.0, band_db[1:] - band_db[:-1])

    # Convert frame indices back to seconds.
    times = np.arange(band_db.size) * (hop / float(sr))

    def _frame_to_t0(frame_idx: int) -> float:
        # Center the window on the frame, clamp.
        t_center = float(times[int(np.clip(frame_idx, 0, times.size - 1))])
        t0 = t_center - dur * 0.5
        return float(np.clip(t0, 0.0, max(0.0, total_s - dur)))

    used: list[tuple[float, float]] = [(r.t0, r.t0 + r.dur) for r in out]

    def _overlaps(t0: float) -> bool:
        a, b = t0, t0 + dur
        for u0, u1 in used:
            if a < u1 and u0 < b:
                return True
        return False

    def _add_from_indices(idxs: np.ndarray, label: str) -> bool:
        for i in idxs:
            t0 = _frame_to_t0(int(i))
            if not _overlaps(t0):
                out.append(Region(t0=t0, dur=dur, label=label))
                used.append((t0, t0 + dur))
                return True
        return False

    pool_quiet = np.argsort(midhf_db + 0.5 * band_db)        # smallest first -> quiet
    pool_loud = np.argsort(-band_db)                          # largest first -> loud
    pool_trans = np.argsort(-flux)                            # largest first -> transient

    plans = [("quiet", pool_quiet), ("loud", pool_loud), ("trans", pool_trans)]
    for label, pool in plans:
        if len(out) - (1 if locked else 0) >= n:
            break
        _add_from_indices(pool, label)

    # Fallback if we ran out of non-overlapping candidates.
    while len(out) - (1 if locked else 0) < n:
        t0 = float(np.clip(len(out) * dur, 0.0, max(0.0, total_s - dur)))
        if _overlaps(t0):
            t0 = max(0.0, total_s - dur - 0.01)
        out.append(Region(t0=t0, dur=dur, label="extra"))
        used.append((t0, t0 + dur))

    return out


# ---------------------------------------------------------------------------
# Analysis (Layer 1)
# ---------------------------------------------------------------------------
def _detect_shimmer_band(
    mag_concat: np.ndarray,
    freqs: np.ndarray,
    *,
    scan_lo: float = 3500.0,
    scan_hi: float = 14000.0,
    win_hz: float = 300.0,
    step_hz: float = 100.0,
    thr_db: float = 6.0,
) -> tuple[float, float, float]:
    """
    Slide a narrow window across [scan_lo, scan_hi]; for each position compute
    the fraction of frames containing at least one bin whose log-magnitude
    residual above the local frequency median exceeds thr_db.

    Returns (band_lo, band_hi, peak_density_score_db).
    """
    nyq = float(freqs[-1]) if freqs.size else 0.0
    scan_hi = float(min(scan_hi, max(scan_lo + win_hz, nyq - 100.0)))
    if scan_hi <= scan_lo + win_hz:
        return (5100.0, 7200.0, 0.0)

    centers = np.arange(scan_lo + win_hz / 2.0, scan_hi - win_hz / 2.0, step_hz)
    densities: list[float] = []
    peak_db_per_center: list[float] = []
    for c in centers:
        lo = c - win_hz / 2.0
        hi = c + win_hz / 2.0
        idx = _band_idx(freqs, lo, hi)
        if idx.size < 6:
            densities.append(0.0)
            peak_db_per_center.append(0.0)
            continue
        L = np.log(mag_concat[idx, :] + _EPS)
        resid = _residual_above_median(L, k=9)
        mask = resid > thr_db
        # Density: fraction of frames with at least one above-threshold bin.
        density = float(np.mean(np.any(mask, axis=0)))
        # Strength: median of the above-threshold residuals (only where mask is true).
        if mask.any():
            peak_db = float(np.median(resid[mask]))
        else:
            peak_db = 0.0
        densities.append(density)
        peak_db_per_center.append(peak_db)

    densities = np.asarray(densities, dtype=np.float32)
    peaks = np.asarray(peak_db_per_center, dtype=np.float32)
    if densities.size == 0 or float(densities.max()) <= 0.0:
        return (5100.0, 7200.0, 0.0)

    knee = max(0.05, 0.5 * float(densities.max()))
    above = densities >= knee
    if not above.any():
        # Fallback: pick window with single max density.
        i = int(np.argmax(densities))
        c = float(centers[i])
        return (max(20.0, c - win_hz / 2.0 - 200.0), min(nyq - 50.0, c + win_hz / 2.0 + 200.0), float(peaks[i]))

    # Largest contiguous run of above-knee positions.
    runs: list[tuple[int, int]] = []
    s = None
    for i, v in enumerate(above):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, above.size - 1))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    s, e = runs[0]
    band_lo = float(max(20.0, centers[s] - win_hz / 2.0 - 200.0))
    band_hi = float(min(nyq - 50.0, centers[e] + win_hz / 2.0 + 200.0))
    if band_hi - band_lo < 400.0:
        band_hi = min(nyq - 50.0, band_lo + 800.0)
    peak_strength = float(np.mean(peaks[s:e + 1]))
    return (band_lo, band_hi, peak_strength)


def _detect_resonances(
    mag_concat: np.ndarray,
    freqs: np.ndarray,
    *,
    scan_lo: float = 200.0,
    scan_hi: float = 12000.0,
    persist_thr_db: float = 3.0,
) -> list[tuple[float, float]]:
    """
    Find persistent narrow peaks: bins whose median log-magnitude is well above
    the smoothed-frequency baseline AND whose temporal variance is low.

    Returns a list of (freq_hz, prominence_db), strongest first.
    """
    idx = _band_idx(freqs, scan_lo, scan_hi)
    if idx.size < 16:
        return []
    L = np.log(mag_concat[idx, :] + _EPS)
    L_med_t = np.median(L, axis=1)                         # per-bin temporal median
    L_smoothed = uniform_filter1d(L_med_t, size=31, mode="nearest")
    prominence_db = (L_med_t - L_smoothed) * (20.0 / math.log(10.0))

    # Temporal stability = inverse of frame-to-frame variance, normalised.
    L_std_t = np.std(L, axis=1)
    L_std_norm = L_std_t / (np.median(L_std_t) + _EPS)
    stability = 1.0 / (1.0 + L_std_norm)                   # in (0, 1)

    score = prominence_db * stability
    out: list[tuple[float, float]] = []
    f_band = freqs[idx]
    # Local maxima only.
    for i in range(1, score.size - 1):
        if score[i] > persist_thr_db and score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            out.append((float(f_band[i]), float(prominence_db[i])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def analyze(
    x: np.ndarray,
    sr: int,
    regions: Optional[list[Region]] = None,
    *,
    base_params: Optional[_m.Params] = None,
) -> tuple[_m.Params, _m.MasterParams, AnalysisReport]:
    """
    Run cheap heuristic analysis to derive an initial Params + MasterParams.
    """
    xm = _to_mono_float(x)
    sr = int(sr)
    total_s = xm.shape[0] / float(sr)
    nyq = 0.5 * float(sr)

    if regions is None or len(regions) == 0:
        regions = pick_regions(xm, sr, n=3, dur=min(5.0, max(1.0, total_s / 3.0)))

    # Aggregate STFT magnitudes across regions (concatenate along time axis).
    mags: list[np.ndarray] = []
    flatnesses: list[float] = []
    transient_counts: list[int] = []
    region_durs: list[float] = []
    freqs_ref: Optional[np.ndarray] = None
    for r in regions:
        seg = _slice(xm, sr, r.t0, r.dur)
        if seg.size < 2048:
            continue
        mag, freqs = _stft_mag(seg, sr, n_fft=2048, hop=512)
        if freqs_ref is None:
            freqs_ref = freqs
        elif freqs.shape != freqs_ref.shape:
            continue  # shouldn't happen since we pin n_fft
        mags.append(mag)
        P = mag * mag
        flat = _flatness(P)
        flatnesses.extend(flat.tolist())
        band_db = _frame_db(mag)
        flux = np.zeros_like(band_db)
        flux[1:] = np.maximum(0.0, band_db[1:] - band_db[:-1])
        # Count flux peaks > 6 dB.
        transient_counts.append(int(np.sum(flux > 6.0)))
        region_durs.append(float(r.dur))

    if not mags or freqs_ref is None:
        # Pathological: empty signal. Fall back to defaults.
        p = base_params if base_params is not None else _m.Params()
        mp = _m.MasterParams(enabled=False)
        rep = AnalysisReport(
            detected_band=(p.start_hz, p.end_hz),
            band_strength_db=0.0,
            noise_floor_db=-60.0,
            noise_dynamic_range_db=0.0,
            resonance_count=0,
            resonance_freqs=[],
            transient_density_per_s=0.0,
            flatness_p25=0.25,
            flatness_p75=0.75,
            lufs=None,
            regions=list(regions),
            notes=["Signal too short for full analysis; using defaults."],
        )
        return p, mp, rep

    mag_concat = np.concatenate(mags, axis=1)

    # --- Shimmer band ---
    band_lo, band_hi, band_strength_db = _detect_shimmer_band(mag_concat, freqs_ref)

    # --- Noise floor ---
    dn_idx = _band_idx(freqs_ref, 120.0, min(16000.0, nyq - 100.0))
    if dn_idx.size:
        floor_per_bin = np.percentile(mag_concat[dn_idx, :], 10.0, axis=1)
        median_per_bin = np.median(mag_concat[dn_idx, :], axis=1)
        floor_db = 20.0 * math.log10(float(np.median(floor_per_bin)) + _EPS)
        median_db = 20.0 * math.log10(float(np.median(median_per_bin)) + _EPS)
        dyn_range_db = float(median_db - floor_db)
    else:
        floor_db = -60.0
        dyn_range_db = 30.0

    # --- Resonances ---
    res = _detect_resonances(mag_concat, freqs_ref, scan_hi=min(12000.0, nyq - 100.0))
    res_freqs = [f for f, _ in res[:8]]

    # --- Flatness percentiles ---
    flat_arr = np.asarray(flatnesses, dtype=np.float32)
    flat_p25 = float(np.percentile(flat_arr, 25.0))
    flat_p75 = float(np.percentile(flat_arr, 75.0))

    # --- Transient density ---
    total_dur = max(1e-3, float(sum(region_durs)))
    trans_density = float(sum(transient_counts) / total_dur)

    # --- LUFS (best-effort, full file is fine here) ---
    try:
        lufs_v = _m.measure_lufs(xm, sr)
    except Exception:
        lufs_v = None

    # ----- Map measurements to Params -----
    p = base_params if base_params is not None else _m.Params()
    p = replace(p)  # don't mutate caller's instance

    # Shimmer band.
    p.start_hz = float(band_lo)
    p.end_hz = float(band_hi)
    p.edge_hz = float(max(100.0, 0.05 * (band_hi - band_lo)))

    # Noise gating thresholds from flatness distribution.
    p.flat_start = float(np.clip(flat_p25, 0.05, 0.6))
    p.flat_end = float(np.clip(max(flat_p75, p.flat_start + 0.1), 0.2, 0.95))

    # Shimmer thresholds: tune by detected band strength.
    if band_strength_db >= 9.0:
        p.thr_db = 6.5
        p.slope = 0.8
    elif band_strength_db >= 6.0:
        p.thr_db = 8.0
        p.slope = 0.65
    else:
        p.thr_db = 9.5
        p.slope = 0.5

    # Density thresholds: leave defaults but bias by transient density.
    p.density_lo = 0.02
    p.density_hi = 0.15 if trans_density < 5.0 else 0.20  # busier audio -> back off harder

    # Flux protection.
    p.flux_thr_db = 6.0
    p.flux_range_db = 8.0
    p.noise_resynth = 0.0

    # Denoise: enable proportional to dynamic range; floor sits ~6 dB below estimated floor.
    if dyn_range_db < 12.0:
        # Very compressed dynamic range — denoise probably not needed, can hurt.
        p.denoise = 0.0
    elif dyn_range_db < 25.0:
        p.denoise = 0.25
    else:
        p.denoise = 0.45
    p.dn_floor_db = float(np.clip(floor_db - 6.0, -60.0, -8.0))
    p.dn_start_hz = 120.0
    p.dn_end_hz = float(min(16000.0, nyq - 100.0))
    p.dn_freq_smooth_bins = 5
    p.dn_release_ms = 150.0
    p.dn_attack_ms = 5.0

    # De-resonator: enable only if we found persistent peaks.
    if len(res) >= 1:
        p.deres = float(np.clip(0.3 + 0.1 * len(res), 0.3, 0.8))
        # Threshold from the median prominence we saw.
        med_prom = float(np.median([pr for _, pr in res]))
        p.deq_thr_db = float(np.clip(max(4.0, med_prom * 0.6), 4.0, 9.0))
        p.deq_persist_ms = 700.0
        p.deq_persist_thr_db = 2.5
        # Span deq scan to cover the lowest..highest detected resonance with margin.
        f_min = max(150.0, min(res_freqs) - 500.0)
        f_max = min(nyq - 100.0, max(res_freqs) + 500.0)
        p.deq_start_hz = float(min(f_min, p.deq_start_hz))
        p.deq_end_hz = float(max(f_max, p.deq_end_hz))
    else:
        p.deres = 0.0

    # Mastering: only suggest enabling if input is noticeably quieter than a typical -14 LUFS target.
    mp = _m.MasterParams(enabled=False)
    if lufs_v is not None and math.isfinite(lufs_v) and lufs_v < -18.0:
        mp = _m.MasterParams(
            enabled=True,
            target_lufs=-14.0,
            ceiling_dbtp=-1.0,
            os_factor=4,
            hp_hz=20.0,
        )

    notes: list[str] = []
    if band_strength_db < 4.0:
        notes.append("Shimmer band signal is weak; consider disabling shimmer suppression (mix=0) if there's nothing to fix.")
    if dyn_range_db < 8.0:
        notes.append("Very compressed input — denoise is likely to do more harm than good.")
    if trans_density > 8.0:
        notes.append("Dense transients detected; flux protection kept conservative.")

    rep = AnalysisReport(
        detected_band=(band_lo, band_hi),
        band_strength_db=band_strength_db,
        noise_floor_db=floor_db,
        noise_dynamic_range_db=dyn_range_db,
        resonance_count=len(res),
        resonance_freqs=res_freqs,
        transient_density_per_s=trans_density,
        flatness_p25=flat_p25,
        flatness_p75=flat_p75,
        lufs=lufs_v,
        regions=list(regions),
        notes=notes,
    )
    return p, mp, rep


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class Weights:
    band_reduction: float
    out_of_band_change: float
    musical_noise: float
    transient_leak: float
    harmonic_leak: float
    loudness_loss: float


def weights_from_aggressiveness(agg: float) -> Weights:
    """0.0 = conservative (preserve content), 1.0 = aggressive (max reduction)."""
    a = float(np.clip(agg, 0.0, 1.0))
    return Weights(
        band_reduction=1.0 + 0.8 * a,
        out_of_band_change=0.4 + 1.6 * (1.0 - a),
        musical_noise=0.3 + 1.2 * (1.0 - a),
        transient_leak=0.5 + 1.5 * (1.0 - a),
        harmonic_leak=0.5 + 1.5 * (1.0 - a),
        loudness_loss=0.2 + 0.8 * (1.0 - a),
    )


def _energy_db(mag: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(mag * mag) + _EPS)) + _EPS)


def _band_residual_energy_db(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float, k: int = 9) -> float:
    idx = _band_idx(freqs, lo, hi)
    if idx.size < 4:
        return -120.0
    L = np.log(mag[idx, :] + _EPS)
    resid = _residual_above_median(L, k=k)
    pos = np.maximum(0.0, resid)
    return float(np.mean(pos * pos))


def score_processed(
    x_in: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    *,
    band: tuple[float, float],
    weights: Weights,
) -> dict[str, float]:
    """
    Composite quality score for one processed region.
    Higher = better. Returns full breakdown dict including 'total'.
    """
    xm_in = _to_mono_float(x_in)
    xm_out = _to_mono_float(y_out)
    n = min(xm_in.size, xm_out.size)
    if n < 2048:
        return {"total": 0.0, "_note": "region too short to score"}
    xm_in = xm_in[:n]
    xm_out = xm_out[:n]
    diff = xm_in - xm_out

    mag_in, freqs = _stft_mag(xm_in, sr)
    mag_out, _ = _stft_mag(xm_out, sr)
    mag_diff, _ = _stft_mag(diff, sr)

    # 1) Band residual reduction (positive when artifact peakiness in band drops).
    res_in = _band_residual_energy_db(mag_in, freqs, band[0], band[1])
    res_out = _band_residual_energy_db(mag_out, freqs, band[0], band[1])
    band_reduction = float(10.0 * math.log10((res_in + 1e-6) / (res_out + 1e-6)))
    band_reduction = float(np.clip(band_reduction, -10.0, 30.0))

    # 2) Out-of-band content change (penalty: should be small).
    oob_in_db = _energy_db(np.delete(mag_in, _band_idx(freqs, band[0], band[1]), axis=0))
    oob_out_db = _energy_db(np.delete(mag_out, _band_idx(freqs, band[0], band[1]), axis=0))
    out_of_band_change = float(abs(oob_in_db - oob_out_db))

    # 3) Musical noise: variance of frame-to-frame spectral entropy in low-energy frames.
    band_db_out = _frame_db(mag_out)
    thr = float(np.percentile(band_db_out, 25.0))
    quiet_mask = band_db_out <= thr
    if int(np.sum(quiet_mask)) >= 4:
        Pq = mag_out[:, quiet_mask] ** 2
        Pq = Pq / (np.sum(Pq, axis=0, keepdims=True) + _EPS)
        ent = -np.sum(Pq * np.log(Pq + _EPS), axis=0)
        # Normalize entropy to ~[0, 1] (max entropy = log(F)).
        ent_norm = ent / max(_EPS, math.log(Pq.shape[0]))
        musical_noise = float(np.var(np.diff(ent_norm))) * 100.0  # scale up to comparable magnitudes
    else:
        musical_noise = 0.0

    # 4 + 5) Transient / harmonic leak in the diff (HPSS).
    H_diff, P_diff = _hpss_components(mag_diff)
    diff_total_e = float(np.sum(mag_diff ** 2) + _EPS)
    in_total_e = float(np.sum(mag_in ** 2) + _EPS)
    # Normalise leakage by input total energy so loud diffs don't auto-look bad.
    transient_leak = float(np.sum(P_diff ** 2) / in_total_e)
    harmonic_leak = float(np.sum(H_diff ** 2) / in_total_e)
    # Rescale to nicer magnitudes.
    transient_leak *= 50.0
    harmonic_leak *= 50.0

    # 6) Loudness loss (LUFS preferred, RMS fallback).
    lufs_in = _m.measure_lufs(xm_in, sr)
    lufs_out = _m.measure_lufs(xm_out, sr)
    if lufs_in is not None and lufs_out is not None and math.isfinite(lufs_in) and math.isfinite(lufs_out):
        loud_loss = max(0.0, float(lufs_in) - float(lufs_out) - 1.0)
    else:
        loud_loss = max(0.0, _m.measure_rms_dbfs(xm_in) - _m.measure_rms_dbfs(xm_out) - 1.0)

    total = (
        weights.band_reduction * band_reduction
        - weights.out_of_band_change * out_of_band_change
        - weights.musical_noise * musical_noise
        - weights.transient_leak * transient_leak
        - weights.harmonic_leak * harmonic_leak
        - weights.loudness_loss * loud_loss
    )
    return {
        "total": float(total),
        "band_reduction_db": band_reduction,
        "out_of_band_change_db": out_of_band_change,
        "musical_noise": musical_noise,
        "transient_leak": transient_leak,
        "harmonic_leak": harmonic_leak,
        "loudness_loss_db": loud_loss,
        "diff_total_e": diff_total_e,
    }


# ---------------------------------------------------------------------------
# Refinement (Layer 2)
# ---------------------------------------------------------------------------
def _evaluate_params(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    p: _m.Params,
    weights: Weights,
    band: tuple[float, float],
    refine_dur: float,
) -> float:
    """Evaluate a Params on each region, return the average composite score."""
    mp = _m.MasterParams(enabled=False)  # never run mastering during refinement
    dp = _m.DebugParams(enabled=False)

    # Disable side-effect knobs during refinement: delta_listen would invert audio.
    p_use = replace(p, delta_listen=False, mix=1.0 if p.mix < 1e-6 else p.mix)

    totals: list[float] = []
    for r in regions:
        # Use a centered shorter window inside the analysis region for speed.
        inset = max(0.0, (r.dur - refine_dur) * 0.5)
        seg = _slice(x, sr, r.t0 + inset, min(r.dur, refine_dur))
        if seg.shape[0] < int(0.5 * sr):
            continue
        try:
            y, _info = process_audio(seg, sr, params=p_use, master_params=mp, debug_params=dp)
        except Exception:
            return -1e9
        s = score_processed(seg, y, sr, band=band, weights=weights)
        totals.append(float(s.get("total", 0.0)))
    if not totals:
        return -1e9
    return float(np.mean(totals))


def _make_stage(name: str, p: _m.Params, trial: Any) -> _m.Params:
    """Return a copy of p with the named stage's knobs replaced from trial.suggest_*."""
    if name == "shimmer":
        return replace(
            p,
            thr_db=float(trial.suggest_float("thr_db", 4.0, 12.0)),
            slope=float(trial.suggest_float("slope", 0.3, 1.0)),
            density_lo=float(trial.suggest_float("density_lo", 0.005, 0.06)),
            density_hi=float(trial.suggest_float("density_hi", 0.08, 0.30)),
            noise_resynth=float(trial.suggest_float("noise_resynth", 0.0, 0.5)),
        )
    if name == "denoise":
        return replace(
            p,
            denoise=float(trial.suggest_float("denoise", 0.0, 0.85)),
            dn_floor_db=float(trial.suggest_float("dn_floor_db", -36.0, -8.0)),
            dn_freq_smooth_bins=int(trial.suggest_int("dn_freq_smooth_bins", 1, 9, step=2)),
            dn_release_ms=float(trial.suggest_float("dn_release_ms", 60.0, 300.0)),
            dn_attack_ms=float(trial.suggest_float("dn_attack_ms", 2.0, 20.0)),
        )
    if name == "deres":
        return replace(
            p,
            deres=float(trial.suggest_float("deres", 0.0, 0.9)),
            deq_thr_db=float(trial.suggest_float("deq_thr_db", 3.0, 10.0)),
            deq_slope=float(trial.suggest_float("deq_slope", 0.4, 1.0)),
            deq_persist_ms=float(trial.suggest_float("deq_persist_ms", 300.0, 1500.0)),
            deq_max_att_db=float(trial.suggest_float("deq_max_att_db", 4.0, 14.0)),
        )
    raise ValueError(f"unknown stage: {name}")


def refine(
    x: np.ndarray,
    sr: int,
    *,
    base_params: _m.Params,
    regions: Optional[list[Region]] = None,
    aggressiveness: float = 0.5,
    n_trials_per_stage: int = 12,
    refine_dur: float = 3.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> tuple[_m.Params, dict[str, Any]]:
    """
    Stage A (shimmer) -> Stage B (denoise) -> Stage C (deres). Each stage runs
    an Optuna TPE study; the winner is fed into the next stage.

    Returns (best_params, summary_dict).
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise RuntimeError(
            "optuna is required for refine(). Install with: pip install optuna"
        ) from e

    xm = np.asarray(x)
    sr = int(sr)
    if regions is None or len(regions) == 0:
        regions = pick_regions(xm, sr, n=3, dur=max(refine_dur + 0.5, 5.0))

    weights = weights_from_aggressiveness(aggressiveness)
    band = (float(base_params.start_hz), float(base_params.end_hz))

    p_cur = replace(base_params)
    summary: dict[str, Any] = {
        "aggressiveness": float(aggressiveness),
        "regions": [{"t0": r.t0, "dur": r.dur, "label": r.label} for r in regions],
        "weights": weights.__dict__,
        "stages": [],
    }

    # Decide which stages to run.
    stages: list[str] = ["shimmer"]
    if p_cur.denoise > 1e-3 or aggressiveness >= 0.4:
        stages.append("denoise")
    if p_cur.deres > 1e-3:
        stages.append("deres")

    total_trials = max(1, n_trials_per_stage * len(stages))
    done = 0

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(float(np.clip(frac, 0.0, 1.0)), msg)
            except Exception:
                pass

    _emit(0.0, "Starting refine")

    for stage in stages:
        sampler = TPESampler(seed=42, n_startup_trials=max(3, n_trials_per_stage // 3))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: Any, _stage: str = stage, _p: _m.Params = p_cur) -> float:
            p_try = _make_stage(_stage, _p, trial)
            return _evaluate_params(xm, sr, regions, p_try, weights, band, refine_dur)

        # Manual loop so we can emit progress between trials.
        trial_scores: list[float] = []
        for _ in range(n_trials_per_stage):
            study.optimize(_objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
            trial_scores.append(float(study.best_value))
            done += 1
            _emit(done / total_trials, f"Stage '{stage}': trial {done}/{total_trials}, best={study.best_value:.3f}")

        # Apply best for this stage.
        best_trial = study.best_trial
        p_cur = _make_stage(stage, p_cur, _SnapshotTrial(best_trial.params))
        summary["stages"].append({
            "stage": stage,
            "n_trials": n_trials_per_stage,
            "best_score": float(study.best_value),
            "best_params": dict(best_trial.params),
            "trial_scores": trial_scores,
        })

    # Final evaluation of full params.
    final_score = _evaluate_params(xm, sr, regions, p_cur, weights, band, refine_dur)
    summary["final_score"] = float(final_score)
    _emit(1.0, f"Done. Final score={final_score:.3f}")
    return p_cur, summary


class _SnapshotTrial:
    """Lightweight stand-in for an optuna trial that just returns fixed params.

    Lets us reuse `_make_stage` to apply best-trial values without rerunning
    the suggest_* machinery.
    """

    def __init__(self, params: dict[str, Any]):
        self._p = params

    def suggest_float(self, name: str, lo: float, hi: float, **_: Any) -> float:
        return float(self._p[name])

    def suggest_int(self, name: str, lo: int, hi: int, **_: Any) -> int:
        return int(self._p[name])
