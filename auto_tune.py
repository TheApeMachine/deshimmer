"""
auto_tune.py (Upgraded for High-Fidelity Automated Search)

Automatic settings discovery for the deshimmer pipeline.

Two layers:
  1. analyze(...)  -- cheap, deterministic: derive priors directly from the
                      input signal and produce a ready-to-use Params object.
  2. refine(...)   -- staged multi-objective optimization (Optuna NSGA-II) over
                      preview regions with high-fidelity, content-preservative weights.
"""

from __future__ import annotations

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportPrivateUsage=false

import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import stft as _scipy_stft

import master as _m
from master import align_lr_delay, balance_lr_spectrum, dynamic_spectral_carver
from tonal_repair import TonalRepair
from deshimmer_api import process_audio, preview_context_seconds, slice_with_context


OBJECTIVE_NAMES = (
    "artifact_reduction", "music_preservation", "diff_preservation",
    "stereo_preservation", "loudness_tilt_preservation", "tonal_coherence",
)


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
    band_db_diff: float = 0.0
    spectral_entropy: float = 0.0
    detected_whines: list[float] = field(default_factory=list)
    measured_noise_floor_db: float = -60.0
    notes: list[str] = field(default_factory=list)
    tonal_regions: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["**Auto analysis report**", ""]
        b0, b1 = self.detected_band
        lines.append(f"- Shimmer band detected: **{b0:.0f} – {b1:.0f} Hz**")
        lines.append(f"- Estimated noise floor: **{self.measured_noise_floor_db:+.1f} dBFS**")
        if self.detected_whines:
            whines_s = ", ".join(f"{w:.0f} Hz" for w in self.detected_whines[:6])
            lines.append(f"- Identified stationary whines: **{len(self.detected_whines)}** ({whines_s})")
        else:
            lines.append("- Identified stationary whines: none detected")
        for n in self.notes:
            lines.append(f"- {n}")

        for region in self.tonal_regions:
            disagreement = region["p95_disagreement_cents"]
            measured = f"{disagreement:.2f} cents p95 disagreement" if disagreement is not None else "insufficient family evidence"
            lines.append(f"- Tonal region **{region['label']}**: {region['families']} families, {measured}.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_EPS = 1e-12


def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=1).astype(np.float32, copy=False)
    raise ValueError("audio must be 1D or 2D")


def _slice(x: np.ndarray, sr: int, t0: float, dur: float) -> np.ndarray:
    if x.ndim == 1:
        n = x.shape[0]
        s0 = round(max(0.0, t0) * sr)
        s1 = round(max(0.0, t0 + dur) * sr)
        s0 = max(0, min(n, s0))
        s1 = max(s0, min(n, s1))
        return x[s0:s1]
    n = x.shape[0]
    s0 = round(max(0.0, t0) * sr)
    s1 = round(max(0.0, t0 + dur) * sr)
    s0 = max(0, min(n, s0))
    s1 = max(s0, min(n, s1))
    return x[s0:s1, :]


def _stft_mag(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    xm = _to_mono_float(x)
    if xm.size < n_fft:
        xm = np.pad(xm, (0, n_fft - xm.size))
    f, _, Z = _scipy_stft(
        xm, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, boundary="zeros", padded=True,
    )
    return np.abs(Z).astype(np.float32) + _EPS, f.astype(np.float32)


def _residual_above_median(L: np.ndarray, k: int) -> np.ndarray:
    if k % 2 == 0:
        k += 1
    L_med = median_filter(L, size=(k, 1), mode="nearest")
    return (L - L_med) * (20.0 / math.log(10.0))


def _flatness(P: np.ndarray) -> np.ndarray:
    log_P = np.log(P + _EPS)
    return np.exp(np.mean(log_P, axis=0)) / (np.mean(P, axis=0) + _EPS)


def _frame_db(mag: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.mean(mag, axis=0) + _EPS)


def _hpss_components(mag: np.ndarray, t_win: int = 17, f_win: int = 17) -> tuple[np.ndarray, np.ndarray]:
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
    n: int = 5,
    dur: float = 5.0,
    locked: Optional[Region] = None,
) -> list[Region]:
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

    if total_s < dur * (len(out) + n) + 0.5:
        slots = max(1, len(out) + n)
        labels = (["quiet", "loud", "trans"] * 3)[:slots]
        spans = np.linspace(0.0, max(0.0, total_s - dur), slots)
        for i, t0 in enumerate(spans):
            if i < len(out):
                continue
            out.append(Region(t0=float(t0), dur=dur, label=labels[i]))
        return out[: max(slots, 1)]

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

    times = np.arange(band_db.size) * (hop / float(sr))

    def _frame_to_t0(frame_idx: int) -> float:
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

    pool_quiet = np.argsort(midhf_db + 0.5 * band_db)
    pool_loud = np.argsort(-band_db)
    pool_trans = np.argsort(-flux)

    plans = [("quiet", pool_quiet), ("loud", pool_loud), ("trans", pool_trans)]
    for label, pool in plans:
        if len(out) - (1 if locked else 0) >= n:
            break
        _add_from_indices(pool, label)

    # Tail/sustain
    if len(out) - (1 if locked else 0) < n and midhf.size:
        hf_flat = _flatness(P[midhf, :])
        tail_score = hf_flat - 0.35 * flux - 0.15 * (band_db - np.min(band_db))
        pool_tail = np.argsort(-tail_score)
        if len(out) - (1 if locked else 0) < n:
            _add_from_indices(pool_tail, "tail")

    # Harmonic lead preservation risk
    if len(out) - (1 if locked else 0) < n:
        vocal_idx = _band_idx(freqs, 1000.0, min(6000.0, nyq))
        if vocal_idx.size >= 4:
            vocal_db = _frame_db(mag[vocal_idx, :])
            H_v, _P_v = _hpss_components(mag[vocal_idx, :], t_win=17, f_win=9)
            harm_ratio = np.sum(H_v ** 2, axis=0) / (np.sum(mag[vocal_idx, :] ** 2, axis=0) + _EPS)
            vocal_score = harm_ratio * np.clip(vocal_db - np.percentile(band_db, 20), 0.0, 30.0)
            pool_vocal = np.argsort(-vocal_score)
            if len(out) - (1 if locked else 0) < n:
                _add_from_indices(pool_vocal, "vocal")

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
        density = float(np.mean(np.any(mask, axis=0)))
        if mask.any():
            peak_db = float(np.median(resid[mask]))
        else:
            peak_db = 0.0
        densities.append(density)
        peak_db_per_center.append(peak_db)

    dens_arr = np.asarray(densities, dtype=np.float32)
    peaks_arr = np.asarray(peak_db_per_center, dtype=np.float32)
    if dens_arr.size == 0 or float(dens_arr.max()) <= 0.0:
        return (5100.0, 7200.0, 0.0)

    knee = max(0.05, 0.5 * float(dens_arr.max()))
    above = dens_arr >= knee
    if not above.any():
        i = int(np.argmax(dens_arr))
        c = float(centers[i])
        return (max(20.0, c - win_hz / 2.0 - 200.0), min(nyq - 50.0, c + win_hz / 2.0 + 200.0), float(peaks_arr[i]))

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
    peak_strength = float(np.mean(peaks_arr[s:e + 1]))
    return (band_lo, band_hi, peak_strength)


def _detect_resonances(
    mag_concat: np.ndarray,
    freqs: np.ndarray,
    *,
    scan_lo: float = 200.0,
    scan_hi: float = 12000.0,
    persist_thr_db: float = 3.0,
) -> list[tuple[float, float]]:
    idx = _band_idx(freqs, scan_lo, scan_hi)
    if idx.size < 16:
        return []
    L = np.log(mag_concat[idx, :] + _EPS)
    L_med_t = np.median(L, axis=1)
    L_smoothed = uniform_filter1d(L_med_t, size=31, mode="nearest")
    prominence_db = (L_med_t - L_smoothed) * (20.0 / math.log(10.0))

    L_std_t = np.std(L, axis=1)
    L_std_norm = L_std_t / (np.median(L_std_t) + _EPS)
    stability = 1.0 / (1.0 + L_std_norm)

    score = prominence_db * stability
    out: list[tuple[float, float]] = []
    f_band = freqs[idx]
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
    xm = _to_mono_float(x)
    sr = sr
    total_s = xm.shape[0] / float(sr)
    nyq = 0.5 * float(sr)

    if regions is None or len(regions) == 0:
        regions = pick_regions(xm, sr, n=5, dur=min(5.0, max(1.0, total_s / 3.0)))

    mags: list[np.ndarray] = []
    flatnesses: list[float] = []
    freqs_ref: Optional[np.ndarray] = None
    for r in regions:
        seg = _slice(xm, sr, r.t0, r.dur)
        if seg.size < 2048:
            continue
        mag, freqs = _stft_mag(seg, sr, n_fft=2048, hop=512)
        if freqs_ref is None:
            freqs_ref = freqs
        elif freqs.shape != freqs_ref.shape:
            continue
        mags.append(mag)
        P = mag * mag
        flat = _flatness(P)
        flatnesses.extend(flat.tolist())

    if not mags or freqs_ref is None:
        p = replace(base_params if base_params is not None else _m.Params(),
                    tonal_repair=_m.Params().tonal_repair)
        mp = _m.MasterParams(enabled=False)
        rep = AnalysisReport(
            detected_band=(p.start_hz, p.end_hz),
            notes=["Signal too short for full analysis; using defaults."],
        )
        return p, mp, rep

    mag_concat = np.concatenate(mags, axis=1)

    band_lo, band_hi, band_strength_db = _detect_shimmer_band(mag_concat, freqs_ref)

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

    res = _detect_resonances(mag_concat, freqs_ref, scan_hi=min(12000.0, nyq - 100.0))
    res_freqs = [f for f, _ in res[:8]]

    flat_arr = np.asarray(flatnesses, dtype=np.float32)
    flat_p25 = float(np.percentile(flat_arr, 25.0))
    flat_p75 = float(np.percentile(flat_arr, 75.0))

    p = base_params if base_params is not None else _m.Params()
    p = replace(p, tonal_repair=_m.Params().tonal_repair)

    p.start_hz = float(band_lo)
    p.end_hz = float(band_hi)
    p.edge_hz = float(max(100.0, 0.05 * (band_hi - band_lo)))

    p.flat_start = float(np.clip(flat_p25, 0.05, 0.6))
    p.flat_end = float(np.clip(max(flat_p75, p.flat_start + 0.1), 0.2, 0.95))

    if band_strength_db >= 9.0:
        p.thr_db = 6.5
        p.slope = 0.8
    else:
        p.thr_db = 8.0
        p.slope = 0.6

    p.density_lo = 0.02
    p.density_hi = 0.15

    p.denoise = 0.25 if dyn_range_db < 25.0 else 0.0
    p.dn_floor_db = -18.0

    if res_freqs:
        p.deres = 0.4
        p.deq_thr_db = 6.0
    else:
        p.deres = 0.0

    p.swish_start_hz = float(max(2000.0, band_lo - 500.0))
    p.swish_end_hz = float(min(nyq - 100.0, band_hi + 500.0))
    if flat_p75 < 0.12:
        p.swish_repair = 0.35
        p.hf_decorrelate = 0.20

    mp = _m.MasterParams(enabled=False)

    tonal_regions = [
        {"label": region.label, **TonalRepair().analyze(_slice(np.asarray(x), sr, region.t0, region.dur), sr).metrics()}
        for region in regions
    ]
    notes: list[str] = ["Self-referenced tonal repair is active; Optimize includes its coherence objective and stage."]
    if flat_p75 < 0.12:
        notes.append("Highly tonal material: configured Swish Phase Repair and high-frequency stereo decorrelation.")

    rep = AnalysisReport(
        detected_band=(band_lo, band_hi),
        band_db_diff=band_strength_db,
        detected_whines=res_freqs,
        measured_noise_floor_db=floor_db,
        notes=notes,
        tonal_regions=tonal_regions,
    )
    return p, mp, rep


# ---------------------------------------------------------------------------
# High-Fidelity Scoring
# ---------------------------------------------------------------------------
@dataclass
class Weights:
    band_reduction: float
    swish_reduction: float
    out_of_band_change: float
    musical_noise: float
    transient_leak: float
    harmonic_leak: float
    loudness_loss: float
    band_energy_loss: float
    spectral_tilt: float
    centroid_shift: float
    diff_onset_corr: float
    stereo_width: float
    tonal_coherence: float = 1.0

    def objectives(self, metrics: dict[str, float]) -> tuple[float, ...]:
        """Partition weighted utility into six objectives, counting each term once."""
        return (
            self.band_reduction * metrics["band_reduction_db"]
            + self.swish_reduction * metrics["swish_reduction"],
            -(self.out_of_band_change * metrics["out_of_band_change_db"]
              + self.musical_noise * metrics["musical_noise"]
              + self.band_energy_loss * metrics["band_energy_loss_db"]
              + self.centroid_shift * metrics["centroid_shift"]),
            -(self.transient_leak * metrics["transient_leak"]
              + self.harmonic_leak * metrics["harmonic_leak"]
              + self.diff_onset_corr * metrics["diff_onset_corr"]),
            -self.stereo_width * metrics["stereo_width_delta"],
            -(self.loudness_loss * metrics["loudness_loss_db"]
              + self.spectral_tilt * metrics["spectral_tilt_delta"]),
            self.tonal_coherence * metrics["tonal_coherence_gain"],
        )


def flat_weights() -> Weights:
    return Weights(
        band_reduction=1.0,
        swish_reduction=1.0,
        out_of_band_change=1.0,
        musical_noise=1.0,
        transient_leak=1.0,
        harmonic_leak=1.0,
        loudness_loss=1.0,
        band_energy_loss=1.0,
        spectral_tilt=1.0,
        centroid_shift=1.0,
        diff_onset_corr=1.0,
        stereo_width=1.0,
    )


def weights_from_aggressiveness(agg: float) -> Weights:
    """
    Fidelity-First Scoring Curve:
    - Scales artifact reduction positively.
    - Prevents the music-damage penalties from dropping to zero at high aggressiveness.
    - Keeps transient leaks, musical noise, and stereo image deviations heavily penalized.
    """
    a = float(np.clip(agg, 0.0, 1.0))
    return Weights(
        band_reduction=1.0 + 1.2 * a,
        swish_reduction=0.9 + 1.5 * a,
        # Even at agg=1.0, keep content-preservation penalties highly active
        out_of_band_change=1.2 + 0.8 * (1.0 - a),
        musical_noise=1.5 + 0.5 * (1.0 - a),
        transient_leak=1.8 + 0.6 * (1.0 - a),  # strict transient preservation
        harmonic_leak=1.4 + 0.6 * (1.0 - a),
        loudness_loss=0.8 + 0.4 * (1.0 - a),
        band_energy_loss=1.0 + 0.6 * (1.0 - a),
        spectral_tilt=0.8 + 0.6 * (1.0 - a),
        centroid_shift=0.7 + 0.5 * (1.0 - a),
        diff_onset_corr=1.5 + 0.9 * (1.0 - a),
        stereo_width=1.6 + 0.8 * (1.0 - a),  # strict soundstage preservation
        tonal_coherence=1.0 + 1.2 * a,  # same utility curve as magnitude artifact reduction
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


def _band_energy_db(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    idx = _band_idx(freqs, lo, hi)
    if idx.size == 0:
        return -120.0
    return _energy_db(mag[idx, :])


def _spectral_centroid_hz(mag: np.ndarray, freqs: np.ndarray) -> float:
    P = mag.astype(np.float64) ** 2
    num = float(np.sum(freqs[:, None] * P))
    den = float(np.sum(P) + _EPS)
    return num / den


def _spectral_tilt_db(mag: np.ndarray, freqs: np.ndarray, pivot_hz: float = 3000.0) -> float:
    idx = np.where(freqs >= pivot_hz)[0]
    if idx.size < 4:
        return 0.0
    lf = np.log(freqs[idx].astype(np.float64) + _EPS)
    lm = np.log(np.mean(mag[idx, :], axis=1).astype(np.float64) + _EPS)
    if lf.size < 2:
        return 0.0
    slope = float(np.polyfit(lf, lm, 1)[0])
    return slope * (20.0 / math.log(10.0))


def _onset_envelope(x: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    xm = _to_mono_float(x)
    n_fft = 2048
    if xm.size < n_fft:
        return np.zeros(1, dtype=np.float32)
    _, _, Z = _scipy_stft(
        xm, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        nfft=n_fft, boundary="zeros", padded=True,
    )
    mag = np.abs(Z).astype(np.float32)
    band_db = _frame_db(mag)
    env = np.zeros_like(band_db)
    env[1:] = np.maximum(0.0, band_db[1:] - band_db[:-1])
    return env.astype(np.float32, copy=False)


def _stereo_width_ratio(x: np.ndarray) -> float:
    x2 = np.asarray(x, dtype=np.float32)
    if x2.ndim == 1 or x2.shape[1] < 2:
        return 0.0
    mid = 0.5 * (x2[:, 0] + x2[:, 1])
    side = 0.5 * (x2[:, 0] - x2[:, 1])
    r_mid = float(np.sqrt(np.mean(mid * mid) + _EPS))
    r_side = float(np.sqrt(np.mean(side * side) + _EPS))
    return r_side / (r_mid + _EPS)


def score_processed(
    x_in: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    *,
    band: tuple[float, float],
    weights: Weights,
    swish_band: tuple[float, float],
) -> dict[str, float]:
    """Measure shimmer and swish in their respective bands and apply weighted utility."""
    if sr <= 0:
        raise ValueError("scoring requires a positive sample rate")

    xm_in = _to_mono_float(x_in)
    xm_out = _to_mono_float(y_out)

    if _m._as_2d(np.asarray(x_in)).shape != _m._as_2d(np.asarray(y_out)).shape:
        raise ValueError("scoring requires matching input and output shapes")

    if not np.all(np.isfinite(x_in)) or not np.all(np.isfinite(y_out)):
        raise ValueError("scoring requires finite audio samples")

    n = xm_in.size

    if n < 2048:
        raise ValueError("scoring requires at least 2048 samples")
    diff = xm_in - xm_out

    mag_in, freqs = _stft_mag(xm_in, sr)
    mag_out, _ = _stft_mag(xm_out, sr)
    mag_diff, _ = _stft_mag(diff, sr)

    for name, bounds in (("shimmer", band), ("swish", swish_band)):
        if not np.all(np.isfinite(bounds)) or _band_idx(freqs, *bounds).size < 4:
            raise ValueError(f"{name} scoring band requires at least four FFT bins below Nyquist")

    res_in = _band_residual_energy_db(mag_in, freqs, band[0], band[1])
    res_out = _band_residual_energy_db(mag_out, freqs, band[0], band[1])
    band_reduction = float(10.0 * math.log10((res_in + 1e-6) / (res_out + 1e-6)))
    band_reduction = float(np.clip(band_reduction, -10.0, 30.0))

    inst_in = _m.measure_phase_instability(xm_in, sr, *swish_band)
    inst_out = _m.measure_phase_instability(xm_out, sr, *swish_band)
    inst_denom = max(inst_in, 0.02)
    swish_reduction = float(np.clip((inst_in - inst_out) / inst_denom, -0.5, 1.0)) * 12.0

    outside_band = (freqs < band[0]) | (freqs > band[1])
    # A full-spectrum treatment has no out-of-band bins to change.
    out_of_band_change = 0.0

    if np.any(outside_band):
        out_of_band_change = abs(_energy_db(mag_in[outside_band]) - _energy_db(mag_out[outside_band]))

    band_db_in = _frame_db(mag_in)
    thr = float(np.percentile(band_db_in, 25.0))
    quiet_mask = band_db_in <= thr
    if int(np.sum(quiet_mask)) >= 4:
        # Penalize added musical noise in the same input-defined quiet frames.
        Pq = np.stack((mag_in[:, quiet_mask], mag_out[:, quiet_mask]), axis=2) ** 2
        Pq = Pq / (np.sum(Pq, axis=0, keepdims=True) + _EPS)
        ent = -np.sum(Pq * np.log(Pq + _EPS), axis=0)
        ent_norm = ent / max(_EPS, math.log(Pq.shape[0]))
        variation = np.var(np.diff(ent_norm, axis=0), axis=0)
        musical_noise = float(max(0.0, variation[1] - variation[0])) * 100.0
    else:
        musical_noise = 0.0

    H_diff, P_diff = _hpss_components(mag_diff)
    diff_total_e = float(np.sum(mag_diff ** 2) + _EPS)
    in_total_e = float(np.sum(mag_in ** 2) + _EPS)
    transient_leak = float(np.sum(P_diff ** 2) / in_total_e) * 50.0
    harmonic_leak = float(np.sum(H_diff ** 2) / in_total_e) * 50.0

    lufs_in = _m.measure_lufs(xm_in, sr)
    lufs_out = _m.measure_lufs(xm_out, sr)
    if lufs_in is not None and lufs_out is not None and math.isfinite(lufs_in) and math.isfinite(lufs_out):
        loud_loss = max(0.0, float(lufs_in) - float(lufs_out) - 1.0)
    else:
        loud_loss = max(0.0, _m.measure_rms_dbfs(xm_in) - _m.measure_rms_dbfs(xm_out) - 1.0)

    band_in_db = _band_energy_db(mag_in, freqs, band[0], band[1])
    band_out_db = _band_energy_db(mag_out, freqs, band[0], band[1])
    band_energy_loss = float(max(0.0, band_in_db - band_out_db))

    tilt_in = _spectral_tilt_db(mag_in, freqs, pivot_hz=3000.0)
    tilt_out = _spectral_tilt_db(mag_out, freqs, pivot_hz=3000.0)
    spectral_tilt_delta = float(abs(tilt_out - tilt_in))

    cent_in = _spectral_centroid_hz(mag_in, freqs)
    cent_out = _spectral_centroid_hz(mag_out, freqs)
    centroid_shift = float(max(0.0, cent_in - cent_out) / max(cent_in, 1.0)) * 1000.0

    env_in = _onset_envelope(xm_in, sr)
    env_diff = _onset_envelope(diff, sr)
    mlen = min(env_in.size, env_diff.size)
    if mlen >= 8 and np.std(env_in[:mlen]) > 0.0 and np.std(env_diff[:mlen]) > 0.0:
        c = np.corrcoef(env_in[:mlen], env_diff[:mlen])[0, 1]
        diff_onset_corr = float(max(0.0, c)) if math.isfinite(float(c)) else 0.0
    else:
        diff_onset_corr = 0.0

    w_in = _stereo_width_ratio(x_in)
    w_out = _stereo_width_ratio(y_out)
    stereo_width_delta = float(abs(w_out - w_in)) * 50.0

    tonal = TonalRepair()
    input_field = tonal.analyze(x_in, sr)
    output_field = tonal.analyze(y_out, sr)

    metrics = {
        # Fractional, energy-weighted coherence improvement expressed in percent.
        "tonal_coherence_gain": 100.0 * input_field.compare(output_field),
        "tonal_families_in": float(len(input_field.families)),
        "band_reduction_db": band_reduction,
        "swish_reduction": swish_reduction,
        "phase_instability_in": inst_in,
        "phase_instability_out": inst_out,
        "out_of_band_change_db": out_of_band_change,
        "musical_noise": musical_noise,
        "transient_leak": transient_leak,
        "harmonic_leak": harmonic_leak,
        "loudness_loss_db": loud_loss,
        "band_energy_loss_db": band_energy_loss,
        "spectral_tilt_delta": spectral_tilt_delta,
        "centroid_shift": centroid_shift,
        "diff_onset_corr": diff_onset_corr,
        "stereo_width_delta": stereo_width_delta,
        "diff_total_e": diff_total_e,
    }
    objectives = weights.objectives(metrics)
    metrics.update({
        "total": float(sum(objectives)),
        "artifact_reduction": objectives[0],
        "music_damage": -objectives[1],
        "diff_music_leakage": -objectives[2],
        "stereo_damage": -objectives[3],
        "loudness_or_tilt_damage": -objectives[4],
    })
    return metrics


def score_objectives(
    x_in: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    *,
    band: tuple[float, float],
    weights: Weights,
    swish_band: tuple[float, float],
) -> tuple[float, ...]:
    metrics = score_processed(
        x_in, y_out, sr, band=band, swish_band=swish_band, weights=weights,
    )
    return weights.objectives(metrics)


MATH_FAILURES = (ValueError, FloatingPointError, RuntimeWarning, ArithmeticError, np.linalg.LinAlgError)


class EvaluationFailureLog:
    def __init__(
        self,
        debug_dir: str,
        *,
        planned_trials_per_stage: int,
        max_consecutive: int = 5,
        max_failure_rate: float = 0.20,
    ):
        self.debug_dir = debug_dir
        self.path = os.path.join(debug_dir, "failed_trials.json")
        self.planned_trials_per_stage = max(1, planned_trials_per_stage)
        self.max_consecutive = max(1, max_consecutive)
        self.max_failure_rate = float(max(0.0, max_failure_rate))
        self.consecutive = 0
        self.total_failures = 0
        self.stage_attempts: dict[str, int] = {}
        self.stage_failures: dict[str, int] = {}
        self._events: list[dict[str, Any]] = []

    def begin_trial(self, stage: str) -> None:
        self.stage_attempts[stage] = self.stage_attempts.get(stage, 0) + 1

    def success(self, stage: str) -> None:
        self.consecutive = 0

    def record(
        self,
        *,
        stage: str,
        trial_number: int | None,
        region: Region,
        params: _m.Params,
        exc: BaseException,
    ) -> bool:
        self.consecutive += 1
        self.total_failures += 1
        self.stage_failures[stage] = self.stage_failures.get(stage, 0) + 1
        event = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage": stage,
            "trial_number": trial_number,
            "region": asdict(region),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "params": asdict(params),
        }
        self._events.append(event)
        os.makedirs(self.debug_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._events, f, indent=2, sort_keys=True)

        print(
            f"[auto_tune] stage={stage} trial={trial_number} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        allowed_stage_failures = math.floor(self.max_failure_rate * self.planned_trials_per_stage)
        return self.consecutive >= self.max_consecutive or (
            self.stage_failures.get(stage, 0) > max(1, allowed_stage_failures)
        )


def build_synthetic_transition(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    *,
    transition_dur: float = 1.0,
) -> Optional[tuple[np.ndarray, int]]:
    """
    Synthetically splices the tail of a loud region into the head of a quiet region
    (e.g., 1s loud + 1s quiet) to create an artificial dynamic boundary seam.
    Returns (synthetic_audio, seam_sample_idx) or None if suitable contrasting regions
    cannot be identified.
    """
    if len(regions) < 2:
        return None

    xm = np.asarray(x)
    total_samples = xm.shape[0]

    # Look for labeled loud and quiet regions first
    loud_reg = next((r for r in regions if r.label == "loud"), None)
    quiet_reg = next((r for r in regions if r.label == "quiet"), None)

    # Fallback to computing RMS of regions if explicit labels not found
    if loud_reg is None or quiet_reg is None:
        rms_vals = []
        for r in regions:
            s0 = int(r.t0 * sr)
            s1 = min(total_samples, s0 + int(r.dur * sr))
            if s1 - s0 >= 1024:
                seg = xm[s0:s1]
                rms_vals.append((float(np.sqrt(np.mean(seg ** 2) + _EPS)), r))
        if len(rms_vals) < 2:
            return None
        rms_vals.sort(key=lambda t: t[0])
        quiet_reg = rms_vals[0][1]
        loud_reg = rms_vals[-1][1]

    if loud_reg == quiet_reg:
        return None

    dur_samples = max(2048, int(transition_dur * sr))

    loud_s0 = int(loud_reg.t0 * sr)
    loud_s1 = min(total_samples, loud_s0 + int(loud_reg.dur * sr))
    if loud_s1 - loud_s0 < dur_samples:
        loud_tail = xm[loud_s0:loud_s1]
    else:
        loud_tail = xm[loud_s1 - dur_samples:loud_s1]

    quiet_s0 = int(quiet_reg.t0 * sr)
    quiet_s1 = min(total_samples, quiet_s0 + int(quiet_reg.dur * sr))
    if quiet_s1 - quiet_s0 < dur_samples:
        quiet_head = xm[quiet_s0:quiet_s1]
    else:
        quiet_head = xm[quiet_s0:quiet_s0 + dur_samples]

    seam_idx = loud_tail.shape[0]
    synthetic = np.concatenate([loud_tail, quiet_head], axis=0)
    return synthetic, seam_idx


def score_transition_boundary(
    x_trans: np.ndarray,
    y_trans: np.ndarray,
    sr: int,
    seam_idx: int,
    *,
    window_ms: float = 150.0,
) -> dict[str, float]:
    """
    Evaluates dynamic transition consistency across the synthetic seam:
    - boundary_step_error: unnatural gating or squashing across the loud->quiet seam.
    - boundary_pumping: gain fluctuation / hysteresis in the immediate quiet recovery frames.
    - boundary_stereo_delta: stereo soundstage deviation across the transition boundary.
    """
    x2 = np.asarray(x_trans, dtype=np.float32)
    y2 = np.asarray(y_trans, dtype=np.float32)

    win_samples = max(512, int(sr * window_ms / 1000.0))
    pre_start = max(0, seam_idx - win_samples)
    post_end = min(x2.shape[0], seam_idx + win_samples)

    if seam_idx - pre_start < 256 or post_end - seam_idx < 256:
        return {
            "boundary_step_error": 0.0,
            "boundary_pumping": 0.0,
            "boundary_stereo_delta": 0.0,
        }

    x_pre = x2[pre_start:seam_idx]
    x_post = x2[seam_idx:post_end]
    y_pre = y2[pre_start:seam_idx]
    y_post = y2[seam_idx:post_end]

    # 1. Loudness step preservation
    rms_x_pre = float(np.sqrt(np.mean(x_pre ** 2) + _EPS))
    rms_x_post = float(np.sqrt(np.mean(x_post ** 2) + _EPS))
    rms_y_pre = float(np.sqrt(np.mean(y_pre ** 2) + _EPS))
    rms_y_post = float(np.sqrt(np.mean(y_post ** 2) + _EPS))

    step_x_db = 20.0 * math.log10(rms_x_pre / rms_x_post)
    step_y_db = 20.0 * math.log10(rms_y_pre / rms_y_post)
    boundary_step_error = float(abs(step_y_db - step_x_db))

    # 2. Pumping in the initial recovery sub-blocks of the quiet section
    post_len = post_end - seam_idx
    sub_len = max(1, post_len // 3)
    gains = []
    for k in range(3):
        k0 = seam_idx + k * sub_len
        k1 = min(post_end, k0 + sub_len)
        rk_x = float(np.sqrt(np.mean(x2[k0:k1] ** 2) + _EPS))
        rk_y = float(np.sqrt(np.mean(y2[k0:k1] ** 2) + _EPS))
        gains.append(rk_y / rk_x)

    gain_variance = float(np.var(gains))
    gain_gating = float(max(0.0, gains[-1] - gains[0]))
    boundary_pumping = float(gain_variance * 100.0 + gain_gating * 10.0)

    # 3. Stereo width jump across seam
    w_x_post = _stereo_width_ratio(x_post)
    w_y_post = _stereo_width_ratio(y_post)
    boundary_stereo_delta = float(abs(w_y_post - w_x_post)) * 50.0

    return {
        "boundary_step_error": boundary_step_error,
        "boundary_pumping": boundary_pumping,
        "boundary_stereo_delta": boundary_stereo_delta,
    }


def _evaluate_params_multi(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    p: _m.Params,
    weights: Weights,
    band: tuple[float, float],
    refine_dur: float,
    *,
    failure_log: EvaluationFailureLog | None = None,
    stage: str = "refine",
    trial_number: int | None = None,
    transition_aware: bool = False,
    variance_weight: float = 0.15,
) -> tuple[float, ...]:
    mp = _m.MasterParams(enabled=False)
    dp = _m.DebugParams(enabled=False)
    p_use = replace(p, delta_listen=False)
    context_s = preview_context_seconds(p_use)
    objs_list: list[tuple[float, ...]] = []
    count = 0
    for r in regions:
        inset = max(0.0, (r.dur - refine_dur) * 0.5)
        seg_t0 = r.t0 + inset
        seg_dur = min(r.dur, refine_dur)
        if seg_dur <= 0.0:
            raise ValueError(f"region '{r.label}' has no audio to score")
        x_ctx, target_s0, target_s1, ctx_s0 = slice_with_context(
            x, sr, seg_t0, seg_dur, context_s=context_s, params=p_use,
        )
        if target_s1 - target_s0 < max(2048, int(0.5 * sr)):
            raise ValueError(f"region '{r.label}' requires at least 0.5 seconds and 2048 samples")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
                    y_ctx, _info = process_audio(x_ctx, sr, params=p_use, master_params=mp, debug_params=dp)
                    trim0 = target_s0 - ctx_s0
                    trim1 = target_s1 - ctx_s0
                    y = np.asarray(y_ctx)[trim0:trim1]
                    seg = x[target_s0:target_s1]

                    if seg.ndim == 2:
                        y = _m._as_2d(y)

                    objectives = score_objectives(
                        seg, y, sr, band=band,
                        swish_band=(p_use.swish_start_hz, p_use.swish_end_hz),
                        weights=weights,
                    )

                    if not np.all(np.isfinite(objectives)):
                        raise ValueError("scoring produced non-finite objectives")
        except MATH_FAILURES as exc:
            if failure_log is not None:
                fatal = failure_log.record(
                    stage=stage,
                    trial_number=trial_number,
                    region=r,
                    params=p_use,
                    exc=exc,
                )
                if fatal:
                    raise RuntimeError(
                        "Optimization failed due to persistent mathematical instability. "
                        f"See {failure_log.path}"
                    ) from exc
                import optuna

                raise optuna.TrialPruned(f"invalid DSP evaluation: {exc}") from exc

            raise

        objs_list.append(objectives)
        count += 1

    if count == 0:
        raise ValueError("evaluation requires at least one scorable region")
    if failure_log is not None:
        failure_log.success(stage)

    mean = np.mean(objs_list, axis=0)

    if transition_aware and len(objs_list) > 1:
        var_objs = np.var(objs_list, axis=0)
        penalized = mean - variance_weight * var_objs

        trans_data = build_synthetic_transition(x, sr, regions, transition_dur=min(1.0, refine_dur * 0.5))
        if trans_data is not None:
            x_trans, seam_idx = trans_data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_trans, _ = process_audio(x_trans, sr, params=p_use, master_params=mp, debug_params=dp)
                    b_metrics = score_transition_boundary(x_trans, y_trans, sr, seam_idx)
                    boundary_penalty = (
                        0.5 * b_metrics["boundary_step_error"]
                        + 0.3 * b_metrics["boundary_pumping"]
                        + 0.2 * b_metrics["boundary_stereo_delta"]
                    )
                    # Apply boundary penalty to loudness_tilt_preservation (index 4) and music_preservation (index 1)
                    penalized[4] -= 0.5 * boundary_penalty
                    penalized[1] -= 0.5 * boundary_penalty
            except Exception:
                pass

        return tuple(float(value) for value in penalized)

    return tuple(float(value) for value in mean)


def _evaluate_params_single(
    x: np.ndarray,
    sr: int,
    regions: list[Region],
    p: _m.Params,
    weights: Weights,
    band: tuple[float, float],
    refine_dur: float,
) -> float:
    obj = _evaluate_params_multi(x, sr, regions, p, weights, band, refine_dur)
    # Each metric contributes to exactly one weighted objective.
    return float(sum(obj))


# ---------------------------------------------------------------------------
# Optuna Refine Stage (Layer 2)
# ---------------------------------------------------------------------------
def _make_stage(name: str, p: _m.Params, trial: Any) -> _m.Params:
    """
    Search-space configuration for Optuna.
    Upgraded to automatically search and optimize the new high-fidelity DSP settings:
    - Mid/Side phase-coherent reconstruction parameters.
    - Advanced denoise windowing.
    - Persistent deresonation thresholds.
    """
    if name == "tonal":
        return replace(p, tonal_repair=float(trial.suggest_float("tonal_repair", 0.0, 1.0)))

    if name == "shimmer":
        return replace(
            p,
            thr_db=float(trial.suggest_float("thr_db", 4.0, 14.0)),
            slope=float(trial.suggest_float("slope", 0.3, 1.2)),
            density_lo=float(trial.suggest_float("density_lo", 0.005, 0.06)),
            density_hi=float(trial.suggest_float("density_hi", 0.08, 0.30)),
            noise_resynth=float(trial.suggest_float("noise_resynth", 0.0, 0.40)),
        )
    if name == "denoise":
        return replace(
            p,
            denoise=float(trial.suggest_float("denoise", 0.0, 0.85)),
            dn_floor_db=float(trial.suggest_float("dn_floor_db", -36.0, -12.0)),
            dn_freq_smooth_bins=int(trial.suggest_int("dn_freq_smooth_bins", 1, 9, step=2)),
            dn_release_ms=float(trial.suggest_float("dn_release_ms", 60.0, 300.0)),
            dn_attack_ms=float(trial.suggest_float("dn_attack_ms", 2.0, 20.0)),
            dn_minwin_ms=float(trial.suggest_float("dn_minwin_ms", 200.0, 1000.0)), # search min statistics window
        )
    if name == "deres":
        return replace(
            p,
            deres=float(trial.suggest_float("deres", 0.0, 0.9)),
            deq_thr_db=float(trial.suggest_float("deq_thr_db", 3.0, 10.0)),
            deq_slope=float(trial.suggest_float("deq_slope", 0.4, 1.2)),
            deq_persist_ms=float(trial.suggest_float("deq_persist_ms", 300.0, 1500.0)),
            deq_max_att_db=float(trial.suggest_float("deq_max_att_db", 4.0, 14.0)),
            deq_time_floor=bool(trial.suggest_categorical("deq_time_floor", [True, False])),
        )
    if name == "swish":
        return replace(
            p,
            swish_repair=float(trial.suggest_float("swish_repair", 0.0, 0.75)),
            swish_time_amt=float(trial.suggest_float("swish_time_amt", 0.15, 0.85)),
            swish_freq_amt=float(trial.suggest_float("swish_freq_amt", 0.0, 0.55)),
            hf_decorrelate=float(trial.suggest_float("hf_decorrelate", 0.0, 0.55)),
            ms_process=bool(trial.suggest_categorical("ms_process", [True, False])), # search if M/S improves soundstage
            ms_side_scale=float(trial.suggest_float("ms_side_scale", 0.15, 0.65)),
        )
    raise ValueError(f"unknown stage: {name}")


def _pick_pareto_trial(study: Any, mode: str = "balanced") -> Any:
    """Select weighted utility on the study's maximizing Pareto front."""
    front_trials = study.best_trials

    if not front_trials:
        raise RuntimeError("no completed, valid trials")

    if not all(np.all(np.isfinite(trial.values)) for trial in front_trials):
        raise ValueError("Pareto selection requires finite objectives")

    if mode == "safe":
        return max(front_trials, key=lambda trial: sum(trial.values[1:5]))

    if mode == "aggressive":
        return max(front_trials, key=lambda trial: trial.values[0] + sum(trial.values[5:]))

    if mode != "balanced":
        raise ValueError(f"unknown Pareto selection mode: {mode}")

    return max(front_trials, key=lambda trial: sum(trial.values))


def refine(
    x: np.ndarray,
    sr: int,
    *,
    base_params: _m.Params,
    regions: Optional[list[Region]] = None,
    aggressiveness: float = 0.5,
    n_trials_per_stage: int = 12,
    refine_dur: float = 3.0,
    debug_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    transition_aware: bool = True,
) -> tuple[_m.Params, dict[str, Any]]:
    try:
        import optuna
        from optuna.samplers import NSGAIISampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as e:
        raise RuntimeError(
            "optuna is required for refine(). Install with: pip install optuna"
        ) from e

    xm = np.asarray(x)
    sr = sr
    if regions is None or len(regions) == 0:
        regions = pick_regions(xm, sr, n=5, dur=max(refine_dur + 0.5, 5.0))

    if n_trials_per_stage < 1 or refine_dur <= 0.0:
        raise ValueError("refinement requires positive trial count and duration")

    selection_weights = weights_from_aggressiveness(aggressiveness)
    objective_weights = selection_weights
    band = (float(base_params.start_hz), float(base_params.end_hz))
    failure_log = EvaluationFailureLog(
        debug_dir or os.path.join(os.getcwd(), "auto_tune_debug"),
        planned_trials_per_stage=n_trials_per_stage,
        max_consecutive=5,
        max_failure_rate=0.20,
    )

    # Refining from bypass must return the wet mix that was actually evaluated.
    p_cur = replace(base_params, mix=1.0 if base_params.mix == 0.0 else base_params.mix)
    summary: dict[str, Any] = {
        "aggressiveness": float(aggressiveness),
        "transition_aware": transition_aware,
        "regions": [{"t0": r.t0, "dur": r.dur, "label": r.label} for r in regions],
        "objective_weights": objective_weights.__dict__,
        "selection_weights": selection_weights.__dict__,
        "objective_names": list(OBJECTIVE_NAMES),
        "failure_log": failure_log.path,
        "stages": [],
    }

    stages: list[str] = ["shimmer"]
    if p_cur.denoise > 1e-3 or aggressiveness >= 0.4:
        stages.append("denoise")
    if p_cur.deres > 1e-3:
        stages.append("deres")
    if (
        float(getattr(p_cur, "swish_repair", 0.0)) > 1e-3
        or float(getattr(p_cur, "hf_decorrelate", 0.0)) > 1e-3
        or aggressiveness >= 0.35
    ):
        stages.append("swish")

    stages.append("tonal")

    total_trials = max(1, n_trials_per_stage * len(stages))
    done = 0

    def _emit(frac: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(float(np.clip(frac, 0.0, 1.0)), msg)

    _emit(0.0, "Starting refine")

    for stage in stages:
        sampler = NSGAIISampler(seed=42)
        study = optuna.create_study(
            directions=["maximize"] * len(OBJECTIVE_NAMES),
            sampler=sampler,
        )

        # Retain the current settings when every proposed change scores worse.
        baseline = _evaluate_params_multi(
            xm, sr, regions, p_cur, objective_weights, band, refine_dur,
            transition_aware=transition_aware,
        )
        study.add_trial(optuna.trial.create_trial(values=baseline))

        def _objective(trial: Any, _stage: str = stage, _p: _m.Params = p_cur) -> tuple[float, ...]:
            p_try = _make_stage(_stage, _p, trial)
            return _evaluate_params_multi(
                xm,
                sr,
                regions,
                p_try,
                objective_weights,
                band,
                refine_dur,
                failure_log=failure_log,
                stage=_stage,
                trial_number=int(getattr(trial, "number", -1)),
                transition_aware=transition_aware,
            )

        trial_scores: list[float] = []
        for _ in range(n_trials_per_stage):
            failure_log.begin_trial(stage)
            study.optimize(_objective, n_trials=1, gc_after_trial=True, show_progress_bar=False)
            best_bal = _pick_pareto_trial(study, mode="balanced")
            trial_scores.append(float(sum(best_bal.values)))
            done += 1
            _emit(done / total_trials, f"Stage '{stage}': trial {done}/{total_trials}, score={trial_scores[-1]:.3f}")

        best_trial = _pick_pareto_trial(study, mode="balanced")
        safe_trial = _pick_pareto_trial(study, mode="safe")
        agg_trial = _pick_pareto_trial(study, mode="aggressive")
        if best_trial.params:
            p_cur = _make_stage(stage, p_cur, _SnapshotTrial(best_trial.params))
        summary["stages"].append({
            "stage": stage,
            "n_trials": n_trials_per_stage,
            "selected_score": float(sum(best_trial.values)),
            "selected_params": dict(best_trial.params),
            "selection_mode": "weighted_utility",
            "baseline_score": float(sum(baseline)),
            "retained_baseline": not bool(best_trial.params),
            "best_score": float(sum(best_trial.values)),
            "best_params": dict(best_trial.params),
            "trial_scores": trial_scores,
            "pareto": {
                "balanced": {"values": list(best_trial.values or []), "params": dict(best_trial.params)},
                "safe": {"values": list(safe_trial.values or []), "params": dict(safe_trial.params)},
                "aggressive": {"values": list(agg_trial.values or []), "params": dict(agg_trial.params)},
            },
        })

    # Final evaluation of full params
    final_score = _evaluate_params_single(xm, sr, regions, p_cur, selection_weights, band, refine_dur)
    summary["final_score"] = float(final_score)
    _emit(1.0, f"Done. Final score={final_score:.3f}")
    return p_cur, summary


class _SnapshotTrial:
    def __init__(self, params: dict[str, Any]):
        self._p = params

    def suggest_float(self, name: str, lo: float, hi: float, **_: Any) -> float:
        return float(self._p[name])

    def suggest_int(self, name: str, lo: int, hi: int, **_: Any) -> int:
        return int(self._p[name])

    def suggest_categorical(self, name: str, choices: list[Any], **_: Any) -> Any:
        return self._p[name]
