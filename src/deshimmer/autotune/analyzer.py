"""Signal analysis and deterministic prior estimation for auto-tuning."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..core.params import MasterParams, Params
from ..tonal.repair import TonalRepair
from .scoring import (
    _EPS,
    _band_idx,
    _flatness,
    _frame_db,
    _hpss_components,
    _residual_above_median,
    _slice,
    _stft_mag,
    _to_mono_float,
)
from .types import AnalysisReport, Region


def pick_regions(
    x: np.ndarray,
    sr: int,
    *,
    n: int = 5,
    dur: float = 5.0,
    locked: Optional[Region] = None,
) -> list[Region]:
    """Select representative audio slices covering quiet, loud, transient, and tonal conditions."""
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
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def analyze(
    x: np.ndarray,
    sr: int,
    regions: Optional[list[Region]] = None,
    *,
    base_params: Optional[Params] = None,
) -> tuple[Params, MasterParams, AnalysisReport]:
    """Perform deterministic signal analysis to configure initial pipeline parameters."""
    xm = _to_mono_float(x)
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
        p = replace(base_params if base_params is not None else Params(),
                    tonal_repair=Params().tonal_repair)
        mp = MasterParams(enabled=False)
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

    p = base_params if base_params is not None else Params()
    p = replace(p, tonal_repair=Params().tonal_repair)

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

    # --- Perceptual Diagnostic Layer ---
    # 1. Tonal Balance Diagnosis across 5 perceptual bands
    tb_summary: dict[str, float] = {}
    band_defs = [
        ("Sub_Bass", 40.0, 250.0),
        ("Low_Mid", 250.0, 800.0),
        ("Mid", 800.0, 2500.0),
        ("Presence", 2500.0, 8000.0),
        ("Air", 8000.0, min(20000.0, nyq - 50.0)),
    ]
    band_energies = []
    for b_name, lo_b, hi_b in band_defs:
        idx_b = _band_idx(freqs_ref, lo_b, hi_b)
        if idx_b.size > 0:
            e_b = float(np.mean(mag_concat[idx_b, :] ** 2))
            db_b = 10.0 * math.log10(e_b + _EPS)
        else:
            db_b = -80.0
        band_energies.append(db_b)
        tb_summary[b_name] = round(db_b, 1)

    if len(band_energies) == 5:
        sub_e, lmid_e, mid_e, pres_e, air_e = band_energies
        mid_ref = 0.5 * (sub_e + pres_e)
        low_mid_dev = lmid_e - mid_ref
        air_dev = air_e - (pres_e - 4.0)

        if low_mid_dev > 3.5:
            p.tonal_balance = 0.5
            p.tb_low_mid_db = -float(np.clip(low_mid_dev * 0.35, 0.5, 2.0))
        if air_dev < -3.5:
            p.tonal_balance = 0.5
            p.tb_air_db = float(np.clip(abs(air_dev) * 0.35, 0.5, 2.0))

    # 2. Dynamics & Punch Diagnosis
    seg_peaks = []
    seg_rms = []
    x_arr = np.asarray(x)
    for r in regions:
        seg = _slice(xm, sr, r.t0, r.dur)
        if seg.size >= 1024:
            seg_peaks.append(float(np.max(np.abs(seg)) + _EPS))
            seg_rms.append(float(np.sqrt(np.mean(seg ** 2)) + _EPS))

    if seg_peaks and seg_rms:
        overall_peak = max(seg_peaks)
        overall_rms = float(np.mean(seg_rms))
        crest_factor_db = 20.0 * math.log10((overall_peak + _EPS) / (overall_rms + _EPS))
        max_rms = max(seg_rms)
        min_rms = min(seg_rms)
        dyn_spread_db = 20.0 * math.log10((max_rms + _EPS) / (min_rms + _EPS))
    else:
        crest_factor_db = 12.0
        dyn_spread_db = 15.0

    if crest_factor_db < 9.0:
        p.transient_punch = 0.35
        p.punch_attack_boost_db = 2.0
        p.punch_low = 0.6
        p.punch_high = 0.5
        dyn_verdict = "compressed / lacks punch"
    else:
        dyn_verdict = "healthy dynamics"

    dyn_summary = {
        "crest_factor_db": round(crest_factor_db, 1),
        "dynamic_range_db": round(dyn_spread_db, 1),
        "verdict": dyn_verdict,
    }

    # 3. Low-End Coherence & Stereo Translation Diagnosis
    stereo_summary = {}
    if x_arr.ndim == 2 and x_arr.shape[1] >= 2:
        L = x_arr[:, 0]
        R = x_arr[:, 1]
        corr = float(np.sum(L * R) / (np.sqrt(np.sum(L ** 2) * np.sum(R ** 2)) + _EPS))

        sub_side_rms = 0.0
        sub_mid_rms = 0.0
        for r in regions:
            s = _slice(x_arr, sr, r.t0, r.dur)
            if s.shape[0] >= 2048:
                s_mid = 0.5 * (s[:, 0] + s[:, 1])
                s_side = 0.5 * (s[:, 0] - s[:, 1])
                M_sub = np.abs(np.fft.rfft(s_mid))
                S_sub = np.abs(np.fft.rfft(s_side))
                f_fft = np.fft.rfftfreq(s.shape[0], d=1.0 / sr)
                m_idx = (f_fft >= 20.0) & (f_fft <= 120.0)
                if np.any(m_idx):
                    sub_mid_rms += float(np.mean(M_sub[m_idx] ** 2))
                    sub_side_rms += float(np.mean(S_sub[m_idx] ** 2))

        sub_ratio = float(np.sqrt((sub_side_rms + _EPS) / (sub_mid_rms + _EPS)))

        p.bass_mono = True
        p.bass_mono_hz = 120.0
        if sub_ratio > 0.15:
            p.bass_tighten = 0.25
            st_verdict = "sub-bass phase leakage detected"
        else:
            p.bass_tighten = 0.0
            st_verdict = "coherent stereo field"

        if corr < 0.2:
            p.stereo_safe_mono = True
            p.stereo_width = 1.0
        elif corr > 0.92 and sub_ratio < 0.08:
            p.stereo_safe_mono = True
            p.stereo_width = 1.15
        else:
            p.stereo_safe_mono = True
            p.stereo_width = 1.0

        stereo_summary = {
            "phase_correlation": round(corr, 2),
            "sub_side_ratio": round(sub_ratio, 2),
            "verdict": st_verdict,
        }

    mp = MasterParams(enabled=False)

    tonal_regions = [
        {"label": region.label, **TonalRepair().analyze(_slice(np.asarray(x), sr, region.t0, region.dur), sr).metrics()}
        for region in regions
    ]
    notes: list[str] = ["Self-referenced tonal repair is active; Optimize includes its coherence objective and stage."]
    if flat_p75 < 0.12:
        notes.append("Highly tonal material: configured Swish Phase Repair and high-frequency stereo decorrelation.")
    if p.tonal_balance > 1e-3:
        notes.append("Tonal balance: automated corrective EQ configured based on spectral profile.")
    if p.transient_punch > 1e-3:
        notes.append("Dynamics: microdynamic punch restoration activated to recover transient attacks.")
    if p.bass_tighten > 1e-3:
        notes.append("Low-end: mono bass centering and bass tightening active to eliminate phase mud.")

    rep = AnalysisReport(
        detected_band=(band_lo, band_hi),
        band_db_diff=band_strength_db,
        detected_whines=res_freqs,
        measured_noise_floor_db=floor_db,
        notes=notes,
        tonal_regions=tonal_regions,
        tonal_balance_summary=tb_summary,
        dynamics_summary=dyn_summary,
        stereo_summary=stereo_summary,
    )
    return p, mp, rep
