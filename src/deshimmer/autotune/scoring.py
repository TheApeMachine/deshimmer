"""Scoring objectives, acoustic metrics, transition boundary evaluation, and weighting."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft as _scipy_stft

from ..core import (
    _as_2d,
    measure_lufs,
    measure_phase_instability,
    measure_rms_dbfs,
)
from ..core.params import Params
from ..tonal import TonalRepair
from .types import Region, Weights

_EPS = 1e-12


def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=1).astype(np.float32, copy=False)
    raise ValueError("audio must be 1D or 2D")


def _slice(x: np.ndarray, sr: int, t0: float, dur: float) -> np.ndarray:
    n = x.shape[0]
    s0 = round(max(0.0, t0) * sr)
    s1 = round(max(0.0, t0 + dur) * sr)
    s0 = max(0, min(n, s0))
    s1 = max(s0, min(n, s1))
    if x.ndim == 1:
        return x[s0:s1]
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
    - Prevents content-preservation penalties from dropping to zero at high aggressiveness.
    - Keeps transient leaks, musical noise, and stereo image deviations heavily penalized.
    """
    a = float(np.clip(agg, 0.0, 1.0))
    return Weights(
        band_reduction=1.0 + 1.2 * a,
        swish_reduction=0.9 + 1.5 * a,
        out_of_band_change=1.2 + 0.8 * (1.0 - a),
        musical_noise=1.5 + 0.5 * (1.0 - a),
        transient_leak=1.8 + 0.6 * (1.0 - a),
        harmonic_leak=1.4 + 0.6 * (1.0 - a),
        loudness_loss=0.8 + 0.4 * (1.0 - a),
        band_energy_loss=1.0 + 0.6 * (1.0 - a),
        spectral_tilt=0.8 + 0.6 * (1.0 - a),
        centroid_shift=0.7 + 0.5 * (1.0 - a),
        diff_onset_corr=1.5 + 0.9 * (1.0 - a),
        stereo_width=1.6 + 0.8 * (1.0 - a),
        tonal_coherence=1.0 + 1.2 * a,
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
    n_fft = 2 * hop
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

    if _as_2d(np.asarray(x_in)).shape != _as_2d(np.asarray(y_out)).shape:
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

    inst_in = measure_phase_instability(xm_in, sr, *swish_band)
    inst_out = measure_phase_instability(xm_out, sr, *swish_band)
    inst_denom = max(inst_in, 0.02)
    swish_reduction = float(np.clip((inst_in - inst_out) / inst_denom, -0.5, 1.0)) * 12.0

    outside_band = (freqs < band[0]) | (freqs > band[1])
    out_of_band_change = 0.0

    if np.any(outside_band):
        out_of_band_change = abs(_energy_db(mag_in[outside_band]) - _energy_db(mag_out[outside_band]))

    band_db_in = _frame_db(mag_in)
    thr = float(np.percentile(band_db_in, 25.0))
    quiet_mask = band_db_in <= thr
    if int(np.sum(quiet_mask)) >= 4:
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

    lufs_in = measure_lufs(xm_in, sr)
    lufs_out = measure_lufs(xm_out, sr)
    if lufs_in is not None and lufs_out is not None and math.isfinite(lufs_in) and math.isfinite(lufs_out):
        loud_loss = max(0.0, float(lufs_in) - float(lufs_out) - 1.0)
    else:
        loud_loss = max(0.0, measure_rms_dbfs(xm_in) - measure_rms_dbfs(xm_out) - 1.0)

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


def compute_enhancement_rewards(
    x_in: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    stage: Optional[str] = None,
    p: Optional[Params] = None,
) -> dict[int, float]:
    """Calculate dedicated measurable improvement rewards for sound goodizer stages."""
    if p is None and isinstance(stage, Params):
        p, stage = stage, None
    if p is None:
        return {}

    eval_all = stage is None or stage in ("all", "refine")
    rewards: dict[int, float] = {}

    # 1. Tonal Balance
    if (eval_all or stage == "tonal_balance") and getattr(p, "tonal_balance", 0.0) > 1e-4:
        # Measure balance improvement: reduction in 5-band spectral variance
        mag_in, freqs = _stft_mag(x_in, sr)
        mag_out, _ = _stft_mag(y_out, sr)
        bands = [(40.0, 250.0), (200.0, 900.0), (700.0, 3000.0), (2500.0, 8000.0), (7000.0, min(18000.0, sr * 0.48))]
        e_in = [_band_energy_db(mag_in, freqs, lo, hi) for lo, hi in bands]
        e_out = [_band_energy_db(mag_out, freqs, lo, hi) for lo, hi in bands]
        ref_slope = np.array([0.0, -3.0, -6.0, -9.0, -12.0])
        var_in = float(np.var(np.array(e_in) - ref_slope))
        var_out = float(np.var(np.array(e_out) - ref_slope))
        var_reduction = max(0.0, var_in - var_out)
        r_val = float(np.clip(1.5 * var_reduction, 0.0, 8.0))
        rewards[4] = rewards.get(4, 0.0) + r_val
        rewards[1] = rewards.get(1, 0.0) + 0.5 * r_val

    # 2. Transient Punch
    if (eval_all or stage == "transient_punch") and getattr(p, "transient_punch", 0.0) > 1e-4:
        # Measure punch improvement: attack transient crest factor gain on onset frames
        xm_in = _to_mono_float(x_in)
        xm_out = _to_mono_float(y_out)
        hop = 256
        frame_len = 1024
        # Explicitly pass hop to align onset detection frame indices with sample slicing
        env_in = _onset_envelope(xm_in, sr, hop=hop)
        onsets = np.where(env_in > np.percentile(env_in, 85.0))[0]
        if len(onsets) > 2:
            crest_in_list = []
            crest_out_list = []
            for o in onsets:
                s0 = o * hop
                s1 = min(len(xm_in), s0 + frame_len)
                if s1 - s0 >= 256:
                    c_in = float(np.max(np.abs(xm_in[s0:s1])) / (np.sqrt(np.mean(xm_in[s0:s1] ** 2)) + 1e-6))
                    c_out = float(np.max(np.abs(xm_out[s0:s1])) / (np.sqrt(np.mean(xm_out[s0:s1] ** 2)) + 1e-6))
                    crest_in_list.append(c_in)
                    crest_out_list.append(c_out)
            if crest_in_list:
                mean_cin = float(np.mean(crest_in_list))
                mean_cout = float(np.mean(crest_out_list))
                crest_gain_db = 20.0 * math.log10(max(1e-3, mean_cout) / max(1e-3, mean_cin))
                punch_reward = float(np.clip(max(0.0, crest_gain_db) * 2.5, 0.0, 7.5))
                rewards[2] = rewards.get(2, 0.0) + punch_reward
                rewards[1] = rewards.get(1, 0.0) + 0.5 * punch_reward

    # 3. Low-End Coherence
    if (eval_all or stage == "low_end") and (bool(getattr(p, "bass_mono", False)) or getattr(p, "bass_tighten", 0.0) > 1e-4):
        # Measure low-end coherence: reduction in sub-bass side energy
        if x_in.ndim == 2 and x_in.shape[1] >= 2 and y_out.ndim == 2 and y_out.shape[1] >= 2:
            s_mid_in = 0.5 * (x_in[:, 0] + x_in[:, 1])
            s_side_in = 0.5 * (x_in[:, 0] - x_in[:, 1])
            s_mid_out = 0.5 * (y_out[:, 0] + y_out[:, 1])
            s_side_out = 0.5 * (y_out[:, 0] - y_out[:, 1])
            f_fft = np.fft.rfftfreq(len(s_mid_in), d=1.0 / sr)
            m_sub = (f_fft >= 20.0) & (f_fft <= 120.0)
            if np.any(m_sub):
                M_in = float(np.mean(np.abs(np.fft.rfft(s_mid_in))[m_sub] ** 2))
                S_in = float(np.mean(np.abs(np.fft.rfft(s_side_in))[m_sub] ** 2))
                S_out = float(np.mean(np.abs(np.fft.rfft(s_side_out))[m_sub] ** 2))
                sub_ratio_in = float(np.sqrt((S_in + 1e-12) / (M_in + 1e-12)))
                sub_ratio_out = float(np.sqrt((S_out + 1e-12) / (M_in + 1e-12)))
                mono_reward = float(np.clip(max(0.0, sub_ratio_in - sub_ratio_out) * 12.0, 0.0, 6.0))
                rewards[3] = rewards.get(3, 0.0) + mono_reward
                rewards[0] = rewards.get(0, 0.0) + 0.5 * mono_reward

    # 4. Stereo Translation
    if (eval_all or stage == "stereo_enhance") and (abs(float(getattr(p, "stereo_width", 1.0)) - 1.0) > 1e-3 or bool(getattr(p, "stereo_safe_mono", False))):
        # Measure stereo translation: safe width without anti-phase corruption
        if x_in.ndim == 2 and x_in.shape[1] >= 2 and y_out.ndim == 2 and y_out.shape[1] >= 2:
            L_in, R_in = x_in[:, 0], x_in[:, 1]
            L_out, R_out = y_out[:, 0], y_out[:, 1]
            corr_in = float(np.sum(L_in * R_in) / (np.sqrt(np.sum(L_in ** 2) * np.sum(R_in ** 2)) + 1e-12))
            corr_out = float(np.sum(L_out * R_out) / (np.sqrt(np.sum(L_out ** 2) * np.sum(R_out ** 2)) + 1e-12))
            w_in = _stereo_width_ratio(x_in)
            w_out = _stereo_width_ratio(y_out)
            if corr_out >= min(corr_in - 0.05, 0.4):
                w_gain = float(np.clip(max(0.0, w_out - w_in) * 15.0, 0.0, 5.0))
                rewards[3] = rewards.get(3, 0.0) + w_gain

    # 5. Harmonic Enhancement
    if (eval_all or stage == "harmonic") and getattr(p, "harmonic_enhance", 0.0) > 1e-4:
        # Measure coherent harmonic generation in target high band:
        # Reward genuine harmonic prominence (peak-to-median ratio gain),
        # not just diffuse broadband energy or simple high-shelf boosts.
        mag_in, freqs = _stft_mag(x_in, sr)
        mag_out, _ = _stft_mag(y_out, sr)
        start_hz = float(getattr(p, "harmonic_start_hz", 3500.0))
        h_idx = np.where(freqs >= start_hz)[0]
        if h_idx.size > 0:
            e_in = float(np.mean(mag_in[h_idx, :] ** 2))
            e_out = float(np.mean(mag_out[h_idx, :] ** 2))
            crest_in = float(np.max(mag_in[h_idx, :]) / (np.median(mag_in[h_idx, :]) + 1e-6))
            crest_out = float(np.max(mag_out[h_idx, :]) / (np.median(mag_out[h_idx, :]) + 1e-6))
            prominence_gain = max(0.0, crest_out - crest_in)
            h_gain = max(0.0, 10.0 * math.log10((e_out + 1e-12) / (e_in + 1e-12)))
            harm_reward = float(np.clip(h_gain * 0.8 + prominence_gain * 0.5, 0.0, 5.0))
            rewards[5] = rewards.get(5, 0.0) + harm_reward
            rewards[1] = rewards.get(1, 0.0) + 0.5 * harm_reward

    return rewards


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
    """
    if len(regions) < 2:
        return None

    xm = np.asarray(x)
    total_samples = xm.shape[0]

    loud_reg = next((r for r in regions if r.label == "loud"), None)
    quiet_reg = next((r for r in regions if r.label == "quiet"), None)

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
    - boundary_step_error: unnatural gating across the loud->quiet seam.
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
