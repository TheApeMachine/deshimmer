"""Harmonic/Percussive source separation (HPSS) masks, magnitude inpainting, phasor smoothing, and phase repair."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .measurements import _phase_instability_map


def _hpss_harmonic_mask(log_mag: np.ndarray, time_frames: int, freq_bins: int, eps: float = 1e-12) -> np.ndarray:
    tf = max(1, time_frames)
    ff = max(1, freq_bins)
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
    mag = np.abs(Z).astype(np.float32)
    w = (weights.astype(np.float32) * mag).astype(np.float32)
    unit = Z / (mag + eps)
    k = max(3, size | 1)

    def _filt(a: np.ndarray) -> np.ndarray:
        return uniform_filter1d(a.astype(np.float64), size=k, axis=axis, mode="nearest").astype(np.float32)

    wr = _filt(w * np.real(unit))
    wi = _filt(w * np.imag(unit))
    ww = _filt(w) + eps
    u = (wr + 1j * wi) / ww
    u /= (np.abs(u) + eps)
    return u.astype(np.complex64, copy=False)


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
    if band_idx.size == 0 or strength <= 1e-6:
        return

    zm = np.mean(Z[band_idx, :, :], axis=2)
    instab = _phase_instability_map(zm, eps=eps)

    log_mag = np.log(Mag_mono[band_idx, :] + eps)
    t_win = max(3, hpss_time_frames | 1)
    f_win = max(3, hpss_freq_bins | 1)
    h_lin = np.exp(median_filter(log_mag, size=(1, t_win), mode="nearest"))
    p_lin = np.exp(median_filter(log_mag, size=(f_win, 1), mode="nearest"))
    mh = (h_lin ** 2 / (h_lin ** 2 + p_lin ** 2 + eps)).astype(np.float32, copy=False)

    w_trans = 1.0 - w_nontrans.astype(np.float32, copy=False)
    w_prot = (1.0 - float(np.clip(transient_protect, 0.0, 1.0)) * w_trans).astype(np.float32, copy=False)

    depth = (
        float(np.clip(strength, 0.0, 1.0))
        * instab
        * w_prot[None, :]
        * mh
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
    if band_idx.size == 0:
        return np.zeros((0, Mag_mono.shape[1]), dtype=np.float32)
    mag = Mag_mono[band_idx, :]
    L = np.log(mag + eps)
    L_med = median_filter(L, size=(max(3, freq_med_bins), 1), mode="nearest")
    resid = (L - L_med) * (20.0 / np.log(10.0))
    conf = np.clip((resid - float(thr_db)) / max(1e-6, float(thr_db)), 0.0, 1.0)
    return conf.astype(np.float32, copy=False)
