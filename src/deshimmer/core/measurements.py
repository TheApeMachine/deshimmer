"""Audio quality, level, loudness, and phase instability measurement functions."""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.signal import resample_poly, stft

from .utils import _as_2d, _lin_to_db


def measure_true_peak_db(x: np.ndarray, os_factor: int = 4) -> float:
    """Measure true peak in dBTP using polyphase oversampling."""
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    osf = max(1, os_factor)
    if osf > 1:
        x_os = resample_poly(x2, osf, 1, axis=0).astype(np.float32, copy=False)
    else:
        x_os = x2
    peak = float(np.max(np.abs(x_os))) if x_os.size else 0.0
    return float(_lin_to_db(peak))


def measure_rms_dbfs(x: np.ndarray) -> float:
    """Measure RMS level in dBFS."""
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    rms = float(np.sqrt(np.mean(x2 * x2) + 1e-12))
    return float(_lin_to_db(rms))


def measure_lufs(x: np.ndarray, sr: int) -> Optional[float]:
    """Measure integrated loudness in LUFS using pyloudnorm if available."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(_as_2d(x)))
    except Exception:
        return None


def _phase_instability_map(Z_mono: np.ndarray, eps: float) -> np.ndarray:
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
    """Measure high-frequency phase instability (swish/jitter) across time and frequency."""
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
        fs=sr,
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
