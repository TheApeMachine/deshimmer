"""Mathematical and DSP utility conversions and filters."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


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


def _butter_sos(sr: int, kind: str, cutoff_hz, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    if kind in ("lowpass", "highpass"):
        wn = float(cutoff_hz) / nyq
        wn = float(np.clip(wn, 1e-6, 0.999999))
        return np.asarray(butter(order, wn, btype=kind, output="sos"))
    if kind == "bandpass":
        lo, hi = cutoff_hz
        lo = float(np.clip(lo / nyq, 1e-6, 0.999999))
        hi = float(np.clip(hi / nyq, 1e-6, 0.999999))
        if hi <= lo:
            raise ValueError("Invalid bandpass cutoffs")
        return np.asarray(butter(order, [lo, hi], btype="bandpass", output="sos"))
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
