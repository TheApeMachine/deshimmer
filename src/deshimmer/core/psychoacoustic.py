"""Psychoacoustic auditory masking and forward temporal masking decay."""

from __future__ import annotations

from typing import Optional
import numpy as np

from .utils import _frame_coeff

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _calculate_psychoacoustic_mask(PSD_mono: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Vectorized auditory masking threshold calculation (SMR integration):
    - Converts frequencies to Bark scale.
    - Computes absolute threshold of hearing (ATH) in quiet.
    - Spreads energy across Bark bands to determine masking limits.
    - Prevents over-processing of signals that are already psychoacoustically masked.
    """
    F, T = PSD_mono.shape
    bark = 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)

    # ATH in dB SPL
    f_khz = np.clip(freqs / 1000.0, 0.02, 20.0)
    ath = 3.64 * (f_khz ** -0.8) - 6.5 * np.exp(-0.6 * ((f_khz - 3.3) ** 2)) + 1e-3 * (f_khz ** 4)
    # ATH converted to nominal PSD level (0 dBFS = 96 dB SPL)
    ath_psd = 10.0 ** ((ath - 96.0) / 10.0)

    # Bark differences
    bark_diff = bark[:, None] - bark[None, :]
    # Spreading slopes: -25 dB/Bark on high side, -10 dB/Bark on low side
    # We apply a -18 dB SMR offset relative to the tone masker
    spread_db = np.where(bark_diff >= 0, -25.0 * bark_diff, 10.0 * bark_diff) - 18.0
    spread_db = np.clip(spread_db, -100.0, 0.0)
    S_linear = 10.0 ** (spread_db / 10.0)

    # Matrix multiply for vectorized spreading
    mask_psd = np.matmul(S_linear, PSD_mono)

    # Max with absolute threshold of hearing
    mask_psd = np.maximum(mask_psd, ath_psd[:, None])
    return mask_psd.astype(np.float32)


@jit(nopython=True)
def _numba_temporal_mask_decay(mask_psd: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    """
    Forward temporal psychoacoustic masking decay (one-pole peak-hold decay).
    Holds masking threshold high after broadband attacks, modulated by alpha_vec.
    """
    F, T = mask_psd.shape
    out = np.empty((F, T), dtype=np.float32)
    for f in range(F):
        out[f, 0] = mask_psd[f, 0]
    for t in range(1, T):
        a = alpha_vec[t]
        for f in range(F):
            prev = out[f, t - 1] * a
            curr = mask_psd[f, t]
            out[f, t] = curr if curr > prev else prev
    return out


def _py_temporal_mask_decay(mask_psd: np.ndarray, alpha_vec: np.ndarray) -> np.ndarray:
    F, T = mask_psd.shape
    out = np.empty((F, T), dtype=np.float32)
    out[:, 0] = mask_psd[:, 0]
    for t in range(1, T):
        prev = out[:, t - 1] * alpha_vec[t]
        curr = mask_psd[:, t]
        out[:, t] = np.maximum(curr, prev)
    return out


def apply_temporal_masking_decay(
    mask_psd: np.ndarray,
    sr: int,
    hop: int,
    *,
    decay_ms: float = 15.0,
    flatness: Optional[np.ndarray] = None,
    flat_start: float = 0.25,
    flat_end: float = 0.70,
    gated: bool = True,
) -> np.ndarray:
    """
    Applies forward temporal psychoacoustic masking decay across time frames.
    Gated by spectral flatness: sustains broadband reverb tails while letting
    tonal codec ringing fall through for suppression.
    """
    if mask_psd.shape[1] <= 1:
        return mask_psd.astype(np.float32, copy=False)

    alpha_base = np.float32(_frame_coeff(hop, sr, max(1.0, float(decay_ms))))
    T = mask_psd.shape[1]

    if gated and flatness is not None:
        flat = np.asarray(flatness, dtype=np.float32)
        gate = np.clip((flat - float(flat_start)) / max(1e-6, float(flat_end) - float(flat_start)), 0.0, 1.0)
        alpha_vec = (alpha_base * gate).astype(np.float32)
    else:
        alpha_vec = np.full(T, alpha_base, dtype=np.float32)

    if HAS_NUMBA:
        return _numba_temporal_mask_decay(mask_psd.astype(np.float32, copy=False), alpha_vec)
    return _py_temporal_mask_decay(mask_psd.astype(np.float32, copy=False), alpha_vec)
