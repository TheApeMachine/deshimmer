"""Stereo alignment, spectral balancing, and dynamic spectral carving."""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate


def align_lr_delay(x: np.ndarray, sr: int, max_delay_ms: float = 3.0) -> np.ndarray:
    """
    Analyzes and aligns Left and Right channel micro-delays using cross-correlation.
    Eliminates high-frequency comb-filtering and centers the stereo image.
    """
    if x.ndim == 1 or x.shape[1] < 2:
        return x

    left = x[:, 0]
    right = x[:, 1]

    # Analyze the first 15 seconds to calculate the temporal lag quickly and reliably
    analysis_len = min(left.size, (sr * 15))
    l_segment = left[:analysis_len]
    r_segment = right[:analysis_len]

    # FFT correlation avoids quadratic work on the analysis segment.
    corr = correlate(l_segment, r_segment, mode='same', method='fft')
    center = corr.size // 2
    lag = np.argmax(corr) - center

    max_delay_samples = math.ceil((max_delay_ms / 1000.0) * sr)

    if abs(lag) > max_delay_samples or lag == 0:
        return x

    y = x.copy()
    if lag > 0:
        # Positive correlation lag means Left is late: delay Right.
        y[:, 1] = 0
        y[lag:, 1] = right[:-lag]
    else:
        # Negative correlation lag means Right is late: delay Left.
        y[:, 0] = 0
        y[-lag:, 0] = left[:lag]

    return y


def balance_lr_spectrum(Z: np.ndarray, smooth_bins: int = 21) -> np.ndarray:
    """
    Symmetrically balances Left and Right channels' long-term average spectra
    to restore a centered, stable stereo field.
    Z is (F, T, Ch)
    """
    if Z.shape[2] < 2:
        return Z

    eps = 1e-12
    # Compute long-term average magnitude for both channels
    mag_l = np.mean(np.abs(Z[:, :, 0]), axis=1)
    mag_r = np.mean(np.abs(Z[:, :, 1]), axis=1)

    # Calculate difference ratio
    ratio = (mag_l + eps) / (mag_r + eps)

    # Smooth the ratio over frequency to prevent narrow-band distortions
    ratio_smooth = uniform_filter1d(ratio, size=smooth_bins, mode='nearest')

    # Limit maximum correction to +/- 3 dB to preserve artistic mixing intents
    ratio_smooth = np.clip(ratio_smooth, 10.0**(-3.0/20.0), 10.0**(3.0/20.0))

    # Symmetrically apply correction (split the difference between channels)
    comp_l = 1.0 / np.sqrt(ratio_smooth)
    comp_r = np.sqrt(ratio_smooth)

    Z_new = Z.copy()
    Z_new[:, :, 0] *= comp_l[:, None]
    Z_new[:, :, 1] *= comp_r[:, None]
    return Z_new


def dynamic_spectral_carver(Z: np.ndarray, freqs: np.ndarray, target_slope: float = -3.5) -> np.ndarray:
    """
    Dynamically carves and compresses frequency accumulations to conform with
    the statistical target curves of commercial mastering (-3.5 dB/octave slope).
    Z is (F, T, Ch)
    """
    eps = 1e-12
    F, T, Ch = Z.shape

    # Target mastering curve: 0 dB reference at 100 Hz, rolling off by target_slope per octave
    octaves = np.log2(np.clip(freqs, 100.0, freqs[-1]) / 100.0)
    target_db = target_slope * octaves
    target_linear = 10.0 ** (target_db / 20.0)

    Z_new = Z.copy()
    for t in range(T):
        # Average magnitude of the current frame across channels
        mag_frame = np.mean(np.abs(Z[:, t, :]), axis=1)

        # Smooth frame envelope to find broad resonances (muddiness/harshness)
        mag_smooth = uniform_filter1d(mag_frame, size=31, mode='nearest')

        # Normalize the target shape to match the current frame's total energy
        scale_factor = np.sum(mag_frame) / np.sum(target_linear)
        frame_target = target_linear * scale_factor

        # Calculate excess energy relative to high-fidelity target shape
        excess = mag_smooth / (frame_target + eps)

        # Target only build-ups exceeding the curve by 1.5 dB
        carve_mask = excess > 10.0 ** (1.5 / 20.0)

        if np.any(carve_mask):
            # Dynamic compression: apply a gentle 20% correction ratio
            g_carve = 1.0 - 0.20 * (1.0 - 1.0 / excess)
            g_carve = np.clip(g_carve, 0.5, 1.0)  # limit maximum carve cut to -6 dB

            Z_new[carve_mask, t, :] *= g_carve[carve_mask, None]

    return Z_new
