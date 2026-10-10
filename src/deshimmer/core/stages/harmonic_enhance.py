"""Musical harmonic overtone enhancement stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@njit(fastmath=True)
def _numba_synthesize_harmonics(
    Z: np.ndarray,
    freqs: np.ndarray,
    start_hz: float,
    max_hz: float,
    strength: float,
    is_transient: np.ndarray,
) -> None:
    n_bins, n_frames, n_ch = Z.shape
    min_src_hz = start_hz * 0.25

    k_src_min = 1
    while k_src_min < n_bins and freqs[k_src_min] < min_src_hz:
        k_src_min += 1

    k_src_max = k_src_min
    while k_src_max < n_bins and freqs[k_src_max] <= start_hz:
        k_src_max += 1

    if k_src_max <= k_src_min + 2:
        return

    orders = (2, 3, 4)
    alphas = (0.45, 0.22, 0.10)

    for c in range(n_ch):
        for t in range(n_frames):
            trans_scale = 1.0 - 0.5 * is_transient[t]
            for k in range(k_src_min + 1, k_src_max - 1):
                mag_curr = np.abs(Z[k, t, c])
                if mag_curr < 1e-5:
                    continue
                mag_prev = np.abs(Z[k - 1, t, c])
                mag_next = np.abs(Z[k + 1, t, c])
                if mag_curr > mag_prev and mag_curr > mag_next:
                    f_src = freqs[k]
                    phi_src = np.angle(Z[k, t, c])

                    for o_idx in range(3):
                        n_ord = orders[o_idx]
                        alpha = alphas[o_idx]
                        f_target = n_ord * f_src
                        if f_target < start_hz or f_target > max_hz:
                            continue

                        k_target = int(np.round(n_ord * k))
                        if k_target <= 1 or k_target >= n_bins - 1:
                            continue

                        norm_dist = (f_target - start_hz) / max(3000.0, start_hz)
                        taper = np.exp(-0.5 * norm_dist * norm_dist)
                        amp = strength * alpha * mag_curr * taper * trans_scale
                        phi = n_ord * phi_src

                        re = amp * np.cos(phi)
                        im = amp * np.sin(phi)
                        dZ = np.complex64(re + 1j * im)

                        Z[k_target - 1, t, c] += 0.25 * dZ
                        Z[k_target, t, c] += 0.50 * dZ
                        Z[k_target + 1, t, c] += 0.25 * dZ


@dataclass
class HarmonicEnhanceConfig:
    strength: float = 0.0
    start_hz: float = 3500.0
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "HarmonicEnhanceConfig":
        return cls(
            strength=float(getattr(p, "harmonic_enhance", 0.0)),
            start_hz=float(getattr(p, "harmonic_start_hz", 3500.0)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class HarmonicEnhanceStage(DSPStage):
    name = "harmonic_enhance"

    def __init__(self, config: HarmonicEnhanceConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "HarmonicEnhanceStage":
        return cls(HarmonicEnhanceConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enhance and self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled():
            return

        str_val = float(np.clip(self.config.strength, 0.0, 0.5))
        if str_val <= 1e-6:
            return

        freqs = ctx.freqs
        start_hz = max(1000.0, float(self.config.start_hz))
        max_hz = float(ctx.sr) * 0.48

        is_trans = ctx.w_trans if hasattr(ctx, "w_trans") else np.zeros(ctx.n_cols, dtype=np.float32)

        _numba_synthesize_harmonics(
            ctx.Z,
            freqs,
            start_hz,
            max_hz,
            str_val,
            is_trans,
        )

        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
