"""Phase blur stage for diffusing high-Q metallic phase alignment in target bands."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class PhaseBlurConfig:
    strength: float = 0.0
    start_hz: float = 3000.0
    end_hz: float = 8000.0
    harmonic_only: bool = True
    hpss_time_frames: int = 21
    hpss_freq_bins: int = 17

    @classmethod
    def from_params(cls, p: Params) -> "PhaseBlurConfig":
        return cls(
            strength=float(getattr(p, "phase_blur", 0.0)),
            start_hz=float(getattr(p, "pb_start_hz", 3000.0)),
            end_hz=float(getattr(p, "pb_end_hz", 8000.0)),
            harmonic_only=bool(getattr(p, "pb_harmonic_only", True)),
            hpss_time_frames=int(getattr(p, "hpss_time_frames", 21)),
            hpss_freq_bins=int(getattr(p, "hpss_freq_bins", 17)),
        )


class PhaseBlurStage(DSPStage):
    name = "phase_blur"

    def __init__(self, config: PhaseBlurConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "PhaseBlurStage":
        return cls(PhaseBlurConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        pb_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        pb_amt = float(self.config.strength)
        if pb_amt <= 1e-6 or pb_idx.size == 0:
            return

        phases = ctx.rng.uniform(0.0, 2.0 * np.pi, size=ctx.Z[pb_idx].shape[:2]).astype(np.float32)
        zph = (np.cos(phases) + 1j * np.sin(phases)).astype(np.complex64)[:, :, None]
        mix_v = pb_amt
        if self.config.harmonic_only:
            mag_pb = ctx.Mag_mono[pb_idx, :]
            L = np.log(mag_pb + ctx.eps)
            H = median_filter(L, size=(1, self.config.hpss_time_frames), mode='nearest')
            P = median_filter(L, size=(self.config.hpss_freq_bins, 1), mode='nearest')
            Hlin = np.exp(H)
            Plin = np.exp(P)
            Mh = Hlin ** 2 / (Hlin ** 2 + Plin ** 2 + ctx.eps)
            mix_v = pb_amt * Mh

        m = np.asarray(mix_v) if np.isscalar(mix_v) else np.asarray(mix_v)[:, :, None]
        ctx.Z[pb_idx] = (1.0 - m) * ctx.Z[pb_idx] + m * (np.abs(ctx.Z[pb_idx]) * zph)
