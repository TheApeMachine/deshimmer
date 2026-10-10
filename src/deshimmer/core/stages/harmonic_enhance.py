"""Musical harmonic overtone enhancement stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


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
        hf_idx = np.where(freqs >= start_hz)[0]
        src_idx = np.where((freqs >= start_hz * 0.4) & (freqs <= start_hz))[0]

        if hf_idx.size == 0 or src_idx.size == 0:
            return

        taper = np.exp(-0.5 * ((freqs[hf_idx] - start_hz) / max(1000.0, start_hz)) ** 2)[:, None, None]
        excite_gain = 1.0 + str_val * 0.4 * taper

        ctx.Z[hf_idx, :, :] *= excite_gain.astype(np.complex64)
        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
