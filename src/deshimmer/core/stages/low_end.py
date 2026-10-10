"""Low-end coherence, mono bass, and boom containment stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class LowEndConfig:
    bass_mono: bool = False
    mono_hz: float = 120.0
    tighten: float = 0.0
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "LowEndConfig":
        return cls(
            bass_mono=bool(getattr(p, "bass_mono", False)),
            mono_hz=float(getattr(p, "bass_mono_hz", 120.0)),
            tighten=float(getattr(p, "bass_tighten", 0.0)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class LowEndStage(DSPStage):
    name = "low_end"

    def __init__(self, config: LowEndConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "LowEndStage":
        return cls(LowEndConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enhance and (self.config.bass_mono or self.config.tighten > 1e-6)

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled():
            return

        freqs = ctx.freqs
        mono_hz = max(20.0, float(self.config.mono_hz))
        low_idx = np.where(freqs <= mono_hz * 1.5)[0]

        if low_idx.size == 0:
            return

        if self.config.bass_mono and ctx.n_ch >= 2:
            left = ctx.Z[low_idx, :, 0]
            right = ctx.Z[low_idx, :, 1]
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)

            f_sub = freqs[low_idx]
            side_scale = np.clip(f_sub / mono_hz, 0.0, 1.0) ** 2
            side_new = side * side_scale[:, None]

            ctx.Z[low_idx, :, 0] = mid + side_new
            ctx.Z[low_idx, :, 1] = mid - side_new

        if self.config.tighten > 1e-6:
            tight_str = float(np.clip(self.config.tighten, 0.0, 1.0))
            boom_idx = np.where((freqs >= 40.0) & (freqs <= 200.0))[0]
            if boom_idx.size > 0:
                w_decay = ctx.w_nontrans[None, :]
                att_factor = 1.0 - 0.35 * tight_str * w_decay
                att_factor = np.clip(att_factor, 0.5, 1.0).astype(np.float32)
                ctx.Z[boom_idx, :, :] *= att_factor[:, :, None]

        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
