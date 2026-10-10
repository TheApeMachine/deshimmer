"""Transient & microdynamic punch restoration stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class TransientPunchConfig:
    strength: float = 0.0
    attack_boost_db: float = 2.0
    punch_low: float = 0.5
    punch_high: float = 0.5
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "TransientPunchConfig":
        return cls(
            strength=float(getattr(p, "transient_punch", 0.0)),
            attack_boost_db=float(getattr(p, "punch_attack_boost_db", 2.0)),
            punch_low=float(getattr(p, "punch_low", 0.5)),
            punch_high=float(getattr(p, "punch_high", 0.5)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class TransientPunchStage(DSPStage):
    name = "transient_punch"

    def __init__(self, config: TransientPunchConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "TransientPunchStage":
        return cls(TransientPunchConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enhance and self.config.strength > 1e-6 and self.config.attack_boost_db > 1e-3

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled():
            return

        str_val = float(np.clip(self.config.strength, 0.0, 1.0))
        max_boost = 10.0 ** (min(6.0, float(self.config.attack_boost_db) * str_val) / 20.0)
        gain_headroom = max_boost - 1.0

        freqs = ctx.freqs
        w_freq = np.ones_like(freqs, dtype=np.float32) * 0.3
        low_mask = (freqs >= 40.0) & (freqs <= 250.0)
        high_mask = (freqs >= 2000.0) & (freqs <= 10000.0)
        w_freq[low_mask] = float(np.clip(self.config.punch_low, 0.0, 1.0))
        w_freq[high_mask] = float(np.clip(self.config.punch_high, 0.0, 1.0))

        w_t = ctx.w_trans[None, :]
        w_f = w_freq[:, None]

        g_boost = 1.0 + gain_headroom * (w_f * w_t)
        g_boost = np.clip(g_boost, 1.0, 2.0).astype(np.float32)

        ctx.Z *= g_boost[:, :, None]
        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
