"""Self-adapting perceptual tonal balance equalizer stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class TonalBalanceConfig:
    strength: float = 0.0
    low_db: float = 0.0          # ~40 - 250 Hz (bass / sub)
    low_mid_db: float = 0.0      # ~200 - 900 Hz (warmth / mud)
    mid_db: float = 0.0          # ~700 - 3000 Hz (body / presence)
    presence_db: float = 0.0     # ~2500 - 8000 Hz (clarity / harshness)
    air_db: float = 0.0          # ~7000 - 22000 Hz (openness / air)
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "TonalBalanceConfig":
        return cls(
            strength=float(getattr(p, "tonal_balance", 0.0)),
            low_db=float(getattr(p, "tb_low_db", 0.0)),
            low_mid_db=float(getattr(p, "tb_low_mid_db", 0.0)),
            mid_db=float(getattr(p, "tb_mid_db", 0.0)),
            presence_db=float(getattr(p, "tb_presence_db", 0.0)),
            air_db=float(getattr(p, "tb_air_db", 0.0)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class TonalBalanceStage(DSPStage):
    name = "tonal_balance"

    def __init__(self, config: TonalBalanceConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "TonalBalanceStage":
        return cls(TonalBalanceConfig.from_params(p))

    def is_enabled(self) -> bool:
        if not self.config.enhance or self.config.strength <= 1e-6:
            return False
        return any(
            abs(db) > 1e-3
            for db in (
                self.config.low_db,
                self.config.low_mid_db,
                self.config.mid_db,
                self.config.presence_db,
                self.config.air_db,
            )
        )

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled():
            return

        freqs = ctx.freqs
        if freqs.size == 0:
            return

        bands = [
            (120.0, 1.2, self.config.low_db),
            (400.0, 1.0, self.config.low_mid_db),
            (1500.0, 1.0, self.config.mid_db),
            (4500.0, 0.9, self.config.presence_db),
            (12000.0, 0.8, self.config.air_db),
        ]

        str_val = float(np.clip(self.config.strength, 0.0, 1.0))
        gain_db_total = np.zeros_like(freqs, dtype=np.float32)

        log_f = np.log10(np.maximum(20.0, freqs))

        for center_hz, q_oct, db_val in bands:
            if abs(db_val) <= 1e-3:
                continue
            log_c = np.log10(center_hz)
            sigma = q_oct * 0.301
            curve = np.exp(-0.5 * ((log_f - log_c) / max(0.05, sigma)) ** 2)
            gain_db_total += curve.astype(np.float32) * float(db_val * str_val)

        gain_db_total = np.clip(gain_db_total, -6.0, 6.0)
        gain_lin = 10.0 ** (gain_db_total / 20.0)

        ctx.Z *= gain_lin[:, None, None]
        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
