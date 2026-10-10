"""Translation-safe stereo width optimization and mono compatibility stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class StereoTranslationConfig:
    width: float = 1.0
    safe_mono: bool = False
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "StereoTranslationConfig":
        return cls(
            width=float(getattr(p, "stereo_width", 1.0)),
            safe_mono=bool(getattr(p, "stereo_safe_mono", False)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class StereoTranslationStage(DSPStage):
    name = "stereo_translation"

    def __init__(self, config: StereoTranslationConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "StereoTranslationStage":
        return cls(StereoTranslationConfig.from_params(p))

    def is_enabled(self) -> bool:
        if not self.config.enhance:
            return False
        return abs(self.config.width - 1.0) > 1e-3 or self.config.safe_mono

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled() or ctx.n_ch < 2:
            return

        L = ctx.Z[:, :, 0]
        R = ctx.Z[:, :, 1]
        M = 0.5 * (L + R)
        S = 0.5 * (L - R)

        mag_L = np.abs(L)
        mag_R = np.abs(R)
        denom = mag_L * mag_R + ctx.eps
        cos_dphi = np.real(L * np.conj(R)) / denom

        width_target = float(np.clip(self.config.width, 0.5, 1.6))
        s_gain = np.full_like(cos_dphi, width_target, dtype=np.float32)

        if self.config.safe_mono:
            anti_phase_mask = cos_dphi < 0.0
            s_gain[anti_phase_mask] = np.clip(1.0 + cos_dphi[anti_phase_mask], 0.2, 1.0)

        S_new = S * s_gain
        ctx.Z[:, :, 0] = M + S_new
        ctx.Z[:, :, 1] = M - S_new

        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
