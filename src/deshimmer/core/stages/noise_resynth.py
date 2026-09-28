"""Noise resynthesis (random-phase blending) in noise-like frames."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class NoiseResynthConfig:
    strength: float = 0.0
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0
    flat_start: float = 0.25
    flat_end: float = 0.70
    freq_med_bins: int = 9
    thr_db: float = 8.0
    density_lo: float = 0.02
    density_hi: float = 0.15

    @classmethod
    def from_params(cls, p: Params) -> "NoiseResynthConfig":
        return cls(
            strength=float(getattr(p, "noise_resynth", 0.0)),
            start_hz=float(getattr(p, "start_hz", 5100.0)),
            end_hz=float(getattr(p, "end_hz", 7200.0)),
            edge_hz=float(getattr(p, "edge_hz", 200.0)),
            flat_start=float(getattr(p, "flat_start", 0.25)),
            flat_end=float(getattr(p, "flat_end", 0.70)),
            freq_med_bins=int(getattr(p, "freq_med_bins", 9)),
            thr_db=float(getattr(p, "thr_db", 8.0)),
            density_lo=float(getattr(p, "density_lo", 0.02)),
            density_hi=float(getattr(p, "density_hi", 0.15)),
        )


class NoiseResynthStage(DSPStage):
    name = "noise_resynth"

    def __init__(self, config: NoiseResynthConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "NoiseResynthStage":
        return cls(NoiseResynthConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 0.0

    def process(self, ctx: DSPContext) -> None:
        sh_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        if sh_idx.size < 8 or self.config.strength <= 0.0:
            return

        nr = float(self.config.strength)
        log_P = np.log(ctx.Mag_mono[sh_idx, :] ** 2 + ctx.eps)
        flat_sh = np.exp(np.mean(log_P, axis=0)) / (np.mean(ctx.Mag_mono[sh_idx, :] ** 2, axis=0) + ctx.eps)
        w_noise_sh = np.clip(
            (flat_sh - self.config.flat_start) / max(1e-6, self.config.flat_end - self.config.flat_start),
            0.0, 1.0,
        )
        L = np.log(ctx.Mag_mono[sh_idx, :] + ctx.eps)
        L_med = median_filter(L, size=(self.config.freq_med_bins, 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))
        mask = resid > self.config.thr_db
        dens = np.mean(mask, axis=0)
        w_narrow = 1.0 - np.clip(
            (dens - self.config.density_lo) / max(1e-6, self.config.density_hi - self.config.density_lo),
            0.0, 1.0,
        )
        hpss_sh = ctx.depth_scale(sh_idx)
        w_sh = ctx.edge_taper(sh_idx, self.config.start_hz, self.config.end_hz, self.config.edge_hz)
        depth_map = (nr * w_noise_sh * ctx.w_nontrans * w_narrow)[None, :] * w_sh * hpss_sh

        phases = ctx.rng.uniform(0.0, 2.0 * np.pi, size=ctx.Z[sh_idx].shape[:2]).astype(np.float32)
        zph = (np.cos(phases) + 1j * np.sin(phases))[:, :, None]
        d = depth_map[:, :, None]
        if ctx.ms_enabled:
            d = d * ctx.ch_scales.astype(np.float32)[None, None, :]
        ctx.Z[sh_idx] = (1.0 - d) * ctx.Z[sh_idx] + d * (np.abs(ctx.Z[sh_idx]) * zph)
