"""Target-band shimmer and birdie suppression stage using residual thresholding."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter

from ..context import DSPContext
from ..hpss import _apply_magnitude_inpaint
from ..params import Params
from .base import DSPStage


@dataclass
class ShimmerConfig:
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0
    flat_start: float = 0.25
    flat_end: float = 0.70
    freq_med_bins: int = 9
    thr_db: float = 8.0
    slope: float = 0.6
    density_lo: float = 0.02
    density_hi: float = 0.15
    use_inpaint: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "ShimmerConfig":
        return cls(
            start_hz=float(getattr(p, "start_hz", 5100.0)),
            end_hz=float(getattr(p, "end_hz", 7200.0)),
            edge_hz=float(getattr(p, "edge_hz", 200.0)),
            flat_start=float(getattr(p, "flat_start", 0.25)),
            flat_end=float(getattr(p, "flat_end", 0.70)),
            freq_med_bins=int(getattr(p, "freq_med_bins", 9)),
            thr_db=float(getattr(p, "thr_db", 8.0)),
            slope=float(getattr(p, "slope", 0.6)),
            density_lo=float(getattr(p, "density_lo", 0.02)),
            density_hi=float(getattr(p, "density_hi", 0.15)),
            use_inpaint=bool(getattr(p, "magnitude_inpaint", True)),
        )


class ShimmerStage(DSPStage):
    name = "shimmer"

    def __init__(self, config: ShimmerConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "ShimmerStage":
        return cls(ShimmerConfig.from_params(p))

    def is_enabled(self) -> bool:
        return True

    def process(self, ctx: DSPContext) -> None:
        sh_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        if sh_idx.size < 8:
            return

        mag_sh = ctx.Mag_mono[sh_idx, :]
        log_P = np.log(mag_sh ** 2 + ctx.eps)
        flat_sh = np.exp(np.mean(log_P, axis=0)) / (np.mean(mag_sh ** 2, axis=0) + ctx.eps)
        w_noise_sh = np.clip(
            (flat_sh - self.config.flat_start) / max(1e-6, self.config.flat_end - self.config.flat_start),
            0.0, 1.0,
        )

        L = np.log(mag_sh + ctx.eps)
        L_med = median_filter(L, size=(self.config.freq_med_bins, 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))

        mask = resid > self.config.thr_db
        dens = np.mean(mask, axis=0)
        w_narrow = 1.0 - np.clip(
            (dens - self.config.density_lo) / max(1e-6, self.config.density_hi - self.config.density_lo),
            0.0, 1.0,
        )

        att_db = np.zeros_like(resid)
        m2 = resid > self.config.thr_db
        att_db[m2] = self.config.slope * (resid[m2] - self.config.thr_db)
        gain = 10.0 ** (-att_db / 20.0)

        hpss_sh = ctx.depth_scale(sh_idx)
        w_sh = ctx.edge_taper(sh_idx, self.config.start_hz, self.config.end_hz, self.config.edge_hz)
        depth_base = w_noise_sh * ctx.w_nontrans * w_narrow
        depth_map = depth_base[None, :] * w_sh * hpss_sh
        conf_sh = np.clip(
            (resid - float(self.config.thr_db)) / max(1e-6, float(self.config.thr_db)),
            0.0, 1.0,
        ).astype(np.float32)
        ctx.artifact_conf_parts.append(conf_sh)

        if self.config.use_inpaint:
            thr_lin = float(self.config.thr_db) * (np.log(10.0) / 20.0)
            target_mag = np.exp(L_med + thr_lin).astype(np.float32)
            repair_depth = depth_map * conf_sh
            _, g_eff = _apply_magnitude_inpaint(
                ctx.Z_orig[sh_idx, :, :],
                target_mag=target_mag,
                repair_depth=repair_depth,
                confidence=np.ones_like(conf_sh),
                eps=ctx.eps,
                ch_scales=ctx.ch_scales if ctx.ms_enabled else None,
            )
            ctx.g_total[sh_idx, :] *= g_eff
        else:
            g_eff = 1.0 - depth_map * (1.0 - gain)
            ctx.g_total[sh_idx, :] *= g_eff.astype(np.float32, copy=False)

        ctx.debug_maps["g_eff_sh_map"] = g_eff.astype(np.float32, copy=False)
        ctx.debug_maps["sh_depth_map"] = depth_base.astype(np.float32, copy=False)
