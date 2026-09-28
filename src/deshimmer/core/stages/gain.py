"""Gain application stage: combines attenuation floors, psychoacoustic caps, and applies M/S or stereo gains."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class GainApplicationConfig:
    cap_db: float = 12.0
    nuclear_mode: bool = False
    ms_enabled: bool = False
    ms_side_scale: float = 0.35
    exp_start_hz: float = 3000.0
    exp_end_hz: float = 8000.0
    dn_start_hz: float = 120.0
    dn_end_hz: float = 16000.0
    deq_start_hz: float = 180.0
    deq_end_hz: float = 12000.0
    sh_start_hz: float = 5100.0
    sh_end_hz: float = 7200.0

    @classmethod
    def from_params(cls, p: Params) -> "GainApplicationConfig":
        cap_db = 48.0 if bool(getattr(p, "nuclear_mode", False)) else float(getattr(p, "total_att_cap_db", 12.0))
        return cls(
            cap_db=cap_db,
            nuclear_mode=bool(getattr(p, "nuclear_mode", False)),
            ms_enabled=bool(getattr(p, "ms_process", False)),
            ms_side_scale=float(getattr(p, "ms_side_scale", 0.35)),
            exp_start_hz=float(getattr(p, "exp_start_hz", 3000.0)),
            exp_end_hz=float(getattr(p, "exp_end_hz", 8000.0)),
            dn_start_hz=float(getattr(p, "dn_start_hz", 120.0)),
            dn_end_hz=float(getattr(p, "dn_end_hz", 16000.0)),
            deq_start_hz=float(getattr(p, "deq_start_hz", 180.0)),
            deq_end_hz=float(getattr(p, "deq_end_hz", 12000.0)),
            sh_start_hz=float(getattr(p, "start_hz", 5100.0)),
            sh_end_hz=float(getattr(p, "end_hz", 7200.0)),
        )


class GainApplicationStage(DSPStage):
    name = "gain_application"

    def __init__(self, config: GainApplicationConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "GainApplicationStage":
        return cls(GainApplicationConfig.from_params(p))

    def is_enabled(self) -> bool:
        return True

    def process(self, ctx: DSPContext) -> None:
        exp_idx = ctx.idx(self.config.exp_start_hz, self.config.exp_end_hz)
        dn_idx = ctx.idx(self.config.dn_start_hz, self.config.dn_end_hz)
        deq_idx = ctx.idx(self.config.deq_start_hz, self.config.deq_end_hz)
        sh_idx = ctx.idx(self.config.sh_start_hz, self.config.sh_end_hz)

        proc_idx = np.unique(np.concatenate([
            arr for arr in [exp_idx, dn_idx, deq_idx, sh_idx] if arr.size > 0
        ])).astype(np.int64) if (exp_idx.size or dn_idx.size or deq_idx.size or sh_idx.size) else np.array([], dtype=np.int64)

        g_min = float(10.0 ** (-max(0.0, self.config.cap_db) / 20.0))

        if proc_idx.size:
            ctx.g_total[proc_idx, :] = np.maximum(ctx.g_total[proc_idx, :], g_min)

            g_min_allowed = np.minimum(1.0, np.sqrt(ctx.mask_psd[proc_idx, :] / (ctx.PSD[proc_idx, :] + ctx.eps)))
            ctx.g_total[proc_idx, :] = np.maximum(ctx.g_total[proc_idx, :], g_min_allowed)

            ctx.Z = ctx.Z_orig.copy()
            g_apply = ctx.g_total[proc_idx, :].astype(np.float32, copy=False)

            if ctx.ms_enabled and ctx.n_ch >= 2:
                left = ctx.Z_orig[proc_idx, :, 0]
                right = ctx.Z_orig[proc_idx, :, 1]
                mid = 0.5 * (left + right)
                side = 0.5 * (left - right)

                phase_diff = np.angle(side + ctx.eps) - np.angle(mid + ctx.eps)
                mid_new = mid * g_apply

                s_scale = float(self.config.ms_side_scale)
                g_side = 1.0 - (1.0 - g_apply) * s_scale
                side_mag_new = np.abs(side) * g_side

                side_new = side_mag_new * np.exp(1j * (np.angle(mid_new + ctx.eps) + phase_diff))

                ctx.Z[proc_idx, :, 0] = mid_new + side_new
                ctx.Z[proc_idx, :, 1] = mid_new - side_new

                if ctx.n_ch > 2:
                    for ci in range(2, ctx.n_ch):
                        ctx.Z[proc_idx, :, ci] = ctx.Z_orig[proc_idx, :, ci] * g_apply
            else:
                ctx.Z[proc_idx, :, :] = ctx.Z_orig[proc_idx, :, :] * g_apply[:, :, None]
        else:
            ctx.Z = ctx.Z_orig.copy()
