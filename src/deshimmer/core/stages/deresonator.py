"""Adaptive dynamic de-resonator stage for identifying and suppressing stationary spectral whistles."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

from ..context import DSPContext
from ..hpss import _apply_magnitude_inpaint
from ..params import Params
from ..utils import _frame_coeff
from .base import DSPStage

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def _numba_deq_persist(pos, n_cols, a_p):
    persist_map = np.zeros_like(pos)
    state_p = np.zeros(pos.shape[0], dtype=np.float32)
    for i in range(n_cols):
        state_p = np.float32(a_p) * state_p + (np.float32(1.0) - np.float32(a_p)) * pos[:, i]
        persist_map[:, i] = state_p
    return persist_map


@dataclass
class DeResonatorConfig:
    strength: float = 0.0
    start_hz: float = 180.0
    end_hz: float = 12000.0
    edge_hz: float = 150.0
    freq_med_bins: int = 31
    thr_db: float = 6.0
    slope: float = 0.7
    max_att_db: float = 8.0
    density_lo: float = 0.03
    density_hi: float = 0.20
    persist_ms: float = 600.0
    persist_thr_db: float = 2.5
    freq_smooth_bins: int = 5
    tonal_boost_db: float = 6.0
    time_floor: bool = False
    floor_smooth_ms: float = 80.0
    floor_rise_db_per_s: float = 1.0
    floor_thr_db: float = 3.0
    use_inpaint: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "DeResonatorConfig":
        use_inpaint = bool(getattr(p, "magnitude_inpaint", True))
        deq_inpaint = bool(getattr(p, "deq_inpaint", True)) and use_inpaint
        return cls(
            strength=float(getattr(p, "deres", 0.0)),
            start_hz=float(getattr(p, "deq_start_hz", 180.0)),
            end_hz=float(getattr(p, "deq_end_hz", 12000.0)),
            edge_hz=float(getattr(p, "deq_edge_hz", 150.0)),
            freq_med_bins=int(getattr(p, "deq_freq_med_bins", 31)),
            thr_db=float(getattr(p, "deq_thr_db", 6.0)),
            slope=float(getattr(p, "deq_slope", 0.7)),
            max_att_db=float(getattr(p, "deq_max_att_db", 8.0)),
            density_lo=float(getattr(p, "deq_density_lo", 0.03)),
            density_hi=float(getattr(p, "deq_density_hi", 0.20)),
            persist_ms=float(getattr(p, "deq_persist_ms", 600.0)),
            persist_thr_db=float(getattr(p, "deq_persist_thr_db", 2.5)),
            freq_smooth_bins=int(getattr(p, "deq_freq_smooth_bins", 5)),
            tonal_boost_db=float(getattr(p, "deq_tonal_boost_db", 6.0)),
            time_floor=bool(getattr(p, "deq_time_floor", False)),
            floor_smooth_ms=float(getattr(p, "deq_floor_smooth_ms", 80.0)),
            floor_rise_db_per_s=float(getattr(p, "deq_floor_rise_db_per_s", 1.0)),
            floor_thr_db=float(getattr(p, "deq_floor_thr_db", 3.0)),
            use_inpaint=deq_inpaint,
        )


class DeResonatorStage(DSPStage):
    name = "deresonator"

    def __init__(self, config: DeResonatorConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "DeResonatorStage":
        return cls(DeResonatorConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        deq_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        deres_str = float(np.clip(self.config.strength, 0.0, 1.0))
        if deres_str <= 1e-6 or deq_idx.size == 0:
            return

        L = np.log(ctx.Mag_mono[deq_idx, :] + ctx.eps)
        L_med = median_filter(L, size=(self.config.freq_med_bins, 1), mode='nearest')
        resid = (L - L_med) * (20.0 / np.log(10.0))

        thr_eff = self.config.thr_db + self.config.tonal_boost_db * (1.0 - ctx.w_noise_full)
        pos = np.maximum(0.0, resid - thr_eff[None, :])

        if self.config.time_floor:
            psd_deq = ctx.PSD[deq_idx, :].astype(np.float32, copy=False)
            a_floor = _frame_coeff(ctx.hop, ctx.sr, float(self.config.floor_smooth_ms))
            psd_floor_sm: np.ndarray = np.asarray(lfilter([1.0 - a_floor], [1.0, -a_floor], psd_deq, axis=1))

            rise_lin = float(10.0 ** ((float(self.config.floor_rise_db_per_s) * (float(ctx.hop) / float(ctx.sr))) / 10.0))
            floor_env = np.empty_like(psd_floor_sm)
            floor_env[:, 0] = psd_floor_sm[:, 0]
            for ti in range(1, ctx.n_cols):
                prev = floor_env[:, ti - 1]
                floor_env[:, ti] = np.minimum(prev * rise_lin, psd_floor_sm[:, ti])

            floor_win = max(4, round(ctx.sr * 1.0 / ctx.hop))
            floor_map = minimum_filter1d(floor_env, size=floor_win, axis=1, mode="nearest")
            floor_db = 10.0 * np.log10(floor_map + ctx.eps)
            local_floor_db = median_filter(floor_db, size=(self.config.freq_med_bins, 1), mode="nearest")
            floor_thr = float(self.config.floor_thr_db)
            floor_excess = np.maximum(0.0, floor_db - local_floor_db - floor_thr)
            pos = np.maximum(pos, floor_excess.astype(np.float32, copy=False))

        mask = pos > 0.0
        dens = np.mean(mask, axis=0)
        if self.config.time_floor:
            w_narrow = np.ones(ctx.n_cols, dtype=np.float32)
        else:
            w_narrow = 1.0 - np.clip(
                (dens - self.config.density_lo) / max(1e-6, self.config.density_hi - self.config.density_lo),
                0.0, 1.0,
            )

        a_p = _frame_coeff(ctx.hop, ctx.sr, self.config.persist_ms)
        if HAS_NUMBA:
            pos = pos.astype(np.float32)
            persist_map = _numba_deq_persist(pos, ctx.n_cols, a_p)
        else:
            persist_map = np.zeros_like(pos)
            state_p = np.zeros(pos.shape[0])
            for i in range(ctx.n_cols):
                state_p = a_p * state_p + (1.0 - a_p) * pos[:, i]
                persist_map[:, i] = state_p

        gate = np.clip(persist_map / max(1e-6, self.config.persist_thr_db), 0.0, 1.0)
        att_db = np.minimum(self.config.slope * pos * gate, self.config.max_att_db)
        gain = 10.0 ** (-att_db / 20.0)

        if self.config.freq_smooth_bins > 1:
            gain = uniform_filter1d(gain, size=self.config.freq_smooth_bins, axis=0, mode='nearest')

        hpss_deq = ctx.depth_scale(deq_idx)
        w_deq = ctx.edge_taper(deq_idx, self.config.start_hz, self.config.end_hz, self.config.edge_hz)
        depth_base = deres_str * ctx.w_nontrans * w_narrow
        depth_map = depth_base[None, :] * w_deq * hpss_deq
        conf_deq = np.clip(gate, 0.0, 1.0).astype(np.float32, copy=False)
        ctx.artifact_conf_parts.append(conf_deq)

        if self.config.use_inpaint:
            thr_lin = float(self.config.thr_db) * (np.log(10.0) / 20.0)
            target_mag = np.exp(L_med + thr_lin).astype(np.float32)
            repair_depth = depth_map * conf_deq
            _, g_eff = _apply_magnitude_inpaint(
                ctx.Z_orig[deq_idx, :, :],
                target_mag=target_mag,
                repair_depth=repair_depth,
                confidence=np.ones_like(conf_deq),
                eps=ctx.eps,
                ch_scales=ctx.ch_scales if ctx.ms_enabled else None,
            )
            ctx.g_total[deq_idx, :] *= g_eff
        else:
            g_eff = 1.0 - depth_map * (1.0 - gain)
            ctx.g_total[deq_idx, :] *= g_eff.astype(np.float32, copy=False)

        ctx.debug_maps["g_eff_deq_map"] = g_eff.astype(np.float32, copy=False)
        ctx.debug_maps["deq_depth_map"] = depth_base.astype(np.float32, copy=False)
