"""Smart spectral denoise using minimum-statistics noise floor tracking and Wiener gain."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from ..context import DSPContext
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
def _numba_dn_noise_est(curr_min, n_blocks, curr_noise_vec, rise):
    noise_est_blocks = np.zeros_like(curr_min)
    for b in range(n_blocks):
        M = curr_min[:, b]
        curr_noise_vec = np.minimum(M, curr_noise_vec * np.float32(rise))
        noise_est_blocks[:, b] = curr_noise_vec
    return noise_est_blocks


@dataclass
class SmartDenoiseConfig:
    strength: float = 0.0
    start_hz: float = 120.0
    end_hz: float = 16000.0
    edge_hz: float = 200.0
    floor_db: float = -18.0
    psd_smooth_ms: float = 50.0
    minwin_ms: float = 400.0
    up_db_per_s: float = 3.0
    freq_smooth_bins: int = 3

    @classmethod
    def from_params(cls, p: Params) -> "SmartDenoiseConfig":
        return cls(
            strength=float(getattr(p, "denoise", 0.0)),
            start_hz=float(getattr(p, "dn_start_hz", 120.0)),
            end_hz=float(getattr(p, "dn_end_hz", 16000.0)),
            edge_hz=float(getattr(p, "dn_edge_hz", 200.0)),
            floor_db=float(getattr(p, "dn_floor_db", -18.0)),
            psd_smooth_ms=float(getattr(p, "dn_psd_smooth_ms", 50.0)),
            minwin_ms=float(getattr(p, "dn_minwin_ms", 400.0)),
            up_db_per_s=float(getattr(p, "dn_up_db_per_s", 3.0)),
            freq_smooth_bins=int(getattr(p, "dn_freq_smooth_bins", 3)),
        )


class SmartDenoiseStage(DSPStage):
    name = "smart_denoise"

    def __init__(self, config: SmartDenoiseConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "SmartDenoiseStage":
        return cls(SmartDenoiseConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        dn_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        dn_str = float(np.clip(self.config.strength, 0.0, 1.0))
        if dn_str <= 1e-6 or dn_idx.size == 0:
            return

        psd_dn = ctx.PSD[dn_idx, :]
        a_psd = _frame_coeff(ctx.hop, ctx.sr, self.config.psd_smooth_ms)
        psd_sm_map: np.ndarray = np.asarray(lfilter([1.0 - a_psd], [1.0, -a_psd], psd_dn, axis=1))

        minwin = max(4, int(ctx.sr * self.config.minwin_ms / 1000.0 / ctx.hop))
        n_blocks = int(np.ceil(ctx.n_cols / minwin))
        pad_t = n_blocks * minwin - ctx.n_cols
        p_padded = np.pad(psd_sm_map, ((0, 0), (0, pad_t)), constant_values=np.inf)
        p_reshaped = p_padded.reshape(psd_dn.shape[0], n_blocks, minwin)
        curr_min = np.min(p_reshaped, axis=2)

        rise = 10.0 ** ((self.config.up_db_per_s * (minwin * ctx.hop / ctx.sr)) / 10.0)
        curr_noise_vec = psd_dn[:, 0]
        if HAS_NUMBA:
            curr_min = curr_min.astype(np.float32)
            curr_noise_vec = curr_noise_vec.astype(np.float32)
            noise_est_blocks = _numba_dn_noise_est(curr_min, n_blocks, curr_noise_vec, rise)
        else:
            noise_est_blocks = np.zeros_like(curr_min)
            for b in range(n_blocks):
                M = curr_min[:, b]
                curr_noise_vec = np.minimum(M, curr_noise_vec * rise)
                noise_est_blocks[:, b] = curr_noise_vec

        noise_map = np.repeat(noise_est_blocks, minwin, axis=1)[:, :ctx.n_cols]

        F_dn, T_dn = psd_dn.shape
        dn_g = np.ones((F_dn, ctx.n_cols), dtype=np.float32)
        alpha_dd = 0.98
        prev_clean_psd = psd_dn[:, 0]

        floor = 10.0 ** (self.config.floor_db / 20.0)
        min_xi = 10.0 ** (-max(12.0, abs(self.config.floor_db)) / 10.0)

        for ti in range(ctx.n_cols):
            noise_psd = noise_map[:, ti] + ctx.eps
            current_psd = psd_dn[:, ti]

            gamma = current_psd / noise_psd
            xi = alpha_dd * (prev_clean_psd / noise_psd) + (1.0 - alpha_dd) * np.maximum(0.0, gamma - 1.0)
            xi = np.maximum(xi, min_xi)

            g_inst_frame = xi / (xi + 1.0)
            g_inst_frame = np.clip(floor + (1.0 - floor) * g_inst_frame, floor, 1.0)

            dn_g[:, ti] = g_inst_frame
            prev_clean_psd = (g_inst_frame ** 2) * current_psd

        if self.config.freq_smooth_bins > 1:
            dn_g = uniform_filter1d(dn_g, size=self.config.freq_smooth_bins, axis=0, mode='nearest')

        hpss_dn = ctx.depth_scale(dn_idx)
        w_dn = ctx.edge_taper(dn_idx, self.config.start_hz, self.config.end_hz, self.config.edge_hz)
        depth_base = dn_str * (0.5 + 0.5 * ctx.w_noise_full) * ctx.w_nontrans
        depth_map = depth_base[None, :] * w_dn * hpss_dn
        g_eff = 1.0 - depth_map * (1.0 - dn_g)

        ctx.g_total[dn_idx, :] *= g_eff.astype(np.float32, copy=False)
        ctx.debug_maps["g_eff_dn_map"] = g_eff.astype(np.float32, copy=False)
        ctx.debug_maps["dn_depth_map"] = depth_base.astype(np.float32, copy=False)
        ctx.debug_maps["noise_psd_dn_db_map"] = (10.0 * np.log10(noise_map + ctx.eps)).astype(np.float32, copy=False)
