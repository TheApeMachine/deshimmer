"""Downward expander stage for sub-threshold noise gating in high frequency bands."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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
def _numba_exp_env(g_target, n_cols, att, rel):
    g_env = np.ones_like(g_target)
    g_sm = np.float32(1.0)
    for i in range(n_cols):
        curr = g_target[i]
        if curr < g_sm:
            g_sm = np.float32(att) * g_sm + (np.float32(1.0) - np.float32(att)) * curr
        else:
            g_sm = np.float32(rel) * g_sm + (np.float32(1.0) - np.float32(rel)) * curr
        g_env[i] = g_sm
    return g_env


@dataclass
class ExpanderConfig:
    enabled: bool = False
    start_hz: float = 3000.0
    end_hz: float = 8000.0
    threshold_db: float = -45.0
    ratio: float = 2.0
    attack_ms: float = 10.0
    release_ms: float = 150.0

    @classmethod
    def from_params(cls, p: Params) -> "ExpanderConfig":
        return cls(
            enabled=bool(getattr(p, "expander", False)),
            start_hz=float(getattr(p, "exp_start_hz", 3000.0)),
            end_hz=float(getattr(p, "exp_end_hz", 8000.0)),
            threshold_db=float(getattr(p, "exp_threshold_db", -45.0)),
            ratio=float(max(1.0, getattr(p, "exp_ratio", 2.0))),
            attack_ms=float(getattr(p, "exp_attack_ms", 10.0)),
            release_ms=float(getattr(p, "exp_release_ms", 150.0)),
        )


class ExpanderStage(DSPStage):
    name = "expander"

    def __init__(self, config: ExpanderConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "ExpanderStage":
        return cls(ExpanderConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enabled

    def process(self, ctx: DSPContext) -> None:
        exp_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        if not (self.config.enabled and exp_idx.size > 0):
            return

        band_p_exp = np.mean(ctx.PSD[exp_idx, :], axis=0)
        band_db_exp = 10.0 * np.log10(band_p_exp + ctx.eps)
        thr = float(self.config.threshold_db)
        ratio = float(self.config.ratio)

        red_db = np.zeros_like(band_db_exp)
        mask = band_db_exp < thr
        red_db[mask] = (thr - band_db_exp[mask]) * (ratio - 1.0)
        g_target = 10.0 ** (-red_db / 20.0)

        att = _frame_coeff(ctx.hop, ctx.sr, self.config.attack_ms)
        rel = _frame_coeff(ctx.hop, ctx.sr, self.config.release_ms)

        if HAS_NUMBA:
            g_target = g_target.astype(np.float32)
            g_env = _numba_exp_env(g_target, ctx.n_cols, att, rel)
        else:
            g_env = np.ones_like(g_target)
            g_sm = 1.0
            for i in range(ctx.n_cols):
                curr = g_target[i]
                if curr < g_sm:
                    g_sm = att * g_sm + (1.0 - att) * curr
                else:
                    g_sm = rel * g_sm + (1.0 - rel) * curr
                g_env[i] = g_sm

        ctx.g_total[exp_idx, :] *= g_env.astype(np.float32, copy=False)
