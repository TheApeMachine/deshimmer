"""Delivery mastering: DC-offset removal, high-pass filtering, loudness normalization, and true-peak limiting."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly, sosfiltfilt

from .params import MasterParams
from .measurements import measure_lufs, measure_rms_dbfs
from .utils import _as_2d, _butter_sos, _db_to_lin, _lin_to_db

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
def _numba_pdr_limiter(req: np.ndarray, sr: float, fast_ms: float, slow_ms: float) -> np.ndarray:
    """Numba-accelerated Program-Dependent Release (PDR) lookahead smoothing."""
    n = req.size
    g = np.ones(n, dtype=np.float32)
    c_fast = np.float32(math.exp(-1.0 / (sr * (fast_ms / 1000.0))))
    c_slow = np.float32(math.exp(-1.0 / (sr * (slow_ms / 1000.0))))
    env_fast = np.float32(1.0)
    env_slow = np.float32(1.0)
    for i in range(n):
        r = req[i]
        if r < env_fast:
            env_fast = r
        else:
            env_fast = c_fast * env_fast + (1.0 - c_fast) * r
        if r < env_slow:
            env_slow = r
        else:
            env_slow = c_slow * env_slow + (1.0 - c_slow) * r
        reduction = np.float32(1.0) - r
        w_slow = np.minimum(np.float32(1.0), reduction * np.float32(2.0))
        g_sm = (np.float32(1.0) - w_slow) * env_fast + w_slow * env_slow
        g[i] = np.minimum(r, g_sm)
    return g


def _peak_limiter(x: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float, release_ms: float) -> np.ndarray:
    eps = 1e-12
    x2 = _as_2d(x).astype(np.float32, copy=False)

    ceiling = float(_db_to_lin(float(ceiling_db)))
    ceiling = float(np.clip(ceiling, 1e-6, 1.0))

    sc = np.max(np.abs(x2), axis=1).astype(np.float32)

    la = int(float(sr) * (float(lookahead_ms) / 1000.0))
    la = max(0, la)
    size = la + 1
    if size <= 1:
        peak = sc
    else:
        origin = la // 2
        peak = maximum_filter1d(sc, size=size, origin=origin, mode="nearest").astype(np.float32)

    req = np.minimum(1.0, ceiling / (peak + eps)).astype(np.float32)

    fast_ms = 15.0
    slow_ms = max(80.0, float(release_ms))

    if HAS_NUMBA:
        g = _numba_pdr_limiter(req, float(sr), fast_ms, slow_ms)
    else:
        n = req.size
        g = np.ones(n, dtype=np.float32)
        c_fast = math.exp(-1.0 / (sr * (fast_ms / 1000.0)))
        c_slow = math.exp(-1.0 / (sr * (slow_ms / 1000.0)))
        env_fast = 1.0
        env_slow = 1.0
        for i in range(n):
            r = req[i]
            if r < env_fast:
                env_fast = r
            else:
                env_fast = c_fast * env_fast + (1.0 - c_fast) * r
            if r < env_slow:
                env_slow = r
            else:
                env_slow = c_slow * env_slow + (1.0 - c_slow) * r
            reduction = 1.0 - r
            w_slow = min(1.0, reduction * 2.0)
            g[i] = min(r, (1.0 - w_slow) * env_fast + w_slow * env_slow)

    return (x2 * g[:, None]).astype(np.float32, copy=False)


def loudness_normalize(x: np.ndarray, sr: int, mp: MasterParams, debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    x2 = _as_2d(np.asarray(x, dtype=np.float32))

    max_up = float(max(0.0, mp.norm_max_gain_db))
    max_dn = float(max(0.0, mp.norm_max_atten_db))

    gain_db: Optional[float] = None
    lufs = None
    if mp.target_lufs is not None:
        lufs = measure_lufs(x2, sr)
        if lufs is not None and math.isfinite(lufs):
            gain_db = float(mp.target_lufs) - float(lufs)
            info["lufs_in"] = float(lufs)
            info["target_lufs"] = float(mp.target_lufs)

    if gain_db is None:
        if mp.target_rms_dbfs is None:
            info["norm"] = "disabled"
            return x2, info
        rms_db = measure_rms_dbfs(x2)
        gain_db = float(mp.target_rms_dbfs) - float(rms_db)
        info["rms_dbfs_in"] = float(rms_db)
        info["target_rms_dbfs"] = float(mp.target_rms_dbfs)

    gain_db = float(np.clip(gain_db, -max_dn, +max_up))
    info["gain_db_applied"] = float(gain_db)

    g = float(_db_to_lin(gain_db))
    y = (x2 * g).astype(np.float32, copy=False)
    if debug:
        info["peak_after_norm_dbfs"] = float(_lin_to_db(np.max(np.abs(y)) + 1e-12))
    return y, info


def master_post(x: np.ndarray, sr: int, mp: MasterParams, debug: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": mp.enabled}
    if not mp.enabled:
        return x, info

    y = _as_2d(np.asarray(x, dtype=np.float32))

    if mp.dc_remove:
        y = (y - np.mean(y, axis=0, keepdims=True)).astype(np.float32, copy=False)
    info["dc_remove"] = mp.dc_remove

    if mp.hp_hz and mp.hp_hz > 0.0:
        nyq = 0.5 * sr
        if mp.hp_hz < nyq - 10.0:
            sos = _butter_sos(sr, "highpass", float(mp.hp_hz), order=max(1, mp.hp_order))
            y = sosfiltfilt(sos, y, axis=0).astype(np.float32, copy=False)
            info["hp_hz"] = float(mp.hp_hz)
            info["hp_order"] = mp.hp_order

    y, norm_info = loudness_normalize(y, sr, mp, debug=debug)
    info["normalize"] = norm_info

    osf = max(1, mp.os_factor)
    info["limiter_os_factor"] = osf
    if osf > 1:
        y_os = resample_poly(y, osf, 1, axis=0).astype(np.float32, copy=False)
        y_os = _peak_limiter(y_os, sr * osf, mp.ceiling_dbtp, mp.lookahead_ms, mp.release_ms)
        y2 = resample_poly(y_os, 1, osf, axis=0).astype(np.float32, copy=False)
        if y2.shape[0] > y.shape[0]:
            y2 = y2[: y.shape[0], :]
        elif y2.shape[0] < y.shape[0]:
            y2 = np.pad(y2, ((0, y.shape[0] - y2.shape[0]), (0, 0)), mode="constant")
        y = y2
    else:
        y = _peak_limiter(y, sr, mp.ceiling_dbtp, mp.lookahead_ms, mp.release_ms)

    info["ceiling_dbtp"] = float(mp.ceiling_dbtp)
    info["lookahead_ms"] = float(mp.lookahead_ms)
    info["release_ms"] = float(mp.release_ms)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999999:
        y = (y / peak * 0.999999).astype(np.float32)
        info["post_scale_db"] = float(_lin_to_db(0.999999 / peak))

    return y.squeeze(), info
