"""
deshimmer_api.py

Small callable API extracted from master.py so other tools (UI/tests) can reuse
the processing pipeline without shelling out to the CLI.

This intentionally keeps the implementation in master.py as the "source of
truth" and only re-exports the types/functions we need.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

# master.py contains the full implementation; we re-use it to avoid divergence.
import master as _m


def process_audio(
    x: np.ndarray,
    sr: int,
    *,
    params: Optional[_m.Params] = None,
    master_params: Optional[_m.MasterParams] = None,
    debug_params: Optional[_m.DebugParams] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process audio and return (y_out, info).

    - x: (N,) or (N, C) float audio (expects -1..1-ish)
    - sr: sample rate
    - params/master_params/debug_params: optional dataclass instances

    info includes measurements and (if debug enabled) a debug_dir or debug blobs.
    """
    if sr <= 0:
        raise ValueError("sr must be > 0")

    x2 = np.asarray(x, dtype=np.float32)
    if x2.ndim == 1:
        x2 = x2[:, None]
    elif x2.ndim != 2:
        raise ValueError("x must be 1D or 2D")

    p = params if params is not None else _m.Params()
    mp = master_params if master_params is not None else _m.MasterParams(enabled=False)
    dp = debug_params if debug_params is not None else _m.DebugParams(enabled=False)

    # ---- Measurements (input) ----
    info: Dict[str, Any] = {
        "sr": int(sr),
        "channels": int(x2.shape[1]),
        "duration_s": float(x2.shape[0] / sr),
        "params": asdict(p),
        "master_params": asdict(mp),
        "debug_params": asdict(dp),
    }
    info["measure_in"] = {
        "sample_peak_dbfs": float(_m._lin_to_db(np.max(np.abs(x2)) + 1e-12)),
        "true_peak_dbtp": float(_m.measure_true_peak_db(x2, os_factor=max(1, int(mp.os_factor)))),
        "rms_dbfs": float(_m.measure_rms_dbfs(x2)),
        "lufs": _m.measure_lufs(x2, sr),
    }

    # ---- Optional debug collector ----
    dbg_collector = None
    if dp.enabled:
        n_fft = int(p.n_fft)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        nyq = float(freqs[-1])

        def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
            lo = float(max(0.0, lo_hz))
            hi = float(min(nyq, hi_hz))
            if hi <= lo:
                return np.array([], dtype=np.int64)
            return np.where((freqs >= lo) & (freqs <= hi))[0]

        sh_idx = _idx(p.start_hz, p.end_hz)
        dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
        deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)
        pad_offset = n_fft if p.pad else 0
        dbg_collector = _m.DebugCollector(
            sr=sr,
            freqs=freqs,
            dn_idx=dn_idx,
            deq_idx=deq_idx,
            sh_idx=sh_idx,
            stride=int(max(1, dp.stride)),
            pad_offset=pad_offset,
            duration_s=float(x2.shape[0] / sr),
        )

    # ---- STFT repair ----
    y_repaired = _m.process_stft(x2, sr, p, dbg=dbg_collector)
    y_rep_2d = _m._as_2d(y_repaired)
    info["measure_after_repair"] = {
        "sample_peak_dbfs": float(_m._lin_to_db(np.max(np.abs(y_rep_2d)) + 1e-12)),
        "true_peak_dbtp": float(_m.measure_true_peak_db(y_rep_2d, os_factor=max(1, int(mp.os_factor)))),
        "rms_dbfs": float(_m.measure_rms_dbfs(y_rep_2d)),
        "lufs": _m.measure_lufs(y_rep_2d, sr),
    }

    # ---- Optional master ----
    y_out, master_info = _m.master_post(y_repaired, sr, mp, debug=dp.enabled)
    y_out_2d = _m._as_2d(y_out)
    info["measure_out"] = {
        "sample_peak_dbfs": float(_m._lin_to_db(np.max(np.abs(y_out_2d)) + 1e-12)),
        "true_peak_dbtp": float(_m.measure_true_peak_db(y_out_2d, os_factor=max(1, int(mp.os_factor)))),
        "rms_dbfs": float(_m.measure_rms_dbfs(y_out_2d)),
        "lufs": _m.measure_lufs(y_out_2d, sr),
    }
    info["master_info"] = master_info

    dbg_data: Dict[str, Any] = {}
    if dp.enabled and dbg_collector is not None:
        dbg_data = dbg_collector.finalize()
        info["dbg_data_keys"] = sorted(list(dbg_data.keys()))
        # Note: rendering PNGs is intentionally left to callers (UI can choose).
    return y_out_2d.squeeze(), info


