"""High-frequency harmonic exciter / resynthesis stage."""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.signal import sosfilt, sosfiltfilt

from .params import Params
from .utils import _as_2d, _butter_sos


def hf_resynth_post(
    x: np.ndarray,
    sr: int,
    p: Params,
    *,
    artifact_conf: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Nonlinear harmonic generator with confidence-based blending for upper high-frequencies."""
    if not bool(getattr(p, "hf_resynth", False)):
        return x
    x2 = _as_2d(np.asarray(x, dtype=np.float32))
    nyq = 0.5 * float(sr)

    lp_hz = float(np.clip(getattr(p, "hf_lp_hz", 3000.0), 20.0, nyq - 100.0))
    src_lo = float(np.clip(getattr(p, "hf_src_lo_hz", 1000.0), 20.0, nyq - 100.0))
    src_hi = float(np.clip(getattr(p, "hf_src_hi_hz", 2000.0), src_lo + 10.0, nyq - 100.0))
    hp_hz = float(np.clip(getattr(p, "hf_hp_hz", 3000.0), 20.0, nyq - 100.0))
    drive = float(max(0.1, getattr(p, "hf_drive", 2.0)))
    mix = float(np.clip(getattr(p, "hf_mix", 0.35), 0.0, 1.0))
    conf_blend = bool(getattr(p, "hf_confidence_blend", True))

    zero_phase = bool(getattr(p, "hf_zero_phase", True))
    tilt_lp_hz = float(np.clip(getattr(p, "hf_tilt_lp_hz", 12000.0), hp_hz + 50.0, nyq - 50.0))

    def lr4_sos(kind: str, hz: float):
        s = _butter_sos(sr, kind, float(hz), order=2)
        return np.concatenate([np.asarray(s), np.asarray(s)], axis=0)

    def _sos_apply(sos, sig):
        y = sosfilt(sos, sig, axis=0)
        return y[0] if isinstance(y, tuple) else y

    sos_lp = lr4_sos("lowpass", lp_hz)
    sos_hp = lr4_sos("highpass", hp_hz)

    if zero_phase:
        base = sosfiltfilt(sos_lp, x2, axis=0).astype(np.float32, copy=False)
    else:
        base = _sos_apply(sos_lp, x2).astype(np.float32, copy=False)

    sos_bp = _butter_sos(sr, "bandpass", (src_lo, src_hi), order=2)
    if zero_phase:
        src = sosfiltfilt(sos_bp, x2, axis=0).astype(np.float32, copy=False)
    else:
        src = _sos_apply(sos_bp, x2).astype(np.float32, copy=False)
    gen = np.tanh(src * drive).astype(np.float32, copy=False)

    if zero_phase:
        gen_hf = sosfiltfilt(sos_hp, gen, axis=0).astype(np.float32, copy=False)
    else:
        gen_hf = _sos_apply(sos_hp, gen).astype(np.float32, copy=False)

    if tilt_lp_hz < nyq - 50.0:
        sos_tilt = _butter_sos(sr, "lowpass", tilt_lp_hz, order=1)
        if zero_phase:
            gen_hf = sosfiltfilt(sos_tilt, gen_hf, axis=0).astype(np.float32, copy=False)
        else:
            gen_hf = _sos_apply(sos_tilt, gen_hf).astype(np.float32, copy=False)

    y_full = base + mix * gen_hf

    if not conf_blend:
        return y_full.squeeze()

    conf_frames = artifact_conf
    if conf_frames is None or conf_frames.size == 0:
        return y_full.squeeze()

    hop = p.hop
    n_samp = x2.shape[0]
    conf_t = np.mean(conf_frames, axis=0).astype(np.float32)
    t_frames = np.arange(conf_t.size, dtype=np.float32) * (hop / float(sr))
    t_samp = np.arange(n_samp, dtype=np.float32) / float(sr)
    conf_s = np.interp(
        t_samp,
        t_frames,
        conf_t,
        left=float(conf_t[0] if conf_t.size else 0.0),
        right=float(conf_t[-1] if conf_t.size else 0.0),
    ).astype(np.float32)
    conf_s = np.clip(conf_s, 0.0, 1.0)[:, None]
    y_orig_hf = x2 - base
    y = base + conf_s * (mix * gen_hf) + (1.0 - conf_s) * y_orig_hf
    return y.squeeze()
