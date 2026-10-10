"""Stateful processing context for the STFT-domain DSP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .params import Params
from .psychoacoustic import _calculate_psychoacoustic_mask, apply_temporal_masking_decay
from .utils import _edge_taper


@dataclass
class DSPContext:
    Z: np.ndarray             # (F, T, C) complex64
    Z_orig: np.ndarray        # (F, T, C) complex64
    Mag: np.ndarray           # (F, T, C) float32
    Mag_mono: np.ndarray      # (F, T) float32
    PSD: np.ndarray           # (F, T) float32
    freqs: np.ndarray         # (F,) float32
    g_total: np.ndarray       # (F, T) float32 preallocated gain array (mutated in place)
    mask_psd: np.ndarray      # (F, T) float32 psychoacoustic mask
    flat_full: np.ndarray     # (T,) float32
    band_db_full: np.ndarray  # (T,) float32
    w_noise_full: np.ndarray  # (T,) float32
    flux: np.ndarray          # (T,) float32
    w_trans: np.ndarray       # (T,) float32
    w_nontrans: np.ndarray    # (T,) float32
    sr: int
    hop: int
    n_fft: int
    n_cols: int
    n_ch: int
    ms_enabled: bool
    ch_scales: np.ndarray
    eps: float = 1e-12
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    artifact_conf_parts: List[np.ndarray] = field(default_factory=list)
    debug_maps: Dict[str, Any] = field(default_factory=dict)
    depth_scale_for: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def idx(self, lo_hz: float, hi_hz: float) -> np.ndarray:
        nyq = float(self.freqs[-1]) if self.freqs.size else 0.5 * float(self.sr)
        lo = float(max(0.0, lo_hz))
        hi = float(min(nyq, hi_hz))
        if hi <= lo:
            return np.array([], dtype=np.int64)
        return np.where((self.freqs >= lo) & (self.freqs <= hi))[0]

    def edge_taper(self, band_idx: np.ndarray, lo_hz: float, hi_hz: float, edge_hz: float) -> np.ndarray:
        return _edge_taper(self.freqs, band_idx, lo_hz, hi_hz, edge_hz)[:, None]

    def depth_scale(self, band_idx: np.ndarray) -> np.ndarray:
        if self.depth_scale_for is not None:
            return self.depth_scale_for(band_idx)
        return np.ones((band_idx.size, self.n_cols), dtype=np.float32)

    @property
    def n_bins(self) -> int:
        return int(self.freqs.shape[0]) if self.freqs is not None else int(self.Z.shape[0])

    @property
    def n_frames(self) -> int:
        return self.n_cols

    @classmethod
    def create(
        cls,
        Z: np.ndarray,
        sr: int = 44100,
        hop: int = 512,
        n_fft: int = 2048,
        freqs: Optional[np.ndarray] = None,
        p: Optional[Params] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "DSPContext":
        if Z.ndim == 2:
            Z = Z[:, :, None]
        F, T, C = Z.shape
        eps = 1e-12
        if freqs is None:
            freqs = np.linspace(0.0, 0.5 * sr, F).astype(np.float32)
        Mag = np.abs(Z)
        Mag_mono = np.mean(Mag, axis=2)
        PSD = Mag_mono ** 2 + eps
        mask_psd = _calculate_psychoacoustic_mask(PSD, freqs)

        log_P = np.log(PSD + eps)
        flat_full = np.exp(np.mean(log_P, axis=0)) / (np.mean(PSD, axis=0) + eps)
        band_db_full = 10.0 * np.log10(np.mean(PSD, axis=0) + eps)

        f_start = float(getattr(p, "flat_start", 0.25)) if p else 0.25
        f_end = float(getattr(p, "flat_end", 0.70)) if p else 0.70
        fl_thr = float(getattr(p, "flux_thr_db", 6.0)) if p else 6.0
        fl_range = float(getattr(p, "flux_range_db", 8.0)) if p else 8.0

        w_noise_full = np.clip((flat_full - f_start) / max(1e-6, f_end - f_start), 0.0, 1.0)
        flux = np.zeros_like(band_db_full)
        flux[1:] = np.maximum(0.0, band_db_full[1:] - band_db_full[:-1])
        w_trans = np.clip((flux - fl_thr) / max(1e-6, fl_range), 0.0, 1.0)
        w_nontrans = 1.0 - w_trans

        if p is not None and getattr(p, "temporal_masking", True):
            mask_psd = apply_temporal_masking_decay(
                mask_psd,
                sr=sr,
                hop=hop,
                decay_ms=float(getattr(p, "mask_decay_ms", 15.0)),
                flatness=flat_full,
                flat_start=f_start,
                flat_end=f_end,
                gated=bool(getattr(p, "mask_flat_gate", True)),
            )

        ms_enabled = bool(getattr(p, "ms_process", False)) if p else False
        ch_scales = np.ones(C, dtype=np.float32)
        if ms_enabled and C >= 2:
            ch_scales[0] = 1.0
            ch_scales[1] = float(np.clip(getattr(p, "ms_side_scale", 0.35), 0.0, 1.0))

        if rng is None:
            rng = np.random.default_rng(int(getattr(p, "seed", 0)) if p else 0)

        g_total = np.ones((F, T), dtype=np.float32)
        return cls(
            Z=Z.copy(),
            Z_orig=Z.copy(),
            Mag=Mag,
            Mag_mono=Mag_mono,
            PSD=PSD,
            freqs=freqs,
            g_total=g_total,
            mask_psd=mask_psd,
            flat_full=flat_full,
            band_db_full=band_db_full,
            w_noise_full=w_noise_full,
            flux=flux,
            w_trans=w_trans,
            w_nontrans=w_nontrans,
            sr=sr,
            hop=hop,
            n_fft=n_fft,
            n_cols=T,
            n_ch=C,
            ms_enabled=ms_enabled,
            ch_scales=ch_scales,
            eps=eps,
            rng=rng,
        )
