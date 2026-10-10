"""DSP pipeline execution engine and core STFT-domain repair coordinator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import sys
import numpy as np
from scipy.signal import istft, stft

from ..tonal import TonalRepair
from .context import DSPContext
from .debug import DebugCollector
from .hpss import _hpss_depth_scale, _hpss_mask_for_band
from .params import Params
from .psychoacoustic import _calculate_psychoacoustic_mask, apply_temporal_masking_decay
from .resynth import hf_resynth_post
from .spatial import align_lr_delay, balance_lr_spectrum
from .stages import (
    CarverConfig,
    DeResonatorConfig,
    DeResonatorStage,
    DSPStage,
    DynamicCarverStage,
    ExpanderConfig,
    ExpanderStage,
    GainApplicationConfig,
    GainApplicationStage,
    HarmonicEnhanceConfig,
    HarmonicEnhanceStage,
    HFDecorrelationConfig,
    HFDecorrelationStage,
    LowEndConfig,
    LowEndStage,
    NoiseResynthConfig,
    NoiseResynthStage,
    PhaseBlurConfig,
    PhaseBlurStage,
    ShimmerConfig,
    ShimmerStage,
    SmartDenoiseConfig,
    SmartDenoiseStage,
    StereoTranslationConfig,
    StereoTranslationStage,
    SwishRepairConfig,
    SwishRepairStage,
    TonalBalanceConfig,
    TonalBalanceStage,
    TransientPunchConfig,
    TransientPunchStage,
)
from .utils import _as_2d

# Last STFT-frame artifact confidence for HF resynth blending
_LAST_ARTIFACT_CONF_FRAMES: Optional[np.ndarray] = None


class DSPPipeline:
    def __init__(self, stages: Optional[List[DSPStage]] = None):
        self.stages: List[DSPStage] = list(stages) if stages is not None else []

    def add_stage(self, stage: DSPStage) -> "DSPPipeline":
        self.stages.append(stage)
        return self

    def get_stage(self, name: str) -> Optional[DSPStage]:
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def process(self, ctx: DSPContext) -> None:
        for stage in self.stages:
            if stage.is_enabled():
                stage.process(ctx)


def build_default_pipeline(p: Params) -> DSPPipeline:
    stages: List[DSPStage] = [
        ExpanderStage.from_params(p),
        SmartDenoiseStage.from_params(p),
        DeResonatorStage.from_params(p),
        ShimmerStage.from_params(p),
        GainApplicationStage.from_params(p),
        DynamicCarverStage.from_params(p),
        NoiseResynthStage.from_params(p),
        PhaseBlurStage.from_params(p),
        SwishRepairStage.from_params(p),
        HFDecorrelationStage.from_params(p),
    ]
    goodizer_stages: List[DSPStage] = [
        LowEndStage.from_params(p),
        TonalBalanceStage.from_params(p),
        TransientPunchStage.from_params(p),
        StereoTranslationStage.from_params(p),
        HarmonicEnhanceStage.from_params(p),
    ]
    for s in goodizer_stages:
        if s.is_enabled():
            stages.append(s)
    return DSPPipeline(stages)


def process_stft(
    x: np.ndarray,
    sr: int,
    p: Params,
    dbg: Optional[DebugCollector] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    *,
    pipeline: Optional[DSPPipeline] = None,
) -> np.ndarray:
    """Repair the wet signal; mix and delta always reference the original samples."""
    global _LAST_ARTIFACT_CONF_FRAMES
    _LAST_ARTIFACT_CONF_FRAMES = None

    if x.ndim == 1:
        x = x[:, None]

    dry = x

    if p.mix == 0.0:
        if diagnostics is not None:
            diagnostics["tonal"] = {"enabled": True, "status": "bypassed", "changed_partials": 0}

        return (np.zeros_like(dry) if p.delta_listen else dry.copy()).squeeze()

    # --- High-Fidelity Step 1: Align Left/Right micro-delays (Time Domain) ---
    if p.enhance:
        x = align_lr_delay(x, sr)

    x_t = x.T
    n_ch, n_samples = x_t.shape

    n_fft = p.n_fft
    hop = p.hop

    f, t_sec, Z = stft(x_t, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', padded=True, axis=-1)
    Z = Z.transpose(1, 2, 0).astype(np.complex64)

    # --- High-Fidelity Step 2: Symmetric L/R spectral balancing (Spectral Domain) ---
    if p.enhance:
        Z = balance_lr_spectrum(Z)

    freqs = f.astype(np.float32)
    n_cols = Z.shape[1]
    eps = 1e-12
    rng = np.random.default_rng(int(getattr(p, "seed", 0)))

    # --- Base Magnitude / PSD ---
    Mag = np.abs(Z)
    Mag_mono = np.mean(Mag, axis=2)
    PSD = Mag_mono ** 2 + eps

    # --- Enforce Psychoacoustic transparency ---
    mask_psd = _calculate_psychoacoustic_mask(PSD, freqs)

    # --- Global Features ---
    log_P = np.log(PSD + eps)
    flat_full = np.exp(np.mean(log_P, axis=0)) / (np.mean(PSD, axis=0) + eps)
    band_db_full = 10.0 * np.log10(np.mean(PSD, axis=0) + eps)

    w_noise_full = np.clip((flat_full - p.flat_start) / max(1e-6, p.flat_end - p.flat_start), 0.0, 1.0)

    flux = np.zeros_like(band_db_full)
    flux[1:] = np.maximum(0.0, band_db_full[1:] - band_db_full[:-1])
    w_trans = np.clip((flux - p.flux_thr_db) / max(1e-6, p.flux_range_db), 0.0, 1.0)
    w_nontrans = 1.0 - w_trans

    # --- Psychoacoustic Forward Temporal Masking Decay ---
    if getattr(p, "temporal_masking", True):
        mask_psd = apply_temporal_masking_decay(
            mask_psd,
            sr=sr,
            hop=hop,
            decay_ms=float(getattr(p, "mask_decay_ms", 15.0)),
            flatness=flat_full,
            flat_start=p.flat_start,
            flat_end=p.flat_end,
            gated=bool(getattr(p, "mask_flat_gate", True)),
        )

    # Mid/Side channel depth scales (stereo only)
    ms_enabled = bool(getattr(p, "ms_process", False)) and n_ch >= 2
    ch_scales = np.ones(n_ch, dtype=np.float32)
    if ms_enabled:
        ch_scales[0] = 1.0
        ch_scales[1] = float(np.clip(getattr(p, "ms_side_scale", 0.35), 0.0, 1.0))

    hpss_on = bool(getattr(p, "hpss", False))
    hpss_tf = int(getattr(p, "hpss_time_frames", 21))
    hpss_ff = int(getattr(p, "hpss_freq_bins", 17))
    hpss_harm = bool(getattr(p, "hpss_harmonic_only", True))
    hpss_prot = float(getattr(p, "hpss_protect_percussive", 0.0))

    def _depth_scale_for(band_idx: np.ndarray) -> np.ndarray:
        if not hpss_on or band_idx.size == 0:
            return np.ones((band_idx.size, n_cols), dtype=np.float32)
        Mh = _hpss_mask_for_band(Mag_mono, band_idx, time_frames=hpss_tf, freq_bins=hpss_ff, eps=eps)
        return _hpss_depth_scale(Mh, harmonic_only=hpss_harm, protect_percussive=hpss_prot)

    g_total = np.ones((freqs.size, n_cols), dtype=np.float32)

    ctx = DSPContext(
        Z=Z,
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
        n_cols=n_cols,
        n_ch=n_ch,
        ms_enabled=ms_enabled,
        ch_scales=ch_scales,
        eps=eps,
        rng=rng,
        depth_scale_for=_depth_scale_for,
    )

    if pipeline is None:
        pipeline = build_default_pipeline(p)

    pipeline.process(ctx)
    Z = ctx.Z

    # Track artifact confidence
    artifact_conf_frames = None
    if ctx.artifact_conf_parts:
        conf_per_frame = np.zeros(n_cols, dtype=np.float32)
        for part in ctx.artifact_conf_parts:
            conf_per_frame = np.maximum(conf_per_frame, np.max(part, axis=0))
        artifact_conf_frames = conf_per_frame.reshape(1, -1)

    _LAST_ARTIFACT_CONF_FRAMES = artifact_conf_frames

    # --- Debug collection ---
    if dbg is not None:
        flux_full = flux.astype(np.float32, copy=False)

        def _att_db_from_gain(g_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if g_map is None:
                return None
            return (-20.0 * np.log10(np.clip(g_map, 1e-6, 1.0))).astype(np.float32, copy=False)

        att_dn_db_map = _att_db_from_gain(ctx.debug_maps.get("g_eff_dn_map"))
        att_deq_db_map = _att_db_from_gain(ctx.debug_maps.get("g_eff_deq_map"))
        att_sh_db_map = _att_db_from_gain(ctx.debug_maps.get("g_eff_sh_map"))
        noise_psd_dn_db_map = ctx.debug_maps.get("noise_psd_dn_db_map")
        dn_depth_map = ctx.debug_maps.get("dn_depth_map")
        deq_depth_map = ctx.debug_maps.get("deq_depth_map")
        sh_depth_map = ctx.debug_maps.get("sh_depth_map")

        for fi in range(n_cols):
            if dbg.want():
                dbg.push_meta(
                    s=fi * hop,
                    n_fft=n_fft,
                    w_noise=float(w_noise_full[fi]),
                    w_trans=float(w_trans[fi]),
                    flux_db=float(flux_full[fi]),
                    band_db=float(band_db_full[fi]),
                    dn_depth=float(dn_depth_map[fi]) if dn_depth_map is not None else 0.0,
                    deq_depth=float(deq_depth_map[fi]) if deq_depth_map is not None else 0.0,
                    sh_depth=float(sh_depth_map[fi]) if sh_depth_map is not None else 0.0,
                )
                dbg.push_att(
                    att_dn_db_map[:, fi] if att_dn_db_map is not None else None,
                    att_deq_db_map[:, fi] if att_deq_db_map is not None else None,
                    att_sh_db_map[:, fi] if att_sh_db_map is not None else None,
                )
                dbg.push_noise_psd_dn(
                    noise_psd_dn_db_map[:, fi] if noise_psd_dn_db_map is not None else None
                )
            dbg.step()

    # --- Reconstruct & Enforce Spectrogram Consistency ---
    Z_target_mag = np.abs(Z)

    for _proj_iter in range(3):
        Z_t = Z.transpose(2, 0, 1)
        _, y_proj = istft(Z_t, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', time_axis=-1, freq_axis=-2)
        f_proj, _, Z_proj = stft(y_proj, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', axis=-1)
        Z_proj = Z_proj.transpose(1, 2, 0)
        phase_proj = Z_proj / (np.abs(Z_proj) + eps)
        Z = Z_target_mag * phase_proj

    Z_out = Z.transpose(2, 0, 1)
    _, y_rec_t = istft(Z_out, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary='zeros', time_axis=-1, freq_axis=-2)

    y_rec = y_rec_t.T
    if y_rec.shape[0] > x.shape[0]:
        y_rec = y_rec[:x.shape[0], :]
    elif y_rec.shape[0] < x.shape[0]:
        y_rec = np.pad(y_rec, ((0, x.shape[0]-y_rec.shape[0]), (0, 0)))

    # HF resynthesis is part of the wet repair, before mix and delta listening.
    y_rec = _as_2d(hf_resynth_post(y_rec, sr, p, artifact_conf=artifact_conf_frames))
    y_rec, tonal_info = TonalRepair().process(
        y_rec, sr, amount=p.tonal_repair, max_shift_cents=p.tonal_max_shift_cents,
    )

    if diagnostics is not None:
        diagnostics["tonal"] = tonal_info

    mix_p = float(p.mix)
    y_final = mix_p * y_rec + (1.0 - mix_p) * dry

    if p.delta_listen:
        return (dry - y_final).squeeze()
    return y_final.squeeze()
