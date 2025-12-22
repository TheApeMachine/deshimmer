#!/usr/bin/env python3
"""
Gradio UI for master.py

Goals:
- Load an audio file
- Pick a short preview region (fast iteration)
- Tweak Params / denoise / deres / mastering knobs
- Hear A/B (input vs output) + hear diff (removed)
- See spectrograms update (input/output/diff)

Run:
  pip install -r requirements-ui.txt
  python ui_gradio.py
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

import master
from deshimmer_api import process_audio


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("audio must be 1D or 2D")
    if x.dtype.kind in ("i", "u"):
        # best-effort int PCM -> float
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / max(1.0, maxv)
    else:
        x = x.astype(np.float32, copy=False)
    # prevent crazy values from exploding plots
    return np.clip(x, -1.0, 1.0)


def _slice_preview(x: np.ndarray, sr: int, t0: float, dur: float) -> Tuple[np.ndarray, int, int]:
    n = x.shape[0]
    t0 = float(max(0.0, t0))
    dur = float(max(0.05, dur))
    s0 = int(round(t0 * sr))
    s1 = int(round((t0 + dur) * sr))
    s0 = max(0, min(n, s0))
    s1 = max(s0, min(n, s1))
    return x[s0:s1, :], s0, s1


def _spectrogram_png_bytes(x: np.ndarray, sr: int, n_fft: int, hop: int, max_frames: int, max_hz: float) -> bytes:
    # reuse master's helper to compute S_db; then plot to PNG in-memory
    S, f, t = master._compute_mag_spectrogram_db(x, sr, n_fft=n_fft, hop=hop, max_frames=max_frames, max_hz=max_hz)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    extent = [
        float(t[0]) if t.size else 0.0,
        float(t[-1]) if t.size else 0.0,
        float(f[0]) if f.size else 0.0,
        float(f[-1]) if f.size else 0.0,
    ]
    im = ax.imshow(S, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    fig.colorbar(im, ax=ax, label="dB")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    try:
        plt.close(fig)
    except Exception:
        pass
    return buf.getvalue()


def _audio_to_wav_data_uri(x: np.ndarray, sr: int) -> str:
    """Encode small preview audio to a data: URI for <audio> tags."""
    x2 = _to_float_audio(x)
    buf = io.BytesIO()
    # PCM_16 keeps data URIs smaller and is fine for preview A/B.
    sf.write(buf, x2, int(sr), format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _audio_player_html(title: str, wav_data_uri: str, *, autoplay: bool, loop: bool = True) -> str:
    # We only autoplay the output preview; input/diff should not auto-start on rerender.
    autoplay_attr = " autoplay" if autoplay else ""
    loop_attr = " loop" if loop else ""
    return (
        f"<div style='display:flex;flex-direction:column;gap:6px'>"
        f"<div><b>{title}</b></div>"
        f"<audio controls{autoplay_attr}{loop_attr} playsinline style='width:100%' src='{wav_data_uri}'></audio>"
        f"</div>"
    )


def _loop_crossfade_rotate(x: np.ndarray, sr: int, crossfade_ms: float) -> np.ndarray:
    """
    Make a loop-friendly preview by rotating the segment so the wrap point is
    between consecutive samples from the original audio, and embedding a
    crossfade between tail->head at the start.

    This produces a seamless loop without requiring custom JS/WebAudio.
    """
    x2 = _to_float_audio(x)
    n = x2.shape[0]
    if n < 8:
        return x2

    ms = float(max(0.0, crossfade_ms))
    if ms <= 0.0:
        return x2

    f = int(round((ms / 1000.0) * float(sr)))
    # keep it sane relative to segment length
    f = int(max(0, min(f, n // 4)))
    if f < 8:
        return x2

    tail = x2[n - f : n, :]
    head = x2[0:f, :]
    w = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
    cf = (1.0 - w) * tail + w * head
    mid = x2[f : n - f, :]  # skip head/tail used in crossfade
    y = np.concatenate([cf, mid], axis=0)
    return y.astype(np.float32, copy=False)


def _png_bytes_to_rgb(png_bytes: bytes) -> Optional[np.ndarray]:
    if not png_bytes:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.asarray(im)


def _render_metrics_md(info: Dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return f"{float(v):.2f}"
        return str(v)

    mi = info.get("measure_in", {})
    mr = info.get("measure_after_repair", {})
    mo = info.get("measure_out", {})
    lines = [
        "### Measurements",
        "",
        "| Stage | Peak (dBFS) | True peak (dBTP) | RMS (dBFS) | LUFS |",
        "|---|---:|---:|---:|---:|",
        f"| Input | {fmt(mi.get('sample_peak_dbfs'))} | {fmt(mi.get('true_peak_dbtp'))} | {fmt(mi.get('rms_dbfs'))} | {fmt(mi.get('lufs'))} |",
        f"| After repair | {fmt(mr.get('sample_peak_dbfs'))} | {fmt(mr.get('true_peak_dbtp'))} | {fmt(mr.get('rms_dbfs'))} | {fmt(mr.get('lufs'))} |",
        f"| Output | {fmt(mo.get('sample_peak_dbfs'))} | {fmt(mo.get('true_peak_dbtp'))} | {fmt(mo.get('rms_dbfs'))} | {fmt(mo.get('lufs'))} |",
        "",
        "### Notes",
        "- `diff = input - output` (what was removed)",
        "- Preview region is processed independently (fast iteration; results may differ slightly from full-track processing at boundaries)",
    ]
    return "\n".join(lines)


def _build_params(
    *,
    # main band
    start_hz: float,
    end_hz: float,
    edge_hz: float,
    n_fft: int,
    hop: int,
    flat_start: float,
    flat_end: float,
    freq_med_bins: int,
    thr_db: float,
    slope: float,
    density_lo: float,
    density_hi: float,
    flux_thr_db: float,
    flux_range_db: float,
    noise_resynth: float,
    mix: float,
    # denoise
    denoise: float,
    dn_start_hz: float,
    dn_end_hz: float,
    dn_edge_hz: float,
    dn_floor_db: float,
    dn_psd_smooth_ms: float,
    dn_minwin_ms: float,
    dn_up_db_per_s: float,
    dn_attack_ms: float,
    dn_release_ms: float,
    dn_freq_smooth_bins: int,
    # deres
    deres: float,
    deq_start_hz: float,
    deq_end_hz: float,
    deq_edge_hz: float,
    deq_freq_med_bins: int,
    deq_thr_db: float,
    deq_slope: float,
    deq_max_att_db: float,
    deq_density_lo: float,
    deq_density_hi: float,
    deq_persist_ms: float,
    deq_persist_thr_db: float,
    deq_freq_smooth_bins: int,
    deq_tonal_boost_db: float,
    deq_time_floor: bool,
    deq_floor_smooth_ms: float,
    deq_floor_rise_db_per_s: float,
    # downward expander
    expander: bool,
    exp_start_hz: float,
    exp_end_hz: float,
    exp_threshold_db: float,
    exp_ratio: float,
    exp_attack_ms: float,
    exp_release_ms: float,
    # HPSS-ish
    hpss: bool,
    hpss_start_hz: float,
    hpss_end_hz: float,
    hpss_time_frames: int,
    hpss_freq_bins: int,
    hpss_harmonic_only: bool,
    # phase blur
    phase_blur: float,
    pb_start_hz: float,
    pb_end_hz: float,
    pb_harmonic_only: bool,
    # nuclear HF resynthesis
    hf_resynth: bool,
    hf_lp_hz: float,
    hf_src_lo_hz: float,
    hf_src_hi_hz: float,
    hf_drive: float,
    hf_hp_hz: float,
    hf_mix: float,
    # mastering
    master_enabled: bool,
    hp_hz: float,
    target_lufs: float,
    target_rms_dbfs: float,
    norm_max_gain_db: float,
    norm_max_atten_db: float,
    ceiling_dbtp: float,
    lim_lookahead_ms: float,
    lim_release_ms: float,
    tp_os: int,
    # visuals
    spec_n_fft: int,
    spec_hop: int,
    spec_max_frames: int,
    spec_max_hz: float,
) -> Tuple[master.Params, master.MasterParams, master.DebugParams]:
    p = master.Params(
        start_hz=float(start_hz),
        end_hz=float(end_hz),
        edge_hz=float(edge_hz),
        n_fft=int(n_fft),
        hop=int(hop),
        flat_start=float(flat_start),
        flat_end=float(flat_end),
        freq_med_bins=int(freq_med_bins),
        thr_db=float(thr_db),
        slope=float(slope),
        density_lo=float(density_lo),
        density_hi=float(density_hi),
        flux_thr_db=float(flux_thr_db),
        flux_range_db=float(flux_range_db),
        noise_resynth=float(noise_resynth),
        mix=float(mix),
        # keep the preview clean; padding/fade helps boundary clicks
        pad=True,
        fade_ms=5.0,
        seed=0,
        denoise=float(denoise),
        dn_start_hz=float(dn_start_hz),
        dn_end_hz=float(dn_end_hz),
        dn_edge_hz=float(dn_edge_hz),
        dn_floor_db=float(dn_floor_db),
        dn_psd_smooth_ms=float(dn_psd_smooth_ms),
        dn_minwin_ms=float(dn_minwin_ms),
        dn_up_db_per_s=float(dn_up_db_per_s),
        dn_attack_ms=float(dn_attack_ms),
        dn_release_ms=float(dn_release_ms),
        dn_freq_smooth_bins=int(dn_freq_smooth_bins),
        deres=float(deres),
        deq_start_hz=float(deq_start_hz),
        deq_end_hz=float(deq_end_hz),
        deq_edge_hz=float(deq_edge_hz),
        deq_freq_med_bins=int(deq_freq_med_bins),
        deq_thr_db=float(deq_thr_db),
        deq_slope=float(deq_slope),
        deq_max_att_db=float(deq_max_att_db),
        deq_density_lo=float(deq_density_lo),
        deq_density_hi=float(deq_density_hi),
        deq_persist_ms=float(deq_persist_ms),
        deq_persist_thr_db=float(deq_persist_thr_db),
        deq_freq_smooth_bins=int(deq_freq_smooth_bins),
        deq_tonal_boost_db=float(deq_tonal_boost_db),
        deq_time_floor=bool(deq_time_floor),
        deq_floor_smooth_ms=float(deq_floor_smooth_ms),
        deq_floor_rise_db_per_s=float(deq_floor_rise_db_per_s),

        expander=bool(expander),
        exp_start_hz=float(exp_start_hz),
        exp_end_hz=float(exp_end_hz),
        exp_threshold_db=float(exp_threshold_db),
        exp_ratio=float(exp_ratio),
        exp_attack_ms=float(exp_attack_ms),
        exp_release_ms=float(exp_release_ms),

        hpss=bool(hpss),
        hpss_start_hz=float(hpss_start_hz),
        hpss_end_hz=float(hpss_end_hz),
        hpss_time_frames=int(hpss_time_frames),
        hpss_freq_bins=int(hpss_freq_bins),
        hpss_harmonic_only=bool(hpss_harmonic_only),

        phase_blur=float(phase_blur),
        pb_start_hz=float(pb_start_hz),
        pb_end_hz=float(pb_end_hz),
        pb_harmonic_only=bool(pb_harmonic_only),

        hf_resynth=bool(hf_resynth),
        hf_lp_hz=float(hf_lp_hz),
        hf_src_lo_hz=float(hf_src_lo_hz),
        hf_src_hi_hz=float(hf_src_hi_hz),
        hf_drive=float(hf_drive),
        hf_hp_hz=float(hf_hp_hz),
        hf_mix=float(hf_mix),
    )

    # LUFS: if user sets huge value, treat as disabled (match CLI convention)
    target_lufs_opt: Optional[float] = None if float(target_lufs) >= 998.0 else float(target_lufs)
    mp = master.MasterParams(
        enabled=bool(master_enabled),
        hp_hz=float(hp_hz),
        target_lufs=target_lufs_opt,
        target_rms_dbfs=float(target_rms_dbfs),
        norm_max_gain_db=float(norm_max_gain_db),
        norm_max_atten_db=float(norm_max_atten_db),
        ceiling_dbtp=float(ceiling_dbtp),
        lookahead_ms=float(lim_lookahead_ms),
        release_ms=float(lim_release_ms),
        os_factor=int(tp_os),
    )

    # debug disabled for UI runs; we render in-memory
    dp = master.DebugParams(
        enabled=False,
        spec_n_fft=int(spec_n_fft),
        spec_hop=int(spec_hop),
        spec_max_frames=int(spec_max_frames),
        spec_max_hz=float(spec_max_hz),
    )
    return p, mp, dp


def run_once(
    audio_in: Tuple[int, np.ndarray],
    # preview
    preview_t0: float,
    preview_dur: float,
    loop_xfade_ms: float,
    # main band
    start_hz: float,
    end_hz: float,
    edge_hz: float,
    n_fft: int,
    hop: int,
    flat_start: float,
    flat_end: float,
    freq_med_bins: int,
    thr_db: float,
    slope: float,
    density_lo: float,
    density_hi: float,
    flux_thr_db: float,
    flux_range_db: float,
    noise_resynth: float,
    mix: float,
    # denoise
    denoise: float,
    dn_start_hz: float,
    dn_end_hz: float,
    dn_edge_hz: float,
    dn_floor_db: float,
    dn_psd_smooth_ms: float,
    dn_minwin_ms: float,
    dn_up_db_per_s: float,
    dn_attack_ms: float,
    dn_release_ms: float,
    dn_freq_smooth_bins: int,
    # deres
    deres: float,
    deq_start_hz: float,
    deq_end_hz: float,
    deq_edge_hz: float,
    deq_freq_med_bins: int,
    deq_thr_db: float,
    deq_slope: float,
    deq_max_att_db: float,
    deq_density_lo: float,
    deq_density_hi: float,
    deq_persist_ms: float,
    deq_persist_thr_db: float,
    deq_freq_smooth_bins: int,
    deq_tonal_boost_db: float,
    deq_time_floor: bool,
    deq_floor_smooth_ms: float,
    deq_floor_rise_db_per_s: float,
    # downward expander
    expander: bool,
    exp_start_hz: float,
    exp_end_hz: float,
    exp_threshold_db: float,
    exp_ratio: float,
    exp_attack_ms: float,
    exp_release_ms: float,
    # HPSS-ish
    hpss: bool,
    hpss_start_hz: float,
    hpss_end_hz: float,
    hpss_time_frames: int,
    hpss_freq_bins: int,
    hpss_harmonic_only: bool,
    # phase blur
    phase_blur: float,
    pb_start_hz: float,
    pb_end_hz: float,
    pb_harmonic_only: bool,
    # nuclear HF resynthesis
    hf_resynth: bool,
    hf_lp_hz: float,
    hf_src_lo_hz: float,
    hf_src_hi_hz: float,
    hf_drive: float,
    hf_hp_hz: float,
    hf_mix: float,
    # mastering
    master_enabled: bool,
    hp_hz: float,
    target_lufs: float,
    target_rms_dbfs: float,
    norm_max_gain_db: float,
    norm_max_atten_db: float,
    ceiling_dbtp: float,
    lim_lookahead_ms: float,
    lim_release_ms: float,
    tp_os: int,
    # visuals
    spec_n_fft: int,
    spec_hop: int,
    spec_max_frames: int,
    spec_max_hz: float,
) -> Tuple[
    Tuple[int, np.ndarray],
    Tuple[int, np.ndarray],
    Tuple[int, np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    Dict[str, Any],
]:
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    x_seg, s0, s1 = _slice_preview(x, sr, preview_t0, preview_dur)

    p, mp, dp = _build_params(
        start_hz=start_hz,
        end_hz=end_hz,
        edge_hz=edge_hz,
        n_fft=n_fft,
        hop=hop,
        flat_start=flat_start,
        flat_end=flat_end,
        freq_med_bins=freq_med_bins,
        thr_db=thr_db,
        slope=slope,
        density_lo=density_lo,
        density_hi=density_hi,
        flux_thr_db=flux_thr_db,
        flux_range_db=flux_range_db,
        noise_resynth=noise_resynth,
        mix=mix,
        denoise=denoise,
        dn_start_hz=dn_start_hz,
        dn_end_hz=dn_end_hz,
        dn_edge_hz=dn_edge_hz,
        dn_floor_db=dn_floor_db,
        dn_psd_smooth_ms=dn_psd_smooth_ms,
        dn_minwin_ms=dn_minwin_ms,
        dn_up_db_per_s=dn_up_db_per_s,
        dn_attack_ms=dn_attack_ms,
        dn_release_ms=dn_release_ms,
        dn_freq_smooth_bins=dn_freq_smooth_bins,
        deres=deres,
        deq_start_hz=deq_start_hz,
        deq_end_hz=deq_end_hz,
        deq_edge_hz=deq_edge_hz,
        deq_freq_med_bins=deq_freq_med_bins,
        deq_thr_db=deq_thr_db,
        deq_slope=deq_slope,
        deq_max_att_db=deq_max_att_db,
        deq_density_lo=deq_density_lo,
        deq_density_hi=deq_density_hi,
        deq_persist_ms=deq_persist_ms,
        deq_persist_thr_db=deq_persist_thr_db,
        deq_freq_smooth_bins=deq_freq_smooth_bins,
        deq_tonal_boost_db=deq_tonal_boost_db,
        deq_time_floor=deq_time_floor,
        deq_floor_smooth_ms=deq_floor_smooth_ms,
        deq_floor_rise_db_per_s=deq_floor_rise_db_per_s,
        expander=expander,
        exp_start_hz=exp_start_hz,
        exp_end_hz=exp_end_hz,
        exp_threshold_db=exp_threshold_db,
        exp_ratio=exp_ratio,
        exp_attack_ms=exp_attack_ms,
        exp_release_ms=exp_release_ms,
        hpss=hpss,
        hpss_start_hz=hpss_start_hz,
        hpss_end_hz=hpss_end_hz,
        hpss_time_frames=hpss_time_frames,
        hpss_freq_bins=hpss_freq_bins,
        hpss_harmonic_only=hpss_harmonic_only,
        phase_blur=phase_blur,
        pb_start_hz=pb_start_hz,
        pb_end_hz=pb_end_hz,
        pb_harmonic_only=pb_harmonic_only,
        hf_resynth=hf_resynth,
        hf_lp_hz=hf_lp_hz,
        hf_src_lo_hz=hf_src_lo_hz,
        hf_src_hi_hz=hf_src_hi_hz,
        hf_drive=hf_drive,
        hf_hp_hz=hf_hp_hz,
        hf_mix=hf_mix,
        master_enabled=master_enabled,
        hp_hz=hp_hz,
        target_lufs=target_lufs,
        target_rms_dbfs=target_rms_dbfs,
        norm_max_gain_db=norm_max_gain_db,
        norm_max_atten_db=norm_max_atten_db,
        ceiling_dbtp=ceiling_dbtp,
        lim_lookahead_ms=lim_lookahead_ms,
        lim_release_ms=lim_release_ms,
        tp_os=tp_os,
        spec_n_fft=spec_n_fft,
        spec_hop=spec_hop,
        spec_max_frames=spec_max_frames,
        spec_max_hz=spec_max_hz,
    )

    y, info = process_audio(x_seg, sr, params=p, master_params=mp, debug_params=dp)
    y2 = _to_float_audio(y)

    # Diff = what was removed
    diff = (x_seg[: y2.shape[0], :] - y2).astype(np.float32)

    # Make loop seam smoother (rotate+crossfade) for playback
    x_play = _loop_crossfade_rotate(x_seg, sr, loop_xfade_ms)
    y_play = _loop_crossfade_rotate(y2, sr, loop_xfade_ms)
    d_play = _loop_crossfade_rotate(diff, sr, loop_xfade_ms)

    # Spectrograms
    in_png = _spectrogram_png_bytes(x_seg, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))
    out_png = _spectrogram_png_bytes(y2, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))
    diff_png = _spectrogram_png_bytes(diff, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))

    metrics_md = _render_metrics_md(info)
    params_json = {"params": asdict(p), "master_params": asdict(mp)}

    return (
        _audio_player_html("Preview: input (loop)", _audio_to_wav_data_uri(x_play, sr), autoplay=False, loop=True),
        _audio_player_html("Preview: output (loop)", _audio_to_wav_data_uri(y_play, sr), autoplay=True, loop=True),
        _audio_player_html("Preview: diff / removed (loop)", _audio_to_wav_data_uri(d_play, sr), autoplay=False, loop=True),
        _png_bytes_to_rgb(in_png),
        _png_bytes_to_rgb(out_png),
        _png_bytes_to_rgb(diff_png),
        metrics_md,
        params_json,  # dict shown in JSON component
    )


def render_full_to_files(
    audio_in: Tuple[int, np.ndarray],
    # all settings (same as run_once, minus preview t0/dur)
    start_hz: float,
    end_hz: float,
    edge_hz: float,
    n_fft: int,
    hop: int,
    flat_start: float,
    flat_end: float,
    freq_med_bins: int,
    thr_db: float,
    slope: float,
    density_lo: float,
    density_hi: float,
    flux_thr_db: float,
    flux_range_db: float,
    noise_resynth: float,
    mix: float,
    denoise: float,
    dn_start_hz: float,
    dn_end_hz: float,
    dn_edge_hz: float,
    dn_floor_db: float,
    dn_psd_smooth_ms: float,
    dn_minwin_ms: float,
    dn_up_db_per_s: float,
    dn_attack_ms: float,
    dn_release_ms: float,
    dn_freq_smooth_bins: int,
    deres: float,
    deq_start_hz: float,
    deq_end_hz: float,
    deq_edge_hz: float,
    deq_freq_med_bins: int,
    deq_thr_db: float,
    deq_slope: float,
    deq_max_att_db: float,
    deq_density_lo: float,
    deq_density_hi: float,
    deq_persist_ms: float,
    deq_persist_thr_db: float,
    deq_freq_smooth_bins: int,
    deq_tonal_boost_db: float,
    deq_time_floor: bool,
    deq_floor_smooth_ms: float,
    deq_floor_rise_db_per_s: float,
    # downward expander
    expander: bool,
    exp_start_hz: float,
    exp_end_hz: float,
    exp_threshold_db: float,
    exp_ratio: float,
    exp_attack_ms: float,
    exp_release_ms: float,
    # HPSS-ish
    hpss: bool,
    hpss_start_hz: float,
    hpss_end_hz: float,
    hpss_time_frames: int,
    hpss_freq_bins: int,
    hpss_harmonic_only: bool,
    # phase blur
    phase_blur: float,
    pb_start_hz: float,
    pb_end_hz: float,
    pb_harmonic_only: bool,
    # nuclear HF resynthesis
    hf_resynth: bool,
    hf_lp_hz: float,
    hf_src_lo_hz: float,
    hf_src_hi_hz: float,
    hf_drive: float,
    hf_hp_hz: float,
    hf_mix: float,
    master_enabled: bool,
    hp_hz: float,
    target_lufs: float,
    target_rms_dbfs: float,
    norm_max_gain_db: float,
    norm_max_atten_db: float,
    ceiling_dbtp: float,
    lim_lookahead_ms: float,
    lim_release_ms: float,
    tp_os: int,
    # visuals params exist but unused for full render
    spec_n_fft: int,
    spec_hop: int,
    spec_max_frames: int,
    spec_max_hz: float,
) -> Tuple[str, str, str]:
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    p, mp, dp = _build_params(
        start_hz=start_hz,
        end_hz=end_hz,
        edge_hz=edge_hz,
        n_fft=n_fft,
        hop=hop,
        flat_start=flat_start,
        flat_end=flat_end,
        freq_med_bins=freq_med_bins,
        thr_db=thr_db,
        slope=slope,
        density_lo=density_lo,
        density_hi=density_hi,
        flux_thr_db=flux_thr_db,
        flux_range_db=flux_range_db,
        noise_resynth=noise_resynth,
        mix=mix,
        denoise=denoise,
        dn_start_hz=dn_start_hz,
        dn_end_hz=dn_end_hz,
        dn_edge_hz=dn_edge_hz,
        dn_floor_db=dn_floor_db,
        dn_psd_smooth_ms=dn_psd_smooth_ms,
        dn_minwin_ms=dn_minwin_ms,
        dn_up_db_per_s=dn_up_db_per_s,
        dn_attack_ms=dn_attack_ms,
        dn_release_ms=dn_release_ms,
        dn_freq_smooth_bins=dn_freq_smooth_bins,
        deres=deres,
        deq_start_hz=deq_start_hz,
        deq_end_hz=deq_end_hz,
        deq_edge_hz=deq_edge_hz,
        deq_freq_med_bins=deq_freq_med_bins,
        deq_thr_db=deq_thr_db,
        deq_slope=deq_slope,
        deq_max_att_db=deq_max_att_db,
        deq_density_lo=deq_density_lo,
        deq_density_hi=deq_density_hi,
        deq_persist_ms=deq_persist_ms,
        deq_persist_thr_db=deq_persist_thr_db,
        deq_freq_smooth_bins=deq_freq_smooth_bins,
        deq_tonal_boost_db=deq_tonal_boost_db,
        deq_time_floor=deq_time_floor,
        deq_floor_smooth_ms=deq_floor_smooth_ms,
        deq_floor_rise_db_per_s=deq_floor_rise_db_per_s,
        expander=expander,
        exp_start_hz=exp_start_hz,
        exp_end_hz=exp_end_hz,
        exp_threshold_db=exp_threshold_db,
        exp_ratio=exp_ratio,
        exp_attack_ms=exp_attack_ms,
        exp_release_ms=exp_release_ms,
        hpss=hpss,
        hpss_start_hz=hpss_start_hz,
        hpss_end_hz=hpss_end_hz,
        hpss_time_frames=hpss_time_frames,
        hpss_freq_bins=hpss_freq_bins,
        hpss_harmonic_only=hpss_harmonic_only,
        phase_blur=phase_blur,
        pb_start_hz=pb_start_hz,
        pb_end_hz=pb_end_hz,
        pb_harmonic_only=pb_harmonic_only,
        hf_resynth=hf_resynth,
        hf_lp_hz=hf_lp_hz,
        hf_src_lo_hz=hf_src_lo_hz,
        hf_src_hi_hz=hf_src_hi_hz,
        hf_drive=hf_drive,
        hf_hp_hz=hf_hp_hz,
        hf_mix=hf_mix,
        master_enabled=master_enabled,
        hp_hz=hp_hz,
        target_lufs=target_lufs,
        target_rms_dbfs=target_rms_dbfs,
        norm_max_gain_db=norm_max_gain_db,
        norm_max_atten_db=norm_max_atten_db,
        ceiling_dbtp=ceiling_dbtp,
        lim_lookahead_ms=lim_lookahead_ms,
        lim_release_ms=lim_release_ms,
        tp_os=tp_os,
        spec_n_fft=spec_n_fft,
        spec_hop=spec_hop,
        spec_max_frames=spec_max_frames,
        spec_max_hz=spec_max_hz,
    )

    y, _ = process_audio(x, sr, params=p, master_params=mp, debug_params=dp)
    y2 = _to_float_audio(y)
    diff = (x[: y2.shape[0], :] - y2).astype(np.float32)

    outdir = os.path.join(os.path.dirname(__file__), "ui_downloads")
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(outdir, f"output-{stamp}.wav")
    diff_path = os.path.join(outdir, f"diff-{stamp}.wav")
    params_path = os.path.join(outdir, f"params-{stamp}.json")

    sf.write(out_path, y2, sr, subtype="PCM_24")
    sf.write(diff_path, diff, sr, subtype="PCM_24")

    import json

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"params": asdict(p), "master_params": asdict(mp)}, f, indent=2, sort_keys=True)

    return out_path, diff_path, params_path


def build_ui() -> Any:
    import gradio as gr

    def _load_user_presets(path: str) -> Dict[str, Dict[str, Any]]:
        try:
            import json

            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                vals = v.get("values", {})
                if not isinstance(vals, dict):
                    continue
                out[k] = {"desc": str(v.get("desc", "")), "values": vals}
            return out
        except Exception:
            return {}

    def _save_user_preset(path: str, name: str, desc: str, values: Dict[str, Any]) -> None:
        import json

        data = _load_user_presets(path)
        data[name] = {"desc": desc, "values": values}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    USER_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "ui_presets.json")

    BASE_PRESETS: Dict[str, Dict[str, Any]] = {
        # --- Simple (few knobs) ---
        "01 - Bypass (no processing)": {
            "desc": "Mix=0.0 (fully dry). Useful sanity check.",
            "values": {"mix": 0.0, "noise_resynth": 0.0, "denoise": 0.0, "deres": 0.0, "master_enabled": False},
        },
        "02 - Default shimmer (recommended start)": {
            "desc": "Conservative shimmer suppression in 5.1–7.2 kHz.",
            "values": {"mix": 1.0, "thr_db": 8.0, "slope": 0.6, "noise_resynth": 0.0, "denoise": 0.0, "deres": 0.0, "master_enabled": False},
        },
        "03 - Gentle shimmer": {
            "desc": "Higher threshold + lower slope; safest for already-good material.",
            "values": {"thr_db": 10.0, "slope": 0.45, "noise_resynth": 0.0, "mix": 1.0},
        },
        "04 - Aggressive shimmer": {
            "desc": "Lower threshold + higher slope. Can dull cymbals; use preview loop.",
            "values": {"thr_db": 6.5, "slope": 0.85, "noise_resynth": 0.0, "mix": 1.0},
        },
        "05 - Shimmer + de-crystallize (noise resynth)": {
            "desc": "Adds subtle random-phase blend in noise-like frames to soften sparkle.",
            "values": {"thr_db": 8.0, "slope": 0.6, "noise_resynth": 0.25, "mix": 1.0},
        },
        # --- Medium (introduce denoise / deres) ---
        "06 - Shimmer + light denoise": {
            "desc": "Taste-neutral noise floor management; good for codec hiss / swish.",
            "values": {"denoise": 0.25, "dn_floor_db": -18.0, "dn_freq_smooth_bins": 3, "mix": 1.0},
        },
        "07 - Shimmer + stronger denoise": {
            "desc": "More denoise depth, more smoothing to avoid musical noise.",
            "values": {"denoise": 0.55, "dn_floor_db": -18.0, "dn_freq_smooth_bins": 5, "dn_release_ms": 180.0, "mix": 1.0},
        },
        "08 - De-resonator (gentle)": {
            "desc": "Dynamic EQ for persistent narrow resonances; conservative.",
            "values": {"deres": 0.35, "deq_thr_db": 7.0, "deq_max_att_db": 6.0, "deq_persist_ms": 800.0, "mix": 1.0},
        },
        "09 - De-resonator (stronger)": {
            "desc": "More active notch behavior on persistent peaks (still capped).",
            "values": {"deres": 0.70, "deq_thr_db": 6.0, "deq_max_att_db": 10.0, "deq_persist_ms": 700.0, "deq_freq_smooth_bins": 7, "mix": 1.0},
        },
        # --- Complex (touch many knobs) ---
        "10 - Full stack (conservative)": {
            "desc": "Shimmer + light denoise + gentle deres. Good 'set-and-forget' baseline.",
            "values": {
                "thr_db": 8.5,
                "slope": 0.55,
                "noise_resynth": 0.10,
                "denoise": 0.25,
                "dn_floor_db": -18.0,
                "dn_psd_smooth_ms": 60.0,
                "dn_minwin_ms": 450.0,
                "dn_release_ms": 140.0,
                "dn_freq_smooth_bins": 5,
                "deres": 0.35,
                "deq_thr_db": 7.0,
                "deq_max_att_db": 7.0,
                "deq_persist_ms": 900.0,
                "deq_persist_thr_db": 2.8,
                "deq_freq_smooth_bins": 7,
                "mix": 1.0,
                "master_enabled": False,
            },
        },
        "11 - Full stack (aggressive repair)": {
            "desc": "More denoise + more shimmer suppression + stronger deres. Watch for dullness.",
            "values": {
                "thr_db": 7.0,
                "slope": 0.85,
                "noise_resynth": 0.20,
                "denoise": 0.60,
                "dn_floor_db": -20.0,
                "dn_psd_smooth_ms": 40.0,
                "dn_minwin_ms": 350.0,
                "dn_release_ms": 220.0,
                "dn_freq_smooth_bins": 7,
                "deres": 0.75,
                "deq_thr_db": 6.0,
                "deq_max_att_db": 12.0,
                "deq_persist_ms": 650.0,
                "deq_persist_thr_db": 2.2,
                "deq_freq_smooth_bins": 9,
                "mix": 1.0,
                "master_enabled": False,
            },
        },
        "12 - Delivery: -14 LUFS, -1 dBTP": {
            "desc": "Enable mastering stage for streaming-ish loudness.",
            "values": {"master_enabled": True, "target_lufs": -14.0, "ceiling_dbtp": -1.0, "tp_os": 4, "hp_hz": 20.0},
        },
        "13 - Delivery: louder (-10 LUFS), -1 dBTP": {
            "desc": "Hotter delivery (can pump/limit more).",
            "values": {"master_enabled": True, "target_lufs": -10.0, "ceiling_dbtp": -1.0, "tp_os": 4, "hp_hz": 20.0},
        },
        # --- Band experiments ---
        "14 - Band experiment: 2k–6k (harshness zone)": {
            "desc": "Useful if the artifact is more 'presence harshness' than 6–7k sparkle.",
            "values": {"start_hz": 2000.0, "end_hz": 6000.0, "thr_db": 8.0, "slope": 0.6, "mix": 1.0},
        },
        "15 - Band experiment: 5.8k–7.8k": {
            "desc": "Shift shimmer band upward a bit.",
            "values": {"start_hz": 5800.0, "end_hz": 7800.0, "thr_db": 8.0, "slope": 0.6, "mix": 1.0},
        },
        "16 - Suno/Udio: 3.5k whine (exact CLI values)": {
            "desc": "Exactly: --start-hz 3000 --end-hz 4200 --deres 0.8 --deq-thr-db 4.0 --deq-max-att-db 12.0 --deq-freq-med-bins 61 --deq-persist-ms 1000 --thr-db 6.0 --slope 0.9",
            "values": {
                "start_hz": 3000.0,
                "end_hz": 4200.0,
                "deres": 0.8,
                "deq_thr_db": 4.0,
                "deq_max_att_db": 12.0,
                "deq_freq_med_bins": 61,
                "deq_persist_ms": 1000.0,
                "thr_db": 6.0,
                "slope": 0.9,
            },
        },
        "17 - Suno/Udio: 3.5k whine + metallic crickets (full recipe)": {
            "desc": "Whine recipe + denoise/noise-resynth for 'wind/crickets'. Also enables time-stabilized floor for stationary lines.",
            "values": {
                # Whine recipe (matches the CLI values)
                "start_hz": 3000.0,
                "end_hz": 4200.0,
                "thr_db": 6.0,
                "slope": 0.9,
                "deres": 0.8,
                "deq_thr_db": 4.0,
                "deq_max_att_db": 12.0,
                "deq_freq_med_bins": 61,
                "deq_persist_ms": 1000.0,
                # Extra help for stationary lines
                "deq_time_floor": True,
                "deq_floor_smooth_ms": 80.0,
                "deq_floor_rise_db_per_s": 1.0,
                # Denoise for the 'carpet'
                "denoise": 0.8,
                "dn_start_hz": 3000.0,
                "dn_end_hz": 16000.0,
                "dn_floor_db": -24.0,
                "dn_minwin_ms": 1000.0,
                "dn_attack_ms": 10.0,
                "dn_release_ms": 150.0,
                "dn_freq_smooth_bins": 5,
                # Soften metallic texture
                "noise_resynth": 0.35,
                "mix": 1.0,
                "master_enabled": False,
            },
        },
    }

    ALL_PRESETS: Dict[str, Dict[str, Any]] = dict(BASE_PRESETS)
    ALL_PRESETS.update(_load_user_presets(USER_PRESETS_PATH))

    with gr.Blocks(title="Deshimmer - master.py UI") as demo:
        gr.Markdown(
            "## Deshimmer UI (master.py)\n"
            "Load a file, pick a preview region, tweak knobs, then listen to input/output/diff and inspect spectrograms."
        )

        with gr.Row():
            preset = gr.Dropdown(choices=list(ALL_PRESETS.keys()), value="02 - Default shimmer (recommended start)", label="Presets (simple → complex)")
            preset_apply = gr.Button("Apply preset", variant="secondary")
        preset_desc = gr.Markdown()

        with gr.Row():
            audio_in = gr.Audio(label="Input audio", type="numpy")
            with gr.Column():
                preview_t0 = gr.Slider(0.0, 600.0, value=0.0, step=0.05, label="Preview start (s)")
                preview_dur = gr.Slider(0.5, 20.0, value=6.0, step=0.05, label="Preview duration (s)")
                loop_xfade_ms = gr.Slider(0.0, 250.0, value=35.0, step=1.0, label="Loop crossfade (ms)")
                run_btn = gr.Button("Run preview")

        with gr.Row():
            a_in = gr.HTML()
            a_out = gr.HTML()
            a_diff = gr.HTML()

        with gr.Row():
            im_in = gr.Image(label="Spectrogram: input", type="numpy")
            im_out = gr.Image(label="Spectrogram: output", type="numpy")
            im_diff = gr.Image(label="Spectrogram: diff", type="numpy")

        metrics = gr.Markdown()
        params_json = gr.JSON(label="Effective params")

        with gr.Row():
            full_btn = gr.Button("Render full & generate downloads", variant="primary")
            dl_out = gr.File(label="Download: full output.wav")
            dl_diff = gr.File(label="Download: full diff.wav")
            dl_params = gr.File(label="Download: params.json")

        with gr.Accordion("Shimmer band + core STFT", open=True):
            with gr.Row():
                start_hz = gr.Number(value=5100.0, label="start_hz")
                end_hz = gr.Number(value=7200.0, label="end_hz")
                edge_hz = gr.Number(value=200.0, label="edge_hz")
            with gr.Row():
                n_fft = gr.Dropdown([1024, 2048, 4096, 8192], value=2048, label="n_fft")
                hop = gr.Dropdown([256, 512, 1024, 2048], value=512, label="hop")
            with gr.Row():
                flat_start = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="flat_start")
                flat_end = gr.Slider(0.0, 1.0, value=0.70, step=0.01, label="flat_end")
            with gr.Row():
                freq_med_bins = gr.Slider(3, 61, value=9, step=2, label="freq_med_bins (odd)")
                thr_db = gr.Slider(0.0, 24.0, value=8.0, step=0.1, label="thr_db")
                slope = gr.Slider(0.0, 2.0, value=0.6, step=0.01, label="slope")
            with gr.Row():
                density_lo = gr.Slider(0.0, 0.5, value=0.02, step=0.005, label="density_lo")
                density_hi = gr.Slider(0.0, 0.5, value=0.15, step=0.005, label="density_hi")
            with gr.Row():
                flux_thr_db = gr.Slider(0.0, 24.0, value=6.0, step=0.1, label="flux_thr_db")
                flux_range_db = gr.Slider(0.0, 24.0, value=8.0, step=0.1, label="flux_range_db")
            with gr.Row():
                noise_resynth = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="noise_resynth")
                mix = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="mix (wet)")

        with gr.Accordion("Smart denoise", open=False):
            with gr.Row():
                denoise = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="denoise")
                dn_floor_db = gr.Slider(-60.0, 0.0, value=-18.0, step=0.5, label="dn_floor_db")
            with gr.Row():
                dn_start_hz = gr.Number(value=120.0, label="dn_start_hz")
                dn_end_hz = gr.Number(value=16000.0, label="dn_end_hz")
                dn_edge_hz = gr.Number(value=200.0, label="dn_edge_hz")
            with gr.Row():
                dn_psd_smooth_ms = gr.Slider(0.0, 300.0, value=50.0, step=1.0, label="dn_psd_smooth_ms")
                dn_minwin_ms = gr.Slider(50.0, 2000.0, value=400.0, step=10.0, label="dn_minwin_ms")
                dn_up_db_per_s = gr.Slider(0.0, 24.0, value=3.0, step=0.1, label="dn_up_db_per_s")
            with gr.Row():
                dn_attack_ms = gr.Slider(0.0, 200.0, value=5.0, step=1.0, label="dn_attack_ms")
                dn_release_ms = gr.Slider(0.0, 1000.0, value=120.0, step=5.0, label="dn_release_ms")
                dn_freq_smooth_bins = gr.Slider(1, 21, value=3, step=1, label="dn_freq_smooth_bins")

        with gr.Accordion("Smart de-resonator (dynamic EQ)", open=False):
            with gr.Row():
                deres = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="deres")
                deq_max_att_db = gr.Slider(0.0, 24.0, value=8.0, step=0.1, label="deq_max_att_db")
            with gr.Row():
                deq_start_hz = gr.Number(value=180.0, label="deq_start_hz")
                deq_end_hz = gr.Number(value=12000.0, label="deq_end_hz")
                deq_edge_hz = gr.Number(value=150.0, label="deq_edge_hz")
            with gr.Row():
                deq_freq_med_bins = gr.Slider(3, 121, value=31, step=2, label="deq_freq_med_bins (odd)")
                deq_thr_db = gr.Slider(0.0, 24.0, value=6.0, step=0.1, label="deq_thr_db")
                deq_slope = gr.Slider(0.0, 2.0, value=0.7, step=0.01, label="deq_slope")
            with gr.Row():
                deq_density_lo = gr.Slider(0.0, 0.5, value=0.03, step=0.005, label="deq_density_lo")
                deq_density_hi = gr.Slider(0.0, 0.5, value=0.20, step=0.005, label="deq_density_hi")
            with gr.Row():
                deq_persist_ms = gr.Slider(0.0, 5000.0, value=600.0, step=25.0, label="deq_persist_ms")
                deq_persist_thr_db = gr.Slider(0.0, 24.0, value=2.5, step=0.1, label="deq_persist_thr_db")
                deq_freq_smooth_bins = gr.Slider(1, 21, value=5, step=1, label="deq_freq_smooth_bins")
                deq_tonal_boost_db = gr.Slider(0.0, 24.0, value=6.0, step=0.1, label="deq_tonal_boost_db")
            with gr.Row():
                deq_time_floor = gr.Checkbox(value=False, label="deq_time_floor (target stationary ringing lines)")
                deq_floor_smooth_ms = gr.Slider(0.0, 500.0, value=80.0, step=5.0, label="deq_floor_smooth_ms")
                deq_floor_rise_db_per_s = gr.Slider(0.0, 12.0, value=1.0, step=0.1, label="deq_floor_rise_db_per_s")

        with gr.Accordion("Advanced: diffusion grit tools (optional)", open=False):
            with gr.Row():
                expander = gr.Checkbox(value=False, label="expander (downward expander in band)")
                exp_threshold_db = gr.Slider(-90.0, 0.0, value=-45.0, step=0.5, label="exp_threshold_db")
                exp_ratio = gr.Slider(1.0, 6.0, value=2.0, step=0.05, label="exp_ratio")
            with gr.Row():
                exp_start_hz = gr.Number(value=3000.0, label="exp_start_hz")
                exp_end_hz = gr.Number(value=8000.0, label="exp_end_hz")
                exp_attack_ms = gr.Slider(0.0, 200.0, value=10.0, step=1.0, label="exp_attack_ms")
                exp_release_ms = gr.Slider(0.0, 1000.0, value=150.0, step=5.0, label="exp_release_ms")

            with gr.Row():
                hpss = gr.Checkbox(value=False, label="hpss (HPSS-ish harmonic mask in band)")
                hpss_harmonic_only = gr.Checkbox(value=True, label="hpss_harmonic_only")
            with gr.Row():
                hpss_start_hz = gr.Number(value=3000.0, label="hpss_start_hz")
                hpss_end_hz = gr.Number(value=8000.0, label="hpss_end_hz")
                hpss_time_frames = gr.Slider(3, 61, value=21, step=2, label="hpss_time_frames (odd)")
                hpss_freq_bins = gr.Slider(3, 61, value=17, step=2, label="hpss_freq_bins (odd)")

            with gr.Row():
                phase_blur = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="phase_blur (0..1)")
                pb_harmonic_only = gr.Checkbox(value=True, label="pb_harmonic_only")
            with gr.Row():
                pb_start_hz = gr.Number(value=3000.0, label="pb_start_hz")
                pb_end_hz = gr.Number(value=8000.0, label="pb_end_hz")

            with gr.Row():
                hf_resynth = gr.Checkbox(value=False, label="hf_resynth (nuclear HF resynthesis)")
                hf_mix = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="hf_mix")
            with gr.Row():
                hf_lp_hz = gr.Number(value=3000.0, label="hf_lp_hz")
                hf_hp_hz = gr.Number(value=3000.0, label="hf_hp_hz")
                hf_drive = gr.Slider(0.1, 10.0, value=2.0, step=0.05, label="hf_drive")
            with gr.Row():
                hf_src_lo_hz = gr.Number(value=1000.0, label="hf_src_lo_hz")
                hf_src_hi_hz = gr.Number(value=2000.0, label="hf_src_hi_hz")

        with gr.Accordion("Delivery mastering", open=False):
            master_enabled = gr.Checkbox(value=False, label="Enable master stage")
            with gr.Row():
                hp_hz = gr.Slider(0.0, 80.0, value=20.0, step=0.5, label="hp_hz")
                tp_os = gr.Dropdown([1, 2, 4, 8], value=4, label="tp_os")
            with gr.Row():
                target_lufs = gr.Slider(-30.0, 999.0, value=-14.0, step=0.1, label="target_lufs (set >=998 to disable)")
                target_rms_dbfs = gr.Slider(-40.0, -6.0, value=-16.0, step=0.1, label="target_rms_dbfs (fallback)")
            with gr.Row():
                norm_max_gain_db = gr.Slider(0.0, 24.0, value=12.0, step=0.5, label="norm_max_gain_db")
                norm_max_atten_db = gr.Slider(0.0, 60.0, value=24.0, step=0.5, label="norm_max_atten_db")
            with gr.Row():
                ceiling_dbtp = gr.Slider(-12.0, 0.0, value=-1.0, step=0.1, label="ceiling_dbtp")
                lim_lookahead_ms = gr.Slider(0.0, 50.0, value=5.0, step=0.5, label="lim_lookahead_ms")
                lim_release_ms = gr.Slider(10.0, 1000.0, value=100.0, step=5.0, label="lim_release_ms")

        with gr.Accordion("Spectrogram settings (UI only)", open=False):
            with gr.Row():
                spec_n_fft = gr.Dropdown([512, 1024, 2048, 4096], value=2048, label="spec_n_fft")
                spec_hop = gr.Dropdown([128, 256, 512, 1024], value=512, label="spec_hop")
            with gr.Row():
                spec_max_frames = gr.Slider(200, 4000, value=1200, step=50, label="spec_max_frames")
                spec_max_hz = gr.Slider(2000.0, 24000.0, value=20000.0, step=100.0, label="spec_max_hz")

        with gr.Accordion("Preset management", open=False):
            preset_name = gr.Textbox(label="New preset name", placeholder="e.g. My grit killer v1")
            preset_user_desc = gr.Textbox(label="Preset description (optional)")
            preset_save = gr.Button("Save preset from current knobs", variant="primary")
            preset_save_status = gr.Markdown()

        def _preset_desc_md(name: str) -> str:
            p = ALL_PRESETS.get(name, {})
            desc = p.get("desc", "")
            return f"**{name}**  \n{desc}" if desc else f"**{name}**"

        def apply_preset(name: str):
            p = ALL_PRESETS.get(name, {})
            vals = dict(p.get("values", {}))
            # return updates in the same order as outputs list below
            def g(key: str, current):
                return vals.get(key, current)

            return (
                g("start_hz", start_hz.value),
                g("end_hz", end_hz.value),
                g("edge_hz", edge_hz.value),
                g("n_fft", n_fft.value),
                g("hop", hop.value),
                g("flat_start", flat_start.value),
                g("flat_end", flat_end.value),
                g("freq_med_bins", freq_med_bins.value),
                g("thr_db", thr_db.value),
                g("slope", slope.value),
                g("density_lo", density_lo.value),
                g("density_hi", density_hi.value),
                g("flux_thr_db", flux_thr_db.value),
                g("flux_range_db", flux_range_db.value),
                g("noise_resynth", noise_resynth.value),
                g("mix", mix.value),
                g("denoise", denoise.value),
                g("dn_start_hz", dn_start_hz.value),
                g("dn_end_hz", dn_end_hz.value),
                g("dn_edge_hz", dn_edge_hz.value),
                g("dn_floor_db", dn_floor_db.value),
                g("dn_psd_smooth_ms", dn_psd_smooth_ms.value),
                g("dn_minwin_ms", dn_minwin_ms.value),
                g("dn_up_db_per_s", dn_up_db_per_s.value),
                g("dn_attack_ms", dn_attack_ms.value),
                g("dn_release_ms", dn_release_ms.value),
                g("dn_freq_smooth_bins", dn_freq_smooth_bins.value),
                g("deres", deres.value),
                g("deq_start_hz", deq_start_hz.value),
                g("deq_end_hz", deq_end_hz.value),
                g("deq_edge_hz", deq_edge_hz.value),
                g("deq_freq_med_bins", deq_freq_med_bins.value),
                g("deq_thr_db", deq_thr_db.value),
                g("deq_slope", deq_slope.value),
                g("deq_max_att_db", deq_max_att_db.value),
                g("deq_density_lo", deq_density_lo.value),
                g("deq_density_hi", deq_density_hi.value),
                g("deq_persist_ms", deq_persist_ms.value),
                g("deq_persist_thr_db", deq_persist_thr_db.value),
                g("deq_freq_smooth_bins", deq_freq_smooth_bins.value),
                g("deq_tonal_boost_db", deq_tonal_boost_db.value),
                g("deq_time_floor", deq_time_floor.value),
                g("deq_floor_smooth_ms", deq_floor_smooth_ms.value),
                g("deq_floor_rise_db_per_s", deq_floor_rise_db_per_s.value),
                g("expander", expander.value),
                g("exp_start_hz", exp_start_hz.value),
                g("exp_end_hz", exp_end_hz.value),
                g("exp_threshold_db", exp_threshold_db.value),
                g("exp_ratio", exp_ratio.value),
                g("exp_attack_ms", exp_attack_ms.value),
                g("exp_release_ms", exp_release_ms.value),
                g("hpss", hpss.value),
                g("hpss_start_hz", hpss_start_hz.value),
                g("hpss_end_hz", hpss_end_hz.value),
                g("hpss_time_frames", hpss_time_frames.value),
                g("hpss_freq_bins", hpss_freq_bins.value),
                g("hpss_harmonic_only", hpss_harmonic_only.value),
                g("phase_blur", phase_blur.value),
                g("pb_start_hz", pb_start_hz.value),
                g("pb_end_hz", pb_end_hz.value),
                g("pb_harmonic_only", pb_harmonic_only.value),
                g("hf_resynth", hf_resynth.value),
                g("hf_lp_hz", hf_lp_hz.value),
                g("hf_src_lo_hz", hf_src_lo_hz.value),
                g("hf_src_hi_hz", hf_src_hi_hz.value),
                g("hf_drive", hf_drive.value),
                g("hf_hp_hz", hf_hp_hz.value),
                g("hf_mix", hf_mix.value),
                g("master_enabled", master_enabled.value),
                g("hp_hz", hp_hz.value),
                g("target_lufs", target_lufs.value),
                g("target_rms_dbfs", target_rms_dbfs.value),
                g("norm_max_gain_db", norm_max_gain_db.value),
                g("norm_max_atten_db", norm_max_atten_db.value),
                g("ceiling_dbtp", ceiling_dbtp.value),
                g("lim_lookahead_ms", lim_lookahead_ms.value),
                g("lim_release_ms", lim_release_ms.value),
                g("tp_os", tp_os.value),
                _preset_desc_md(name),
            )

        preset.change(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])

        preset_outputs = [
            start_hz,
            end_hz,
            edge_hz,
            n_fft,
            hop,
            flat_start,
            flat_end,
            freq_med_bins,
            thr_db,
            slope,
            density_lo,
            density_hi,
            flux_thr_db,
            flux_range_db,
            noise_resynth,
            mix,
            denoise,
            dn_start_hz,
            dn_end_hz,
            dn_edge_hz,
            dn_floor_db,
            dn_psd_smooth_ms,
            dn_minwin_ms,
            dn_up_db_per_s,
            dn_attack_ms,
            dn_release_ms,
            dn_freq_smooth_bins,
            deres,
            deq_start_hz,
            deq_end_hz,
            deq_edge_hz,
            deq_freq_med_bins,
            deq_thr_db,
            deq_slope,
            deq_max_att_db,
            deq_density_lo,
            deq_density_hi,
            deq_persist_ms,
            deq_persist_thr_db,
            deq_freq_smooth_bins,
            deq_tonal_boost_db,
            deq_time_floor,
            deq_floor_smooth_ms,
            deq_floor_rise_db_per_s,
            expander,
            exp_start_hz,
            exp_end_hz,
            exp_threshold_db,
            exp_ratio,
            exp_attack_ms,
            exp_release_ms,
            hpss,
            hpss_start_hz,
            hpss_end_hz,
            hpss_time_frames,
            hpss_freq_bins,
            hpss_harmonic_only,
            phase_blur,
            pb_start_hz,
            pb_end_hz,
            pb_harmonic_only,
            hf_resynth,
            hf_lp_hz,
            hf_src_lo_hz,
            hf_src_hi_hz,
            hf_drive,
            hf_hp_hz,
            hf_mix,
            master_enabled,
            hp_hz,
            target_lufs,
            target_rms_dbfs,
            norm_max_gain_db,
            norm_max_atten_db,
            ceiling_dbtp,
            lim_lookahead_ms,
            lim_release_ms,
            tp_os,
            preset_desc,
        ]

        inputs = [
            audio_in,
            preview_t0,
            preview_dur,
            loop_xfade_ms,
            start_hz,
            end_hz,
            edge_hz,
            n_fft,
            hop,
            flat_start,
            flat_end,
            freq_med_bins,
            thr_db,
            slope,
            density_lo,
            density_hi,
            flux_thr_db,
            flux_range_db,
            noise_resynth,
            mix,
            denoise,
            dn_start_hz,
            dn_end_hz,
            dn_edge_hz,
            dn_floor_db,
            dn_psd_smooth_ms,
            dn_minwin_ms,
            dn_up_db_per_s,
            dn_attack_ms,
            dn_release_ms,
            dn_freq_smooth_bins,
            deres,
            deq_start_hz,
            deq_end_hz,
            deq_edge_hz,
            deq_freq_med_bins,
            deq_thr_db,
            deq_slope,
            deq_max_att_db,
            deq_density_lo,
            deq_density_hi,
            deq_persist_ms,
            deq_persist_thr_db,
            deq_freq_smooth_bins,
            deq_tonal_boost_db,
            deq_time_floor,
            deq_floor_smooth_ms,
            deq_floor_rise_db_per_s,
            expander,
            exp_start_hz,
            exp_end_hz,
            exp_threshold_db,
            exp_ratio,
            exp_attack_ms,
            exp_release_ms,
            hpss,
            hpss_start_hz,
            hpss_end_hz,
            hpss_time_frames,
            hpss_freq_bins,
            hpss_harmonic_only,
            phase_blur,
            pb_start_hz,
            pb_end_hz,
            pb_harmonic_only,
            hf_resynth,
            hf_lp_hz,
            hf_src_lo_hz,
            hf_src_hi_hz,
            hf_drive,
            hf_hp_hz,
            hf_mix,
            master_enabled,
            hp_hz,
            target_lufs,
            target_rms_dbfs,
            norm_max_gain_db,
            norm_max_atten_db,
            ceiling_dbtp,
            lim_lookahead_ms,
            lim_release_ms,
            tp_os,
            spec_n_fft,
            spec_hop,
            spec_max_frames,
            spec_max_hz,
        ]

        run_btn.click(
            fn=run_once,
            inputs=inputs,
            outputs=[a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json],
        )

        # Apply preset, then rerender preview (restart output loop)
        preset_apply.click(fn=apply_preset, inputs=[preset], outputs=preset_outputs).then(
            fn=run_once,
            inputs=inputs,
            outputs=[a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json],
        )

        # Initialize preset description
        demo.load(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])

        # Auto re-render + restart loop on knob release/change.
        def _bind_auto(comp):
            # Prefer .release for sliders (avoid rerender while dragging); fallback to .change.
            handler = getattr(comp, "release", None)
            if callable(handler):
                handler(fn=run_once, inputs=inputs, outputs=[a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json])
                return
            comp.change(fn=run_once, inputs=inputs, outputs=[a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json])

        # Preview controls
        _bind_auto(preview_t0)
        _bind_auto(preview_dur)
        _bind_auto(loop_xfade_ms)

        # Most knobs
        for c in [
            start_hz,
            end_hz,
            edge_hz,
            n_fft,
            hop,
            flat_start,
            flat_end,
            freq_med_bins,
            thr_db,
            slope,
            density_lo,
            density_hi,
            flux_thr_db,
            flux_range_db,
            noise_resynth,
            mix,
            denoise,
            dn_start_hz,
            dn_end_hz,
            dn_edge_hz,
            dn_floor_db,
            dn_psd_smooth_ms,
            dn_minwin_ms,
            dn_up_db_per_s,
            dn_attack_ms,
            dn_release_ms,
            dn_freq_smooth_bins,
            deres,
            deq_start_hz,
            deq_end_hz,
            deq_edge_hz,
            deq_freq_med_bins,
            deq_thr_db,
            deq_slope,
            deq_max_att_db,
            deq_density_lo,
            deq_density_hi,
            deq_persist_ms,
            deq_persist_thr_db,
            deq_freq_smooth_bins,
            deq_tonal_boost_db,
            deq_time_floor,
            deq_floor_smooth_ms,
            deq_floor_rise_db_per_s,
            expander,
            exp_start_hz,
            exp_end_hz,
            exp_threshold_db,
            exp_ratio,
            exp_attack_ms,
            exp_release_ms,
            hpss,
            hpss_start_hz,
            hpss_end_hz,
            hpss_time_frames,
            hpss_freq_bins,
            hpss_harmonic_only,
            phase_blur,
            pb_start_hz,
            pb_end_hz,
            pb_harmonic_only,
            hf_resynth,
            hf_lp_hz,
            hf_src_lo_hz,
            hf_src_hi_hz,
            hf_drive,
            hf_hp_hz,
            hf_mix,
            master_enabled,
            hp_hz,
            target_lufs,
            target_rms_dbfs,
            norm_max_gain_db,
            norm_max_atten_db,
            ceiling_dbtp,
            lim_lookahead_ms,
            lim_release_ms,
            tp_os,
            spec_n_fft,
            spec_hop,
            spec_max_frames,
            spec_max_hz,
        ]:
            _bind_auto(c)

        full_inputs = [
            audio_in,
            start_hz,
            end_hz,
            edge_hz,
            n_fft,
            hop,
            flat_start,
            flat_end,
            freq_med_bins,
            thr_db,
            slope,
            density_lo,
            density_hi,
            flux_thr_db,
            flux_range_db,
            noise_resynth,
            mix,
            denoise,
            dn_start_hz,
            dn_end_hz,
            dn_edge_hz,
            dn_floor_db,
            dn_psd_smooth_ms,
            dn_minwin_ms,
            dn_up_db_per_s,
            dn_attack_ms,
            dn_release_ms,
            dn_freq_smooth_bins,
            deres,
            deq_start_hz,
            deq_end_hz,
            deq_edge_hz,
            deq_freq_med_bins,
            deq_thr_db,
            deq_slope,
            deq_max_att_db,
            deq_density_lo,
            deq_density_hi,
            deq_persist_ms,
            deq_persist_thr_db,
            deq_freq_smooth_bins,
            deq_tonal_boost_db,
            deq_time_floor,
            deq_floor_smooth_ms,
            deq_floor_rise_db_per_s,
            expander,
            exp_start_hz,
            exp_end_hz,
            exp_threshold_db,
            exp_ratio,
            exp_attack_ms,
            exp_release_ms,
            hpss,
            hpss_start_hz,
            hpss_end_hz,
            hpss_time_frames,
            hpss_freq_bins,
            hpss_harmonic_only,
            phase_blur,
            pb_start_hz,
            pb_end_hz,
            pb_harmonic_only,
            hf_resynth,
            hf_lp_hz,
            hf_src_lo_hz,
            hf_src_hi_hz,
            hf_drive,
            hf_hp_hz,
            hf_mix,
            master_enabled,
            hp_hz,
            target_lufs,
            target_rms_dbfs,
            norm_max_gain_db,
            norm_max_atten_db,
            ceiling_dbtp,
            lim_lookahead_ms,
            lim_release_ms,
            tp_os,
            spec_n_fft,
            spec_hop,
            spec_max_frames,
            spec_max_hz,
        ]
        full_btn.click(fn=render_full_to_files, inputs=full_inputs, outputs=[dl_out, dl_diff, dl_params])

        def save_preset_from_knobs(
            name: str,
            desc: str,
            # values follow: must match keys we store
            start_hz_v,
            end_hz_v,
            edge_hz_v,
            n_fft_v,
            hop_v,
            flat_start_v,
            flat_end_v,
            freq_med_bins_v,
            thr_db_v,
            slope_v,
            density_lo_v,
            density_hi_v,
            flux_thr_db_v,
            flux_range_db_v,
            noise_resynth_v,
            mix_v,
            denoise_v,
            dn_start_hz_v,
            dn_end_hz_v,
            dn_edge_hz_v,
            dn_floor_db_v,
            dn_psd_smooth_ms_v,
            dn_minwin_ms_v,
            dn_up_db_per_s_v,
            dn_attack_ms_v,
            dn_release_ms_v,
            dn_freq_smooth_bins_v,
            deres_v,
            deq_start_hz_v,
            deq_end_hz_v,
            deq_edge_hz_v,
            deq_freq_med_bins_v,
            deq_thr_db_v,
            deq_slope_v,
            deq_max_att_db_v,
            deq_density_lo_v,
            deq_density_hi_v,
            deq_persist_ms_v,
            deq_persist_thr_db_v,
            deq_freq_smooth_bins_v,
            deq_tonal_boost_db_v,
            deq_time_floor_v,
            deq_floor_smooth_ms_v,
            deq_floor_rise_db_per_s_v,
            expander_v,
            exp_start_hz_v,
            exp_end_hz_v,
            exp_threshold_db_v,
            exp_ratio_v,
            exp_attack_ms_v,
            exp_release_ms_v,
            hpss_v,
            hpss_start_hz_v,
            hpss_end_hz_v,
            hpss_time_frames_v,
            hpss_freq_bins_v,
            hpss_harmonic_only_v,
            phase_blur_v,
            pb_start_hz_v,
            pb_end_hz_v,
            pb_harmonic_only_v,
            hf_resynth_v,
            hf_lp_hz_v,
            hf_src_lo_hz_v,
            hf_src_hi_hz_v,
            hf_drive_v,
            hf_hp_hz_v,
            hf_mix_v,
            master_enabled_v,
            hp_hz_v,
            target_lufs_v,
            target_rms_dbfs_v,
            norm_max_gain_db_v,
            norm_max_atten_db_v,
            ceiling_dbtp_v,
            lim_lookahead_ms_v,
            lim_release_ms_v,
            tp_os_v,
        ):
            nonlocal ALL_PRESETS
            name = (name or "").strip()
            if not name:
                return gr.update(), "Please enter a preset name."
            desc = (desc or "").strip()
            values = {
                "start_hz": float(start_hz_v),
                "end_hz": float(end_hz_v),
                "edge_hz": float(edge_hz_v),
                "n_fft": int(n_fft_v),
                "hop": int(hop_v),
                "flat_start": float(flat_start_v),
                "flat_end": float(flat_end_v),
                "freq_med_bins": int(freq_med_bins_v),
                "thr_db": float(thr_db_v),
                "slope": float(slope_v),
                "density_lo": float(density_lo_v),
                "density_hi": float(density_hi_v),
                "flux_thr_db": float(flux_thr_db_v),
                "flux_range_db": float(flux_range_db_v),
                "noise_resynth": float(noise_resynth_v),
                "mix": float(mix_v),
                "denoise": float(denoise_v),
                "dn_start_hz": float(dn_start_hz_v),
                "dn_end_hz": float(dn_end_hz_v),
                "dn_edge_hz": float(dn_edge_hz_v),
                "dn_floor_db": float(dn_floor_db_v),
                "dn_psd_smooth_ms": float(dn_psd_smooth_ms_v),
                "dn_minwin_ms": float(dn_minwin_ms_v),
                "dn_up_db_per_s": float(dn_up_db_per_s_v),
                "dn_attack_ms": float(dn_attack_ms_v),
                "dn_release_ms": float(dn_release_ms_v),
                "dn_freq_smooth_bins": int(dn_freq_smooth_bins_v),
                "deres": float(deres_v),
                "deq_start_hz": float(deq_start_hz_v),
                "deq_end_hz": float(deq_end_hz_v),
                "deq_edge_hz": float(deq_edge_hz_v),
                "deq_freq_med_bins": int(deq_freq_med_bins_v),
                "deq_thr_db": float(deq_thr_db_v),
                "deq_slope": float(deq_slope_v),
                "deq_max_att_db": float(deq_max_att_db_v),
                "deq_density_lo": float(deq_density_lo_v),
                "deq_density_hi": float(deq_density_hi_v),
                "deq_persist_ms": float(deq_persist_ms_v),
                "deq_persist_thr_db": float(deq_persist_thr_db_v),
                "deq_freq_smooth_bins": int(deq_freq_smooth_bins_v),
                "deq_tonal_boost_db": float(deq_tonal_boost_db_v),
                "deq_time_floor": bool(deq_time_floor_v),
                "deq_floor_smooth_ms": float(deq_floor_smooth_ms_v),
                "deq_floor_rise_db_per_s": float(deq_floor_rise_db_per_s_v),
                "expander": bool(expander_v),
                "exp_start_hz": float(exp_start_hz_v),
                "exp_end_hz": float(exp_end_hz_v),
                "exp_threshold_db": float(exp_threshold_db_v),
                "exp_ratio": float(exp_ratio_v),
                "exp_attack_ms": float(exp_attack_ms_v),
                "exp_release_ms": float(exp_release_ms_v),
                "hpss": bool(hpss_v),
                "hpss_start_hz": float(hpss_start_hz_v),
                "hpss_end_hz": float(hpss_end_hz_v),
                "hpss_time_frames": int(hpss_time_frames_v),
                "hpss_freq_bins": int(hpss_freq_bins_v),
                "hpss_harmonic_only": bool(hpss_harmonic_only_v),
                "phase_blur": float(phase_blur_v),
                "pb_start_hz": float(pb_start_hz_v),
                "pb_end_hz": float(pb_end_hz_v),
                "pb_harmonic_only": bool(pb_harmonic_only_v),
                "hf_resynth": bool(hf_resynth_v),
                "hf_lp_hz": float(hf_lp_hz_v),
                "hf_src_lo_hz": float(hf_src_lo_hz_v),
                "hf_src_hi_hz": float(hf_src_hi_hz_v),
                "hf_drive": float(hf_drive_v),
                "hf_hp_hz": float(hf_hp_hz_v),
                "hf_mix": float(hf_mix_v),
                "master_enabled": bool(master_enabled_v),
                "hp_hz": float(hp_hz_v),
                "target_lufs": float(target_lufs_v),
                "target_rms_dbfs": float(target_rms_dbfs_v),
                "norm_max_gain_db": float(norm_max_gain_db_v),
                "norm_max_atten_db": float(norm_max_atten_db_v),
                "ceiling_dbtp": float(ceiling_dbtp_v),
                "lim_lookahead_ms": float(lim_lookahead_ms_v),
                "lim_release_ms": float(lim_release_ms_v),
                "tp_os": int(tp_os_v),
            }
            _save_user_preset(USER_PRESETS_PATH, name, desc, values)
            ALL_PRESETS = dict(BASE_PRESETS)
            ALL_PRESETS.update(_load_user_presets(USER_PRESETS_PATH))
            return gr.update(choices=list(ALL_PRESETS.keys()), value=name), f"Saved preset to `{USER_PRESETS_PATH}`."

        preset_save_inputs = [
            preset_name,
            preset_user_desc,
            start_hz,
            end_hz,
            edge_hz,
            n_fft,
            hop,
            flat_start,
            flat_end,
            freq_med_bins,
            thr_db,
            slope,
            density_lo,
            density_hi,
            flux_thr_db,
            flux_range_db,
            noise_resynth,
            mix,
            denoise,
            dn_start_hz,
            dn_end_hz,
            dn_edge_hz,
            dn_floor_db,
            dn_psd_smooth_ms,
            dn_minwin_ms,
            dn_up_db_per_s,
            dn_attack_ms,
            dn_release_ms,
            dn_freq_smooth_bins,
            deres,
            deq_start_hz,
            deq_end_hz,
            deq_edge_hz,
            deq_freq_med_bins,
            deq_thr_db,
            deq_slope,
            deq_max_att_db,
            deq_density_lo,
            deq_density_hi,
            deq_persist_ms,
            deq_persist_thr_db,
            deq_freq_smooth_bins,
            deq_tonal_boost_db,
            deq_time_floor,
            deq_floor_smooth_ms,
            deq_floor_rise_db_per_s,
            expander,
            exp_start_hz,
            exp_end_hz,
            exp_threshold_db,
            exp_ratio,
            exp_attack_ms,
            exp_release_ms,
            hpss,
            hpss_start_hz,
            hpss_end_hz,
            hpss_time_frames,
            hpss_freq_bins,
            hpss_harmonic_only,
            phase_blur,
            pb_start_hz,
            pb_end_hz,
            pb_harmonic_only,
            hf_resynth,
            hf_lp_hz,
            hf_src_lo_hz,
            hf_src_hi_hz,
            hf_drive,
            hf_hp_hz,
            hf_mix,
            master_enabled,
            hp_hz,
            target_lufs,
            target_rms_dbfs,
            norm_max_gain_db,
            norm_max_atten_db,
            ceiling_dbtp,
            lim_lookahead_ms,
            lim_release_ms,
            tp_os,
        ]
        preset_save.click(fn=save_preset_from_knobs, inputs=preset_save_inputs, outputs=[preset, preset_save_status]).then(
            fn=_preset_desc_md,
            inputs=[preset],
            outputs=[preset_desc],
        )

        gr.Markdown(
            "### Tip\n"
            "- Use a short preview (3–8s) around a known-problem spot for fast iteration.\n"
            "- When you like it, run the CLI on the full file using the same values."
        )

    return demo


def main() -> int:
    import gradio as gr

    demo = build_ui()
    demo.queue()
    demo.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


