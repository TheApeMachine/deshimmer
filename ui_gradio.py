#!/usr/bin/env python3
# ruff: noqa
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

# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportPrivateUsage=false
# pyright: reportUnusedCallResult=false

import base64
import io
import math
import os
import time
from dataclasses import asdict
from typing import Any

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


def _slice_preview(x: np.ndarray, sr: int, t0: float, dur: float) -> tuple[np.ndarray, int, int]:
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
    extent = (
        float(t[0]) if t.size else 0.0,
        float(t[-1]) if t.size else 0.0,
        float(f[0]) if f.size else 0.0,
        float(f[-1]) if f.size else 0.0,
    )
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


def _audio_player_html(title: str, wav_data_uri: str, *, autoplay: bool, loop: bool = True, player_id: str = "") -> str:
    """
    Generate HTML for an audio player with a stable ID for position preservation.
    Position saving/restoring is handled by the global _AUDIO_POSITION_SCRIPT.
    """
    autoplay_attr = " autoplay" if autoplay else ""
    loop_attr = " loop" if loop else ""
    id_attr = f' id="{player_id}"' if player_id else ""
    
    return (
        f"<div style='display:flex;flex-direction:column;gap:6px'>"
        f"<div><b>{title}</b></div>"
        f"<audio{id_attr} class='deshimmer-audio' controls{autoplay_attr}{loop_attr} playsinline style='width:100%' src='{wav_data_uri}'></audio>"
        f"</div>"
    )


# Global JavaScript that persists and is injected once via gr.HTML at page load.
# Uses MutationObserver to watch for audio element src changes and restore position.
# Global JavaScript that persists and is injected once via gr.HTML at page load.
# Uses MutationObserver to watch for audio element src changes and restore position.
_AUDIO_POSITION_SCRIPT = """
<script>
(function() {
    console.log('[Deshimmer] Robust audio script loaded');
    
    // State storage by INDEX (0=Input, 1=Output, 2=Diff)
    // We use a global variable on window to survive some re-renders if the script wrapper stays
    if (!window._deshimmerState) {
        window._deshimmerState = {};
    }
    
    function saveState(index, audio) {
        window._deshimmerState[index] = {
            time: audio.currentTime,
            playing: !audio.paused,
            src: audio.src,
            timestamp: Date.now()
        };
        // Also save to session storage for page reloads
        try {
            sessionStorage.setItem('deshimmer_idx_' + index, JSON.stringify(window._deshimmerState[index]));
        } catch(e) {}
    }
    
    function loadState(index) {
        // Try memory first, then session storage
        if (window._deshimmerState[index]) return window._deshimmerState[index];
        try {
            var s = sessionStorage.getItem('deshimmer_idx_' + index);
            if (s) return JSON.parse(s);
        } catch(e) {}
        return null;
    }
    
    function setupPlayers() {
        // Find ALL audio players with our class
        setTimeout(() => {
            console.log('[Deshimmer] Setting up players');
            const players = document.querySelectorAll('.deshimmer-audio');
            
            players.forEach(function(audio, index) {
                // Prevent double-setup
                if (audio.getAttribute('data-setup') === 'true') return;
                audio.setAttribute('data-setup', 'true');
                
                console.log('[Deshimmer] Found player index ' + index);
                
                var state = loadState(index);
                
                // If we have state, logic to restore it
                if (state) {
                    // If this is a FRESH load (new src), restore position
                    // We check if src changed slightly or just assume if it's a re-render we want the old time
                    // The safest is: if we have a saved time, and we are near 0, jump to saved time.
                    
                    var restore = function() {
                        if (state.time > 0 && audio.duration > 0) {
                            var newTime = state.time % audio.duration;
                            if (Math.abs(audio.currentTime - newTime) > 0.5) {
                                console.log('[Deshimmer] Restoring idx ' + index + ' to ' + newTime);
                                audio.currentTime = newTime;
                            }
                        }
                        if (state.playing) {
                            var promise = audio.play();
                            if (promise) promise.catch(e => console.log('Autoplay prevented:', e));
                        }
                    };

                    // Try immediately if ready
                    if (audio.readyState >= 1) restore();
                    
                    // And on metadata load
                    audio.addEventListener('loadedmetadata', restore);
                    
                    // And on 'canplay' for good measure (covers some browser quirks)
                    audio.addEventListener('canplay', function() {
                        // Only restore if we haven't drifted far (prevents fighting user seeks)
                        if (audio.currentTime < 0.5 && state.time > 0.5) restore();
                    });
                })
            
            // Listeners to save state
            audio.addEventListener('timeupdate', function() {
                saveState(index, audio);
            });
            audio.addEventListener('play', function() {
                saveState(index, audio);
            });
            audio.addEventListener('pause', function() {
                saveState(index, audio);
            });
        });
    }

    // Initial run
    setupPlayers();
})();
</script>
"""


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


def _png_bytes_to_rgb(png_bytes: bytes) -> np.ndarray | None:
    if not png_bytes:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.asarray(im)


def _render_metrics_md(info: dict[str, object]) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return f"{float(v):.2f}"
        return str(v)

    mi = info.get("measure_in", {})  # type: ignore[assignment]
    mr = info.get("measure_after_repair", {})  # type: ignore[assignment]
    mo = info.get("measure_out", {})  # type: ignore[assignment]
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
    delta_listen: bool,
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
        delta_listen=bool(delta_listen),
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
    audio_in: tuple[int, np.ndarray] | None,
    # preview
    preview_t0: float,
    preview_dur: float,
    loop_xfade_ms: float,
    full_song_mode: bool,
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
    delta_listen: bool,
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
) -> tuple[str, str, str, np.ndarray | None, np.ndarray | None, np.ndarray | None, str, dict[str, object]]:
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    # In full song mode, process the entire song; otherwise just the preview slice
    if full_song_mode:
        x_seg = x
        s0, s1 = 0, x.shape[0]
    else:
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
        delta_listen=delta_listen,
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

    # If delta_listen is enabled, master.py returns removed-only (input - processed).
    if bool(delta_listen):
        removed = y2
        processed = _to_float_audio(x_seg[: removed.shape[0], :] - removed)
        out_sig = removed
        aux_sig = processed
        if full_song_mode:
            out_title = "Full song: removed-only"
            aux_title = "Full song: processed"
        else:
            out_title = "Preview: removed-only (loop)"
            aux_title = "Preview: processed (loop)"
    else:
        processed = y2
        removed = (x_seg[: processed.shape[0], :] - processed).astype(np.float32)
        out_sig = processed
        aux_sig = removed
        if full_song_mode:
            out_title = "Full song: output"
            aux_title = "Full song: diff / removed"
        else:
            out_title = "Preview: output (loop)"
            aux_title = "Preview: diff / removed (loop)"

    # Make loop seam smoother (rotate+crossfade) for playback - skip for full song mode
    if full_song_mode:
        # No crossfade for full song - just use the audio as-is
        x_play = x_seg
        out_play = out_sig
        aux_play = aux_sig
        in_title = "Full song: input"
        do_loop = False
    else:
        x_play = _loop_crossfade_rotate(x_seg, sr, loop_xfade_ms)
        out_play = _loop_crossfade_rotate(out_sig, sr, loop_xfade_ms)
        aux_play = _loop_crossfade_rotate(aux_sig, sr, loop_xfade_ms)
        in_title = "Preview: input (loop)"
        do_loop = True

    # Spectrograms
    in_png = _spectrogram_png_bytes(x_seg, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))
    out_png = _spectrogram_png_bytes(out_sig, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))
    diff_png = _spectrogram_png_bytes(aux_sig, sr, n_fft=int(spec_n_fft), hop=int(spec_hop), max_frames=int(spec_max_frames), max_hz=float(spec_max_hz))

    metrics_md = _render_metrics_md(info)
    params_json: dict[str, object] = {"params": asdict(p), "master_params": asdict(mp)}

    return (
        _audio_player_html(in_title, _audio_to_wav_data_uri(x_play, sr), autoplay=False, loop=do_loop, player_id="deshimmer_input"),
        _audio_player_html(out_title, _audio_to_wav_data_uri(out_play, sr), autoplay=True, loop=do_loop, player_id="deshimmer_output"),
        _audio_player_html(aux_title, _audio_to_wav_data_uri(aux_play, sr), autoplay=False, loop=do_loop, player_id="deshimmer_diff"),
        _png_bytes_to_rgb(in_png),
        _png_bytes_to_rgb(out_png),
        _png_bytes_to_rgb(diff_png),
        metrics_md,
        params_json,  # dict shown in JSON component
    )


def render_full_to_files(
    audio_in: tuple[int, np.ndarray] | None,
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
    delta_listen: bool,
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
) -> tuple[str, str, str]:
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
        delta_listen=delta_listen,
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
    if bool(delta_listen):
        removed = y2
        processed = _to_float_audio(x[: removed.shape[0], :] - removed)
        out_audio = removed
        diff_audio = processed
    else:
        processed = y2
        removed = (x[: processed.shape[0], :] - processed).astype(np.float32)
        out_audio = processed
        diff_audio = removed

    outdir = os.path.join(os.path.dirname(__file__), "ui_downloads")
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(outdir, f"output-{stamp}.wav")
    diff_path = os.path.join(outdir, f"diff-{stamp}.wav")
    params_path = os.path.join(outdir, f"params-{stamp}.json")

    sf.write(out_path, out_audio, sr, subtype="PCM_24")
    sf.write(diff_path, diff_audio, sr, subtype="PCM_24")

    import json

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"params": asdict(p), "master_params": asdict(mp)}, f, indent=2, sort_keys=True)

    return out_path, diff_path, params_path


def build_ui() -> Any:
    import gradio as gr

    def _load_user_presets(path: str) -> dict[str, dict[str, object]]:
        try:
            import json

            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            out: dict[str, dict[str, object]] = {}
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

    def _save_user_preset(path: str, name: str, desc: str, values: dict[str, object]) -> None:
        import json

        data = _load_user_presets(path)
        data[name] = {"desc": desc, "values": values}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    USER_PRESETS_PATH = os.path.join(os.path.dirname(__file__), "ui_presets.json")

    base_presets: dict[str, dict[str, object]] = {
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

    all_presets: dict[str, dict[str, object]] = dict(base_presets)
    all_presets.update(_load_user_presets(USER_PRESETS_PATH))

    with gr.Blocks(title="Deshimmer - master.py UI") as demo:
        # Inject global JavaScript for audio position preservation (runs once)
        # Note: This is invisible but must not use visible=False or script won't execute
        gr.HTML(_AUDIO_POSITION_SCRIPT)
        
        gr.Markdown(
            "## Deshimmer UI (master.py)\n"
            "Load a file, pick a preview region, tweak knobs, then listen to input/output/diff and inspect spectrograms."
        )

        with gr.Row():
            preset = gr.Dropdown(choices=list(all_presets.keys()), value="02 - Default shimmer (recommended start)", label="Presets (simple → complex)")
            preset_apply = gr.Button("Apply preset", variant="secondary")
        preset_desc = gr.Markdown()

        with gr.Row():
            audio_in = gr.Audio(label="Input audio", type="numpy")
            with gr.Column():
                full_song_mode = gr.Checkbox(
                    value=False, 
                    label="Full Song Mode",
                    info="Process the entire song instead of a short preview. Slower but lets you hear the effect on the whole track."
                )
                preview_t0 = gr.Slider(
                    0.0, 600.0, value=0.0, step=0.05, 
                    label="Preview start (s)",
                    info="Where in the song to start the preview. Move this to test different sections."
                )
                preview_dur = gr.Slider(
                    0.5, 20.0, value=6.0, step=0.05, 
                    label="Preview duration (s)",
                    info="How long the preview clip should be. Shorter = faster processing, longer = hear more context."
                )
                loop_xfade_ms = gr.Slider(
                    0.0, 250.0, value=35.0, step=1.0, 
                    label="Loop crossfade (ms)",
                    info="Smooth the loop point so it doesn't click. Higher = smoother loop but slightly alters the audio."
                )
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

        with gr.Accordion("🎯 Shimmer Removal (main tool for AI sparkle/shimmer)", open=True):
            gr.Markdown("*Targets the annoying 'sparkly' or 'shimmery' artifacts common in AI-generated music. Start here!*")
            with gr.Row():
                start_hz = gr.Number(
                    value=5100.0, 
                    label="Start frequency (Hz)",
                    info="Lower edge of the shimmer band. AI shimmer typically lives around 5000-7000 Hz. Lower = catch more low-mid harshness."
                )
                end_hz = gr.Number(
                    value=7200.0, 
                    label="End frequency (Hz)",
                    info="Upper edge of the shimmer band. Higher = catch more high-frequency sparkle. Don't go above your audio's limit."
                )
                edge_hz = gr.Number(
                    value=200.0, 
                    label="Edge softness (Hz)",
                    info="How gradually the effect fades at band edges. Higher = smoother transition, less obvious processing."
                )
            with gr.Row():
                n_fft = gr.Dropdown(
                    [1024, 2048, 4096, 8192], value=2048, 
                    label="FFT size",
                    info="Analysis window size. Larger = better frequency detail but slower. 2048 is a good balance."
                )
                hop = gr.Dropdown(
                    [256, 512, 1024, 2048], value=512, 
                    label="Hop size",
                    info="How much the analysis window moves. Smaller = smoother but slower. 512 works well for most cases."
                )
            with gr.Row():
                flat_start = gr.Slider(
                    0.0, 1.0, value=0.25, step=0.01, 
                    label="Noise detection: start",
                    info="How 'noisy' audio must be before processing kicks in. Lower = more aggressive, catches more but may affect wanted sounds."
                )
                flat_end = gr.Slider(
                    0.0, 1.0, value=0.70, step=0.01, 
                    label="Noise detection: full",
                    info="When audio is this 'noisy', full processing is applied. Higher = only process very noisy parts."
                )
            with gr.Row():
                freq_med_bins = gr.Slider(
                    3, 61, value=9, step=2, 
                    label="Peak detection width",
                    info="How wide to look when finding shimmer peaks. Larger = ignore broader peaks (like wanted harmonics), catch only narrow spikes."
                )
                thr_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1, 
                    label="Threshold (dB)",
                    info="How much louder than surroundings a peak must be to get reduced. Lower = more aggressive, higher = only obvious spikes."
                )
                slope = gr.Slider(
                    0.0, 2.0, value=0.6, step=0.01, 
                    label="Reduction strength",
                    info="How hard to push down detected shimmer. Higher = more reduction but risk of dullness. Start around 0.5-0.7."
                )
            with gr.Row():
                density_lo = gr.Slider(
                    0.0, 0.5, value=0.02, step=0.005, 
                    label="Density: sparse",
                    info="When few peaks are detected, apply full processing. Shimmer is usually sparse."
                )
                density_hi = gr.Slider(
                    0.0, 0.5, value=0.15, step=0.005, 
                    label="Density: dense (back off)",
                    info="When many peaks detected, back off—probably real music content, not artifacts."
                )
            with gr.Row():
                flux_thr_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1, 
                    label="Transient protection: threshold",
                    info="Energy jump (in dB) that triggers transient protection. Protects drum hits and attacks from being dulled."
                )
                flux_range_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1, 
                    label="Transient protection: range",
                    info="How gradually transient protection kicks in. Higher = more gradual fade-in of protection."
                )
            with gr.Row():
                noise_resynth = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, 
                    label="Texture softening",
                    info="Blend in random-phase noise to soften 'crystalline' texture. 0 = off, 0.2-0.4 = subtle softening. Can help with metallic artifacts."
                )
                mix = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.01, 
                    label="Wet/dry mix",
                    info="0 = original audio (bypass), 1 = fully processed. Use values in between for subtle blending."
                )
                delta_listen = gr.Checkbox(
                    value=False, 
                    label="Listen to removed audio",
                    info="Hear what's being removed instead of the result. Useful for checking you're not removing wanted sounds!"
                )

        with gr.Accordion("🔇 Smart Denoise (hiss/noise floor reduction)", open=False):
            gr.Markdown("*Reduces background hiss and noise floor artifacts. Good for AI-generated 'swishy' or 'windy' sounds.*")
            with gr.Row():
                denoise = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, 
                    label="Denoise strength",
                    info="Overall strength. 0 = off. Start low (0.2-0.3) and increase. Too high = 'underwater' sound."
                )
                dn_floor_db = gr.Slider(
                    -60.0, 0.0, value=-18.0, step=0.5, 
                    label="Noise floor limit (dB)",
                    info="Never reduce below this level. Prevents complete silence and 'pumping'. More negative = deeper cuts allowed."
                )
            with gr.Row():
                dn_start_hz = gr.Number(
                    value=120.0, 
                    label="Start frequency (Hz)",
                    info="Where to start denoising. Keep above ~100 Hz to avoid affecting bass."
                )
                dn_end_hz = gr.Number(
                    value=16000.0, 
                    label="End frequency (Hz)",
                    info="Where to stop denoising. Usually covers most of audible range."
                )
                dn_edge_hz = gr.Number(
                    value=200.0, 
                    label="Edge softness (Hz)",
                    info="Gradual fade at band edges for smoother processing."
                )
            with gr.Row():
                dn_psd_smooth_ms = gr.Slider(
                    0.0, 300.0, value=50.0, step=1.0, 
                    label="Noise tracking speed (ms)",
                    info="How fast noise estimation adapts. Lower = faster tracking, higher = more stable."
                )
                dn_minwin_ms = gr.Slider(
                    50.0, 2000.0, value=400.0, step=10.0, 
                    label="Minimum window (ms)",
                    info="Time window for finding quietest noise level. Longer = better at finding true noise floor but slower to adapt."
                )
                dn_up_db_per_s = gr.Slider(
                    0.0, 24.0, value=3.0, step=0.1, 
                    label="Noise rise rate (dB/s)",
                    info="How fast estimated noise can increase. Prevents sudden jumps. Lower = more conservative."
                )
            with gr.Row():
                dn_attack_ms = gr.Slider(
                    0.0, 200.0, value=5.0, step=1.0, 
                    label="Attack speed (ms)",
                    info="How fast reduction kicks in when noise detected. Lower = faster but may clip transients."
                )
                dn_release_ms = gr.Slider(
                    0.0, 1000.0, value=120.0, step=5.0, 
                    label="Release speed (ms)",
                    info="How fast reduction backs off. Higher = smoother but may 'pump' on dynamic material."
                )
                dn_freq_smooth_bins = gr.Slider(
                    1, 21, value=3, step=1, 
                    label="Frequency smoothing",
                    info="Smooth reduction across frequencies. Higher = less 'musical noise'/chirping but less precise."
                )

        with gr.Accordion("📢 De-Resonator (ringing/whine removal)", open=False):
            gr.Markdown("*Targets persistent ringing tones and 'whine' artifacts. Great for Suno/Udio's characteristic 3-4kHz whine.*")
            with gr.Row():
                deres = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, 
                    label="De-resonator strength",
                    info="Overall strength. 0 = off. Effective range is 0.3-0.8. Attacks narrow, persistent frequency peaks."
                )
                deq_max_att_db = gr.Slider(
                    0.0, 24.0, value=8.0, step=0.1, 
                    label="Maximum cut (dB)",
                    info="Cap on how much to reduce resonances. Higher = more aggressive cuts. 6-12 dB is typical."
                )
            with gr.Row():
                deq_start_hz = gr.Number(
                    value=180.0, 
                    label="Start frequency (Hz)",
                    info="Where to look for resonances. Suno/Udio whine often around 3000-4000 Hz."
                )
                deq_end_hz = gr.Number(
                    value=12000.0, 
                    label="End frequency (Hz)",
                    info="Upper limit for resonance detection."
                )
                deq_edge_hz = gr.Number(
                    value=150.0, 
                    label="Edge softness (Hz)",
                    info="Gradual fade at band edges."
                )
            with gr.Row():
                deq_freq_med_bins = gr.Slider(
                    3, 121, value=31, step=2, 
                    label="Peak detection width",
                    info="How wide to look for 'normal' level. Larger = catch wider resonances. For narrow whines, try 31-61."
                )
                deq_thr_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1, 
                    label="Threshold (dB)",
                    info="How much louder than neighbors a peak must be to get cut. Lower = more aggressive."
                )
                deq_slope = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.01, 
                    label="Reduction strength",
                    info="How hard to cut detected resonances. Higher = stronger cuts."
                )
            with gr.Row():
                deq_density_lo = gr.Slider(
                    0.0, 0.5, value=0.03, step=0.005, 
                    label="Density: sparse (full processing)",
                    info="When few peaks, apply full reduction—probably artifacts."
                )
                deq_density_hi = gr.Slider(
                    0.0, 0.5, value=0.20, step=0.005, 
                    label="Density: dense (back off)",
                    info="When many peaks, back off—probably real harmonics."
                )
            with gr.Row():
                deq_persist_ms = gr.Slider(
                    0.0, 5000.0, value=600.0, step=25.0, 
                    label="Persistence memory (ms)",
                    info="How long a peak must persist to be considered a resonance. Catches steady whines, ignores moving melodies."
                )
                deq_persist_thr_db = gr.Slider(
                    0.0, 24.0, value=2.5, step=0.1, 
                    label="Persistence threshold (dB)",
                    info="Accumulated level needed before peak is treated as resonance. Higher = more conservative."
                )
                deq_freq_smooth_bins = gr.Slider(
                    1, 21, value=5, step=1, 
                    label="Frequency smoothing",
                    info="Smooth cuts across frequencies. Higher = less surgical but fewer artifacts."
                )
                deq_tonal_boost_db = gr.Slider(
                    0.0, 24.0, value=6.0, step=0.1, 
                    label="Tonal protection (dB)",
                    info="Raise threshold when audio sounds 'tonal' (not noisy). Protects real harmonics."
                )
            with gr.Row():
                deq_time_floor = gr.Checkbox(
                    value=False, 
                    label="Enable time-floor mode",
                    info="Track bins that are ALWAYS loud—catches stationary ringing lines that never quiet down."
                )
                deq_floor_smooth_ms = gr.Slider(
                    0.0, 500.0, value=80.0, step=5.0, 
                    label="Floor tracking speed (ms)",
                    info="How fast the 'always loud' detector adapts. Lower = faster."
                )
                deq_floor_rise_db_per_s = gr.Slider(
                    0.0, 12.0, value=1.0, step=0.1, 
                    label="Floor rise rate (dB/s)",
                    info="How fast the floor estimate can rise. Lower = stricter about what counts as 'always on'."
                )

        with gr.Accordion("🔬 Advanced Tools (experimental - use with caution)", open=False):
            gr.Markdown("*These are more aggressive/experimental tools. Most users won't need them.*")
            
            gr.Markdown("**Downward Expander** - Push down quiet parts in a frequency band (reduces lingering artifacts)")
            with gr.Row():
                expander = gr.Checkbox(
                    value=False, 
                    label="Enable expander",
                    info="Turn on the downward expander. Reduces quiet sounds in the target band."
                )
                exp_threshold_db = gr.Slider(
                    -90.0, 0.0, value=-45.0, step=0.5, 
                    label="Threshold (dB)",
                    info="Level below which sounds get reduced. More negative = only very quiet sounds affected."
                )
                exp_ratio = gr.Slider(
                    1.0, 6.0, value=2.0, step=0.05, 
                    label="Ratio",
                    info="How much to reduce. 2:1 = moderate, 4:1 = aggressive. Higher = more expansion."
                )
            with gr.Row():
                exp_start_hz = gr.Number(
                    value=3000.0, 
                    label="Start frequency (Hz)",
                    info="Lower edge of expander band."
                )
                exp_end_hz = gr.Number(
                    value=8000.0, 
                    label="End frequency (Hz)",
                    info="Upper edge of expander band."
                )
                exp_attack_ms = gr.Slider(
                    0.0, 200.0, value=10.0, step=1.0, 
                    label="Attack (ms)",
                    info="How fast expansion kicks in. Lower = faster but may affect transients."
                )
                exp_release_ms = gr.Slider(
                    0.0, 1000.0, value=150.0, step=5.0, 
                    label="Release (ms)",
                    info="How fast expansion backs off. Higher = smoother."
                )

            gr.Markdown("**Harmonic/Percussive Separation** - Protect drums/transients from processing")
            with gr.Row():
                hpss = gr.Checkbox(
                    value=False, 
                    label="Enable HPSS",
                    info="Separate harmonic (sustained) from percussive (transient) content."
                )
                hpss_harmonic_only = gr.Checkbox(
                    value=True, 
                    label="Process harmonic only",
                    info="Only process the sustained/harmonic parts, leave drums/transients alone."
                )
            with gr.Row():
                hpss_start_hz = gr.Number(
                    value=3000.0, 
                    label="Start frequency (Hz)",
                    info="Where to apply HPSS separation."
                )
                hpss_end_hz = gr.Number(
                    value=8000.0, 
                    label="End frequency (Hz)",
                    info="Upper limit for HPSS."
                )
                hpss_time_frames = gr.Slider(
                    3, 61, value=21, step=2, 
                    label="Time window",
                    info="Median filter over time. Larger = better separation but more smearing."
                )
                hpss_freq_bins = gr.Slider(
                    3, 61, value=17, step=2, 
                    label="Frequency window",
                    info="Median filter over frequency. Larger = better separation but less precise."
                )

            gr.Markdown("**Phase Blur** - Soften harsh textures by randomizing phase")
            with gr.Row():
                phase_blur = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01, 
                    label="Amount",
                    info="How much to randomize phase. 0 = off. Subtle values (0.1-0.3) can soften metallic textures."
                )
                pb_harmonic_only = gr.Checkbox(
                    value=True, 
                    label="Harmonic only",
                    info="Only blur harmonic content, preserve transients."
                )
            with gr.Row():
                pb_start_hz = gr.Number(
                    value=3000.0, 
                    label="Start frequency (Hz)",
                    info="Where to apply phase blur."
                )
                pb_end_hz = gr.Number(
                    value=8000.0, 
                    label="End frequency (Hz)",
                    info="Upper limit for phase blur."
                )

            gr.Markdown("**HF Resynthesis** - Nuclear option: remove HF entirely and regenerate from lower frequencies")
            with gr.Row():
                hf_resynth = gr.Checkbox(
                    value=False, 
                    label="Enable HF resynthesis",
                    info="⚠️ Destructive! Removes high frequencies and creates new ones from lower bands. Last resort."
                )
                hf_mix = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.01, 
                    label="Mix amount",
                    info="How much regenerated HF to blend in. Lower = subtler."
                )
            with gr.Row():
                hf_lp_hz = gr.Number(
                    value=3000.0, 
                    label="Low-pass cutoff (Hz)",
                    info="Keep everything below this frequency, remove above."
                )
                hf_hp_hz = gr.Number(
                    value=3000.0, 
                    label="Regenerated HF starts at (Hz)",
                    info="Where regenerated harmonics begin."
                )
                hf_drive = gr.Slider(
                    0.1, 10.0, value=2.0, step=0.05, 
                    label="Drive",
                    info="Saturation amount for generating harmonics. Higher = more harmonics but harsher."
                )
            with gr.Row():
                hf_src_lo_hz = gr.Number(
                    value=1000.0, 
                    label="Source band start (Hz)",
                    info="Lower edge of the band used to generate new harmonics."
                )
                hf_src_hi_hz = gr.Number(
                    value=2000.0, 
                    label="Source band end (Hz)",
                    info="Upper edge of the source band."
                )

        with gr.Accordion("🎚️ Delivery Mastering (loudness/limiting)", open=False):
            gr.Markdown("*Final loudness normalization and limiting for delivery. Only enable when you're done tweaking!*")
            master_enabled = gr.Checkbox(
                value=False, 
                label="Enable mastering",
                info="Turn on loudness normalization and limiting. Leave off while tweaking parameters."
            )
            with gr.Row():
                hp_hz = gr.Slider(
                    0.0, 80.0, value=20.0, step=0.5, 
                    label="High-pass filter (Hz)",
                    info="Remove sub-bass rumble. 20-30 Hz is typical. Set to 0 to disable."
                )
                tp_os = gr.Dropdown(
                    [1, 2, 4, 8], value=4, 
                    label="True peak oversampling",
                    info="Oversampling for accurate peak detection. 4x is standard, 8x is more accurate but slower."
                )
            with gr.Row():
                target_lufs = gr.Slider(
                    -30.0, 999.0, value=-14.0, step=0.1, 
                    label="Target loudness (LUFS)",
                    info="-14 LUFS = Spotify/YouTube. -16 = Apple. Set >=998 to disable loudness normalization."
                )
                target_rms_dbfs = gr.Slider(
                    -40.0, -6.0, value=-16.0, step=0.1, 
                    label="Fallback RMS target (dBFS)",
                    info="Used if LUFS measurement fails. -16 to -14 is typical."
                )
            with gr.Row():
                norm_max_gain_db = gr.Slider(
                    0.0, 24.0, value=12.0, step=0.5, 
                    label="Max boost (dB)",
                    info="Maximum gain increase allowed. Prevents over-boosting quiet tracks."
                )
                norm_max_atten_db = gr.Slider(
                    0.0, 60.0, value=24.0, step=0.5, 
                    label="Max cut (dB)",
                    info="Maximum gain reduction allowed. Prevents over-cutting loud tracks."
                )
            with gr.Row():
                ceiling_dbtp = gr.Slider(
                    -12.0, 0.0, value=-1.0, step=0.1, 
                    label="True peak ceiling (dBTP)",
                    info="Maximum peak level. -1 dBTP is safe for streaming, -0.5 for CD."
                )
                lim_lookahead_ms = gr.Slider(
                    0.0, 50.0, value=5.0, step=0.5, 
                    label="Limiter lookahead (ms)",
                    info="How far ahead the limiter looks. Higher = cleaner limiting but adds latency."
                )
                lim_release_ms = gr.Slider(
                    10.0, 1000.0, value=100.0, step=5.0, 
                    label="Limiter release (ms)",
                    info="How fast limiter recovers. Lower = more aggressive, higher = smoother."
                )

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
            rt_export = gr.Button("Export knobs to realtime_params.json (for realtime_player.py)", variant="secondary")
            rt_export_status = gr.Markdown()

        def _preset_desc_md(name: str) -> str:
            p = all_presets.get(name, {})
            desc = p.get("desc", "")
            return f"**{name}**  \n{desc}" if desc else f"**{name}**"

        def apply_preset(name: str):
            p = all_presets.get(name, {})
            vals = dict(p.get("values", {}))
            # return updates in the same order as outputs list below
            def g(key: str, current: object) -> object:
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
                g("delta_listen", delta_listen.value),
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
            delta_listen,
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
            full_song_mode,
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
            delta_listen,
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
        _bind_auto(full_song_mode)

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
            delta_listen,
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
            delta_listen,
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
            start_hz_v: object,
            end_hz_v: object,
            edge_hz_v: object,
            n_fft_v: object,
            hop_v: object,
            flat_start_v: object,
            flat_end_v: object,
            freq_med_bins_v: object,
            thr_db_v: object,
            slope_v: object,
            density_lo_v: object,
            density_hi_v: object,
            flux_thr_db_v: object,
            flux_range_db_v: object,
            noise_resynth_v: object,
            mix_v: object,
            delta_listen_v: object,
            denoise_v: object,
            dn_start_hz_v: object,
            dn_end_hz_v: object,
            dn_edge_hz_v: object,
            dn_floor_db_v: object,
            dn_psd_smooth_ms_v: object,
            dn_minwin_ms_v: object,
            dn_up_db_per_s_v: object,
            dn_attack_ms_v: object,
            dn_release_ms_v: object,
            dn_freq_smooth_bins_v: object,
            deres_v: object,
            deq_start_hz_v: object,
            deq_end_hz_v: object,
            deq_edge_hz_v: object,
            deq_freq_med_bins_v: object,
            deq_thr_db_v: object,
            deq_slope_v: object,
            deq_max_att_db_v: object,
            deq_density_lo_v: object,
            deq_density_hi_v: object,
            deq_persist_ms_v: object,
            deq_persist_thr_db_v: object,
            deq_freq_smooth_bins_v: object,
            deq_tonal_boost_db_v: object,
            deq_time_floor_v: object,
            deq_floor_smooth_ms_v: object,
            deq_floor_rise_db_per_s_v: object,
            expander_v: object,
            exp_start_hz_v: object,
            exp_end_hz_v: object,
            exp_threshold_db_v: object,
            exp_ratio_v: object,
            exp_attack_ms_v: object,
            exp_release_ms_v: object,
            hpss_v: object,
            hpss_start_hz_v: object,
            hpss_end_hz_v: object,
            hpss_time_frames_v: object,
            hpss_freq_bins_v: object,
            hpss_harmonic_only_v: object,
            phase_blur_v: object,
            pb_start_hz_v: object,
            pb_end_hz_v: object,
            pb_harmonic_only_v: object,
            hf_resynth_v: object,
            hf_lp_hz_v: object,
            hf_src_lo_hz_v: object,
            hf_src_hi_hz_v: object,
            hf_drive_v: object,
            hf_hp_hz_v: object,
            hf_mix_v: object,
            master_enabled_v: object,
            hp_hz_v: object,
            target_lufs_v: object,
            target_rms_dbfs_v: object,
            norm_max_gain_db_v: object,
            norm_max_atten_db_v: object,
            ceiling_dbtp_v: object,
            lim_lookahead_ms_v: object,
            lim_release_ms_v: object,
            tp_os_v: object,
        ):
            nonlocal all_presets
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
                "delta_listen": bool(delta_listen_v),
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
            all_presets = dict(base_presets)
            all_presets.update(_load_user_presets(USER_PRESETS_PATH))
            return gr.update(choices=list(all_presets.keys()), value=name), f"Saved preset to `{USER_PRESETS_PATH}`."

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
            delta_listen,
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

        def export_realtime_params(
            # current knob state (same as save)
            start_hz_v: object,
            end_hz_v: object,
            edge_hz_v: object,
            n_fft_v: object,
            hop_v: object,
            flat_start_v: object,
            flat_end_v: object,
            freq_med_bins_v: object,
            thr_db_v: object,
            slope_v: object,
            density_lo_v: object,
            density_hi_v: object,
            flux_thr_db_v: object,
            flux_range_db_v: object,
            noise_resynth_v: object,
            mix_v: object,
            delta_listen_v: object,
            denoise_v: object,
            dn_start_hz_v: object,
            dn_end_hz_v: object,
            dn_edge_hz_v: object,
            dn_floor_db_v: object,
            dn_psd_smooth_ms_v: object,
            dn_minwin_ms_v: object,
            dn_up_db_per_s_v: object,
            dn_attack_ms_v: object,
            dn_release_ms_v: object,
            dn_freq_smooth_bins_v: object,
            deres_v: object,
            deq_start_hz_v: object,
            deq_end_hz_v: object,
            deq_edge_hz_v: object,
            deq_freq_med_bins_v: object,
            deq_thr_db_v: object,
            deq_slope_v: object,
            deq_max_att_db_v: object,
            deq_density_lo_v: object,
            deq_density_hi_v: object,
            deq_persist_ms_v: object,
            deq_persist_thr_db_v: object,
            deq_freq_smooth_bins_v: object,
            deq_tonal_boost_db_v: object,
            deq_time_floor_v: object,
            deq_floor_smooth_ms_v: object,
            deq_floor_rise_db_per_s_v: object,
            expander_v: object,
            exp_start_hz_v: object,
            exp_end_hz_v: object,
            exp_threshold_db_v: object,
            exp_ratio_v: object,
            exp_attack_ms_v: object,
            exp_release_ms_v: object,
            hpss_v: object,
            hpss_start_hz_v: object,
            hpss_end_hz_v: object,
            hpss_time_frames_v: object,
            hpss_freq_bins_v: object,
            hpss_harmonic_only_v: object,
            phase_blur_v: object,
            pb_start_hz_v: object,
            pb_end_hz_v: object,
            pb_harmonic_only_v: object,
            hf_resynth_v: object,
            hf_lp_hz_v: object,
            hf_src_lo_hz_v: object,
            hf_src_hi_hz_v: object,
            hf_drive_v: object,
            hf_hp_hz_v: object,
            hf_mix_v: object,
            master_enabled_v: object,
            hp_hz_v: object,
            target_lufs_v: object,
            target_rms_dbfs_v: object,
            norm_max_gain_db_v: object,
            norm_max_atten_db_v: object,
            ceiling_dbtp_v: object,
            lim_lookahead_ms_v: object,
            lim_release_ms_v: object,
            tp_os_v: object,
        ) -> str:
            import json

            # Build a real Params + MasterParams using the existing builder, then dump asdict() (full key set).
            p, mp, _ = _build_params(
                start_hz=float(start_hz_v),
                end_hz=float(end_hz_v),
                edge_hz=float(edge_hz_v),
                n_fft=int(n_fft_v),
                hop=int(hop_v),
                flat_start=float(flat_start_v),
                flat_end=float(flat_end_v),
                freq_med_bins=int(freq_med_bins_v),
                thr_db=float(thr_db_v),
                slope=float(slope_v),
                density_lo=float(density_lo_v),
                density_hi=float(density_hi_v),
                flux_thr_db=float(flux_thr_db_v),
                flux_range_db=float(flux_range_db_v),
                noise_resynth=float(noise_resynth_v),
                mix=float(mix_v),
                delta_listen=bool(delta_listen_v),
                denoise=float(denoise_v),
                dn_start_hz=float(dn_start_hz_v),
                dn_end_hz=float(dn_end_hz_v),
                dn_edge_hz=float(dn_edge_hz_v),
                dn_floor_db=float(dn_floor_db_v),
                dn_psd_smooth_ms=float(dn_psd_smooth_ms_v),
                dn_minwin_ms=float(dn_minwin_ms_v),
                dn_up_db_per_s=float(dn_up_db_per_s_v),
                dn_attack_ms=float(dn_attack_ms_v),
                dn_release_ms=float(dn_release_ms_v),
                dn_freq_smooth_bins=int(dn_freq_smooth_bins_v),
                deres=float(deres_v),
                deq_start_hz=float(deq_start_hz_v),
                deq_end_hz=float(deq_end_hz_v),
                deq_edge_hz=float(deq_edge_hz_v),
                deq_freq_med_bins=int(deq_freq_med_bins_v),
                deq_thr_db=float(deq_thr_db_v),
                deq_slope=float(deq_slope_v),
                deq_max_att_db=float(deq_max_att_db_v),
                deq_density_lo=float(deq_density_lo_v),
                deq_density_hi=float(deq_density_hi_v),
                deq_persist_ms=float(deq_persist_ms_v),
                deq_persist_thr_db=float(deq_persist_thr_db_v),
                deq_freq_smooth_bins=int(deq_freq_smooth_bins_v),
                deq_tonal_boost_db=float(deq_tonal_boost_db_v),
                deq_time_floor=bool(deq_time_floor_v),
                deq_floor_smooth_ms=float(deq_floor_smooth_ms_v),
                deq_floor_rise_db_per_s=float(deq_floor_rise_db_per_s_v),
                expander=bool(expander_v),
                exp_start_hz=float(exp_start_hz_v),
                exp_end_hz=float(exp_end_hz_v),
                exp_threshold_db=float(exp_threshold_db_v),
                exp_ratio=float(exp_ratio_v),
                exp_attack_ms=float(exp_attack_ms_v),
                exp_release_ms=float(exp_release_ms_v),
                hpss=bool(hpss_v),
                hpss_start_hz=float(hpss_start_hz_v),
                hpss_end_hz=float(hpss_end_hz_v),
                hpss_time_frames=int(hpss_time_frames_v),
                hpss_freq_bins=int(hpss_freq_bins_v),
                hpss_harmonic_only=bool(hpss_harmonic_only_v),
                phase_blur=float(phase_blur_v),
                pb_start_hz=float(pb_start_hz_v),
                pb_end_hz=float(pb_end_hz_v),
                pb_harmonic_only=bool(pb_harmonic_only_v),
                hf_resynth=bool(hf_resynth_v),
                hf_lp_hz=float(hf_lp_hz_v),
                hf_src_lo_hz=float(hf_src_lo_hz_v),
                hf_src_hi_hz=float(hf_src_hi_hz_v),
                hf_drive=float(hf_drive_v),
                hf_hp_hz=float(hf_hp_hz_v),
                hf_mix=float(hf_mix_v),
                master_enabled=bool(master_enabled_v),
                hp_hz=float(hp_hz_v),
                target_lufs=float(target_lufs_v),
                target_rms_dbfs=float(target_rms_dbfs_v),
                norm_max_gain_db=float(norm_max_gain_db_v),
                norm_max_atten_db=float(norm_max_atten_db_v),
                ceiling_dbtp=float(ceiling_dbtp_v),
                lim_lookahead_ms=float(lim_lookahead_ms_v),
                lim_release_ms=float(lim_release_ms_v),
                tp_os=int(tp_os_v),
                spec_n_fft=2048,
                spec_hop=512,
                spec_max_frames=1200,
                spec_max_hz=20000.0,
            )
            # Realtime-ish player: prefer causal filters (avoid zero-phase filtfilt in chunked playback).
            try:
                p.hf_zero_phase = False
            except Exception:
                pass
            path = os.path.join(os.path.dirname(__file__), "realtime_params.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"params": asdict(p), "master_params": asdict(mp)}, f, indent=2, sort_keys=True)
            return (
                f"Wrote `{path}`. Start `python realtime_player.py YOURFILE.wav --params {os.path.basename(path)}`"
            )

        rt_export_inputs = preset_save_inputs[2:]  # same knob list, without name/desc
        rt_export.click(fn=export_realtime_params, inputs=rt_export_inputs, outputs=[rt_export_status])

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


