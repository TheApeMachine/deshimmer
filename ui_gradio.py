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

import io
import math
import os
import time
from dataclasses import asdict, fields, replace
from typing import Any, Mapping

import numpy as np
import soundfile as sf

import master
import auto_tune
from deshimmer_api import process_audio, slice_with_context


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


def _gradio_audio(x: np.ndarray, sr: int) -> tuple[int, np.ndarray]:
    """Return (sample_rate, float32 audio) for gr.Audio outputs."""
    x2 = _to_float_audio(x)
    if x2.shape[1] == 1:
        return int(sr), x2[:, 0]
    return int(sr), x2


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


def _png_bytes_to_rgb(png_bytes: bytes) -> np.ndarray:
    if not png_bytes:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    try:
        from PIL import Image
    except Exception:
        return np.zeros((1, 1, 3), dtype=np.uint8)
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
        "- Preview region is processed with temporal context padding (matches full-track stateful DSP more closely)",
    ]
    return "\n".join(lines)


PARAM_KNOB_NAMES: tuple[str, ...] = (
    "start_hz", "end_hz", "edge_hz", "n_fft", "hop",
    "flat_start", "flat_end", "freq_med_bins", "thr_db", "slope",
    "density_lo", "density_hi", "flux_thr_db", "flux_range_db",
    "noise_resynth", "mix", "delta_listen",
    "denoise", "dn_start_hz", "dn_end_hz", "dn_edge_hz", "dn_floor_db",
    "dn_psd_smooth_ms", "dn_minwin_ms", "dn_up_db_per_s", "dn_attack_ms",
    "dn_release_ms", "dn_freq_smooth_bins",
    "deres", "deq_start_hz", "deq_end_hz", "deq_edge_hz", "deq_freq_med_bins",
    "deq_thr_db", "deq_slope", "deq_max_att_db", "deq_density_lo",
    "deq_density_hi", "deq_persist_ms", "deq_persist_thr_db",
    "deq_freq_smooth_bins", "deq_tonal_boost_db", "deq_time_floor",
    "deq_floor_smooth_ms", "deq_floor_rise_db_per_s", "deq_floor_thr_db",
    "expander", "exp_start_hz", "exp_end_hz", "exp_threshold_db",
    "exp_ratio", "exp_attack_ms", "exp_release_ms",
    "hpss", "hpss_start_hz", "hpss_end_hz", "hpss_time_frames",
    "hpss_freq_bins", "hpss_harmonic_only", "hpss_protect_percussive",
    "magnitude_inpaint", "deq_inpaint", "total_att_cap_db", "nuclear_mode",
    "ms_process", "ms_side_scale",
    "phase_blur", "pb_start_hz", "pb_end_hz", "pb_harmonic_only",
    "swish_repair", "swish_start_hz", "swish_end_hz",
    "swish_time_amt", "swish_freq_amt",
    "hf_decorrelate", "hf_dec_start_hz", "hf_dec_end_hz",
    "hf_resynth", "hf_lp_hz", "hf_src_lo_hz", "hf_src_hi_hz",
    "hf_drive", "hf_hp_hz", "hf_mix", "hf_confidence_blend",
    "master_enabled", "hp_hz", "target_lufs", "target_rms_dbfs",
    "norm_max_gain_db", "norm_max_atten_db", "ceiling_dbtp",
    "lim_lookahead_ms", "lim_release_ms", "tp_os",
)
SPEC_PARAM_NAMES: tuple[str, ...] = ("spec_n_fft", "spec_hop", "spec_max_frames", "spec_max_hz")
PARAM_STATE_NAMES: tuple[str, ...] = PARAM_KNOB_NAMES + SPEC_PARAM_NAMES

MASTER_TO_UI_NAME = {
    "enabled": "master_enabled",
    "lookahead_ms": "lim_lookahead_ms",
    "release_ms": "lim_release_ms",
    "os_factor": "tp_os",
}
UI_TO_MASTER_NAME = {v: k for k, v in MASTER_TO_UI_NAME.items()}

UI_DEFAULT_OVERRIDES: dict[str, object] = {
    "spec_n_fft": 2048,
    "spec_hop": 512,
    "spec_max_frames": 1200,
    "spec_max_hz": 20000.0,
}


def _coerce_like(value: object, default: object) -> object:
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(round(float(value)))
    if isinstance(default, float) or default is None:
        return float(value)
    return value


def _default_param_state() -> dict[str, object]:
    p = master.Params()
    mp = master.MasterParams()
    state: dict[str, object] = {}
    p_values = asdict(p)
    mp_values = asdict(mp)
    for name in PARAM_KNOB_NAMES:
        if name in p_values:
            state[name] = p_values[name]
            continue
        master_name = UI_TO_MASTER_NAME.get(name, name)
        if master_name in mp_values:
            value = mp_values[master_name]
            if name == "target_lufs" and value is None:
                value = 999.0
            state[name] = value
    state.update(UI_DEFAULT_OVERRIDES)
    return state


def _merge_param_state(
    state: Mapping[str, object] | None,
    updates: Mapping[str, object],
) -> dict[str, object]:
    merged = _default_param_state()
    if isinstance(state, Mapping):
        for name in PARAM_STATE_NAMES:
            if name in state:
                merged[name] = state[name]
    for name, value in updates.items():
        if name in PARAM_STATE_NAMES:
            merged[name] = value
    return merged


def _state_from_dataclasses(
    p: master.Params,
    mp: master.MasterParams,
    dp: master.DebugParams | None = None,
) -> dict[str, object]:
    state = _default_param_state()
    p_values = asdict(p)
    mp_values = asdict(mp)
    for name in PARAM_KNOB_NAMES:
        if name in p_values:
            state[name] = p_values[name]
            continue
        master_name = UI_TO_MASTER_NAME.get(name, name)
        if master_name in mp_values:
            value = mp_values[master_name]
            if name == "target_lufs" and value is None:
                value = 999.0
            state[name] = value
    if dp is not None:
        dp_values = asdict(dp)
        for name in SPEC_PARAM_NAMES:
            field_name = name.removeprefix("spec_")
            if name in dp_values:
                state[name] = dp_values[name]
            elif field_name in dp_values:
                state[name] = dp_values[field_name]
    return state


def _state_to_component_values(state: Mapping[str, object] | None) -> list[object]:
    merged = _merge_param_state(state, {})
    return [merged[name] for name in PARAM_STATE_NAMES]


def _build_params(
    param_state: Mapping[str, object] | None,
) -> tuple[master.Params, master.MasterParams, master.DebugParams]:
    state = _merge_param_state(param_state, {})

    p = master.Params()
    p_updates: dict[str, object] = {}
    for f in fields(master.Params):
        if f.name not in state:
            continue
        p_updates[f.name] = _coerce_like(state[f.name], getattr(p, f.name))
    p_updates.update({"pad": True, "fade_ms": 5.0, "seed": 0})
    p = replace(p, **p_updates)

    mp = master.MasterParams()
    mp_updates: dict[str, object] = {}
    for f in fields(master.MasterParams):
        ui_name = MASTER_TO_UI_NAME.get(f.name, f.name)
        if ui_name not in state:
            continue
        if f.name == "target_lufs":
            raw_lufs = float(state[ui_name])
            mp_updates[f.name] = None if raw_lufs >= 998.0 else raw_lufs
        else:
            mp_updates[f.name] = _coerce_like(state[ui_name], getattr(mp, f.name))
    mp = replace(mp, **mp_updates)

    dp = master.DebugParams(
        enabled=False,
        spec_n_fft=int(state["spec_n_fft"]),
        spec_hop=int(state["spec_hop"]),
        spec_max_frames=int(state["spec_max_frames"]),
        spec_max_hz=float(state["spec_max_hz"]),
    )
    return p, mp, dp


def run_once(
    audio_in: tuple[int, np.ndarray] | None,
    preview_t0: float,
    preview_dur: float,
    loop_xfade_ms: float,
    full_song_mode: bool,
    param_state: Mapping[str, object] | None,
) -> tuple[
    tuple[int, np.ndarray],
    tuple[int, np.ndarray],
    tuple[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    dict[str, object],
]:
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    p, mp, dp = _build_params(param_state)

    if full_song_mode:
        y, info = process_audio(x, sr, params=p, master_params=mp, debug_params=dp)
        y2 = _to_float_audio(y)
        n = min(x.shape[0], y2.shape[0])
        x_seg = x[:n, :]
        if bool(p.delta_listen):
            removed = y2[:n, :]
            processed = _to_float_audio(x_seg - removed)
            out_sig = removed
            aux_sig = processed
        else:
            processed = y2[:n, :]
            removed = (x_seg - processed).astype(np.float32)
            out_sig = processed
            aux_sig = removed
        x_play = x_seg
        out_play = out_sig
        aux_play = aux_sig
        mode_note = (
            f"\n\n**Full song mode** — processed {n / sr:.1f}s. "
            "Spectrograms are downsampled; use *Render full & generate downloads* for WAV exports."
        )
    else:
        x_seg, s0, s1 = _slice_preview(x, sr, preview_t0, preview_dur)
        x_ctx, target_s0, target_s1, ctx_s0 = slice_with_context(
            x, sr, preview_t0, preview_dur, params=p,
        )
        y_ctx, info = process_audio(x_ctx, sr, params=p, master_params=mp, debug_params=dp)
        trim0 = target_s0 - ctx_s0
        trim1 = target_s1 - ctx_s0
        y = np.asarray(y_ctx)[trim0:trim1]
        y2 = _to_float_audio(y)

        if bool(p.delta_listen):
            removed = y2
            processed = _to_float_audio(x_seg[: removed.shape[0], :] - removed)
            out_sig = removed
            aux_sig = processed
        else:
            processed = y2
            removed = (x_seg[: processed.shape[0], :] - processed).astype(np.float32)
            out_sig = processed
            aux_sig = removed

        x_play = _loop_crossfade_rotate(x_seg, sr, loop_xfade_ms)
        out_play = _loop_crossfade_rotate(out_sig, sr, loop_xfade_ms)
        aux_play = _loop_crossfade_rotate(aux_sig, sr, loop_xfade_ms)
        mode_note = ""

    # Spectrograms
    in_png = _spectrogram_png_bytes(x_seg, sr, n_fft=dp.spec_n_fft, hop=dp.spec_hop, max_frames=dp.spec_max_frames, max_hz=dp.spec_max_hz)
    out_png = _spectrogram_png_bytes(out_sig, sr, n_fft=dp.spec_n_fft, hop=dp.spec_hop, max_frames=dp.spec_max_frames, max_hz=dp.spec_max_hz)
    diff_png = _spectrogram_png_bytes(aux_sig, sr, n_fft=dp.spec_n_fft, hop=dp.spec_hop, max_frames=dp.spec_max_frames, max_hz=dp.spec_max_hz)

    metrics_md = _render_metrics_md(info) + mode_note
    params_json: dict[str, object] = {"params": asdict(p), "master_params": asdict(mp)}

    return (
        _gradio_audio(x_play, sr),
        _gradio_audio(out_play, sr),
        _gradio_audio(aux_play, sr),
        _png_bytes_to_rgb(in_png),
        _png_bytes_to_rgb(out_png),
        _png_bytes_to_rgb(diff_png),
        metrics_md,
        params_json,
    )


def render_full_to_files(
    audio_in: tuple[int, np.ndarray] | None,
    param_state: Mapping[str, object] | None,
) -> tuple[str, str, str]:
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    p, mp, dp = _build_params(param_state)

    y, _ = process_audio(x, sr, params=p, master_params=mp, debug_params=dp)
    y2 = _to_float_audio(y)
    if bool(p.delta_listen):
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
                "deq_floor_thr_db": 3.0,
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
                    info="Process and play back the entire track in the players below (slower). Preview start/duration are ignored. For WAV exports, use Render full & generate downloads.",
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
                run_btn = gr.Button("Run preview / full song")

        with gr.Row():
            a_in = gr.Audio(label="Input", type="numpy", interactive=False)
            a_out = gr.Audio(label="Output", type="numpy", interactive=False, autoplay=True)
            a_diff = gr.Audio(label="Diff / removed", type="numpy", interactive=False)

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

        with gr.Accordion("✨ Auto-tune (Analyze + Refine)", open=True):
            gr.Markdown(
                "*Auto-derive sensible defaults from the audio (Analyze), then optionally refine with a "
                "Bayesian search across a few preview regions (Refine). Both write directly into the sliders below.*"
            )
            with gr.Row():
                auto_aggressiveness = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Aggressiveness",
                    info="0 = preserve content (more conservative), 1 = maximise artifact reduction. Refine only.",
                )
                auto_n_trials = gr.Slider(
                    4, 30, value=12, step=1,
                    label="Trials per stage",
                    info="Higher = better tuning but slower. ~12 is a good default. Refine only.",
                )
                auto_refine_dur = gr.Slider(
                    1.0, 6.0, value=3.0, step=0.25,
                    label="Refine region duration (s)",
                    info="Shorter regions = faster optimisation. Refine only.",
                )
            with gr.Row():
                auto_use_preview = gr.Checkbox(
                    value=True,
                    label="Use current preview as one region",
                    info="When on, the current preview start/duration is locked in as one of the auto regions.",
                )
                auto_n_regions = gr.Slider(
                    1, 8, value=5, step=1,
                    label="Number of regions",
                    info="How many short windows to sample (auto-picks quiet/loud/transient).",
                )
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze (set defaults from audio)", variant="secondary")
                refine_btn = gr.Button("⚙️ Refine (Bayesian optimise)", variant="primary")
            auto_report = gr.Markdown()

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
                deq_floor_thr_db = gr.Slider(
                    0.0, 12.0, value=3.0, step=0.1,
                    label="Floor excess threshold (dB)",
                    info="How many dB above the local frequency floor a stationary bin must sit before de-resonance acts."
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
                hpss_protect_percussive = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.05,
                    label="Protect percussive (0–1)",
                    info="Reduce repair depth on percussive bins. 0 = off, 1 = strong cymbal/snare protection."
                )

            gr.Markdown("**Repair philosophy** — inpainting, combined cap, Mid/Side")
            with gr.Row():
                magnitude_inpaint = gr.Checkbox(
                    value=True,
                    label="Magnitude inpainting (shimmer)",
                    info="Cap spikes to local median instead of pure attenuation (preserves air)."
                )
                deq_inpaint = gr.Checkbox(
                    value=True,
                    label="Magnitude inpainting (de-res)",
                    info="Same inpainting approach for persistent resonances."
                )
                ms_process = gr.Checkbox(
                    value=False,
                    label="Mid/Side processing",
                    info="Apply full repair on Mid; scale repair on Side to preserve stereo width."
                )
            with gr.Row():
                total_att_cap_db = gr.Slider(
                    3.0, 24.0, value=12.0, step=0.5,
                    label="Combined attenuation cap (dB)",
                    info="Max product attenuation per bin across denoise/deq/shimmer/expander."
                )
                ms_side_scale = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.05,
                    label="Side channel repair scale",
                    info="When Mid/Side is on, Side gets this fraction of repair depth."
                )
                nuclear_mode = gr.Checkbox(
                    value=False,
                    label="Nuclear mode (disable cap)",
                    info="Disables combined attenuation cap. Use with HF resynth only when desperate."
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

            gr.Markdown("**Swish Repair** — adaptive phase coherence for moving AI 'swish' (not EQ-able)")
            with gr.Row():
                swish_repair = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="Swish repair amount",
                    info="Smooths erratic inter-frame/inter-bin phase where instability is high. Start ~0.3–0.5 on tonal AI material.",
                )
                swish_time_amt = gr.Slider(
                    0.0, 1.0, value=0.55, step=0.01,
                    label="Inter-frame smoothing",
                    info="Time-axis phase coherence (targets moving swish).",
                )
                swish_freq_amt = gr.Slider(
                    0.0, 1.0, value=0.30, step=0.01,
                    label="Inter-bin smoothing",
                    info="Frequency-axis phase coherence.",
                )
            with gr.Row():
                swish_start_hz = gr.Number(
                    value=3500.0,
                    label="Swish band start (Hz)",
                    info="Lower edge for phase repair.",
                )
                swish_end_hz = gr.Number(
                    value=14000.0,
                    label="Swish band end (Hz)",
                    info="Upper edge for phase repair.",
                )
            with gr.Row():
                hf_decorrelate = gr.Slider(
                    0.0, 1.0, value=0.0, step=0.01,
                    label="HF decorrelate",
                    info="Break synthetic L/R phase lock in upper band. Try 0.15–0.35 on stereo AI exports.",
                )
                hf_dec_start_hz = gr.Number(
                    value=4500.0,
                    label="Decorrelate start (Hz)",
                )
                hf_dec_end_hz = gr.Number(
                    value=16000.0,
                    label="Decorrelate end (Hz)",
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
                hf_confidence_blend = gr.Checkbox(
                    value=True,
                    label="Confidence-masked blend",
                    info="Only resynthesize HF where artifact confidence is high."
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

        params_state = gr.State(_default_param_state())
        _component_scope = locals()
        param_components: dict[str, Any] = {
            name: _component_scope[name] for name in PARAM_STATE_NAMES
        }
        param_outputs = [param_components[name] for name in PARAM_STATE_NAMES]
        result_outputs = [a_in, a_out, a_diff, im_in, im_out, im_diff, metrics, params_json]
        render_inputs = [audio_in, preview_t0, preview_dur, loop_xfade_ms, full_song_mode, params_state]

        def _preset_desc_md(name: str) -> str:
            p = all_presets.get(name, {})
            desc = p.get("desc", "")
            return f"**{name}**  \n{desc}" if desc else f"**{name}**"

        def apply_preset(name: str, state: Mapping[str, object] | None) -> tuple[Any, ...]:
            preset_data = all_presets.get(name, {})
            values = preset_data.get("values", {})
            updates = values if isinstance(values, Mapping) else {}
            new_state = _merge_param_state(state, updates)
            return tuple([new_state] + _state_to_component_values(new_state) + [_preset_desc_md(name)])

        preset.change(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])
        preset_apply.click(
            fn=apply_preset,
            inputs=[preset, params_state],
            outputs=[params_state] + param_outputs + [preset_desc],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        def do_analyze(
            audio_in_v: tuple[int, np.ndarray] | None,
            preview_t0_v: float,
            preview_dur_v: float,
            use_preview_v: bool,
            n_regions_v: float,
            state: Mapping[str, object] | None,
        ) -> tuple[Any, ...]:
            current_state = _merge_param_state(state, {})
            if audio_in_v is None:
                report = "**Auto-analyze:** please load an audio file first."
                return tuple([current_state] + _state_to_component_values(current_state) + [report])

            sr_v, x_v = audio_in_v
            sr_v = int(sr_v)
            x_v = _to_float_audio(x_v)
            base_p, _base_mp, _base_dp = _build_params(current_state)

            locked = None
            if bool(use_preview_v):
                locked = auto_tune.Region(t0=float(preview_t0_v), dur=float(preview_dur_v), label="user")
            n_reg = int(max(1, round(float(n_regions_v))))
            regions = auto_tune.pick_regions(x_v, sr_v, n=n_reg, dur=max(2.0, float(preview_dur_v)), locked=locked)
            new_p, new_mp, report = auto_tune.analyze(x_v, sr_v, regions, base_params=base_p)
            new_state = _merge_param_state(current_state, _state_from_dataclasses(new_p, new_mp))
            return tuple([new_state] + _state_to_component_values(new_state) + [report.to_markdown()])

        def do_refine(
            audio_in_v: tuple[int, np.ndarray] | None,
            preview_t0_v: float,
            preview_dur_v: float,
            use_preview_v: bool,
            n_regions_v: float,
            aggressiveness_v: float,
            n_trials_v: float,
            refine_dur_v: float,
            state: Mapping[str, object] | None,
            progress: Any = gr.Progress(track_tqdm=False),
        ) -> tuple[Any, ...]:
            current_state = _merge_param_state(state, {})
            if audio_in_v is None:
                report = "**Auto-refine:** please load an audio file first."
                return tuple([current_state] + _state_to_component_values(current_state) + [report])

            sr_v, x_v = audio_in_v
            sr_v = int(sr_v)
            x_v = _to_float_audio(x_v)
            base_p, base_mp, _base_dp = _build_params(current_state)

            locked = None
            if bool(use_preview_v):
                locked = auto_tune.Region(t0=float(preview_t0_v), dur=float(preview_dur_v), label="user")
            n_reg = int(max(1, round(float(n_regions_v))))
            refine_dur = float(max(1.0, min(refine_dur_v, preview_dur_v + 1.0)))
            region_dur = float(max(refine_dur + 0.5, preview_dur_v, 4.0))
            regions = auto_tune.pick_regions(x_v, sr_v, n=n_reg, dur=region_dur, locked=locked)

            def _cb(frac: float, msg: str) -> None:
                try:
                    progress(float(frac), desc=msg)
                except Exception:
                    pass

            progress(0.0, desc="Refining...")
            new_p, summary = auto_tune.refine(
                x_v,
                sr_v,
                base_params=base_p,
                regions=regions,
                aggressiveness=float(aggressiveness_v),
                n_trials_per_stage=int(n_trials_v),
                refine_dur=refine_dur,
                progress_cb=_cb,
            )

            stages_md_lines = ["**Auto-refine results**", ""]
            stages_md_lines.append(
                f"- Aggressiveness: **{float(aggressiveness_v):.2f}**, "
                f"trials/stage: **{int(n_trials_v)}**, regions: {len(regions)}"
            )
            stages_md_lines.append(f"- Final composite score: **{summary.get('final_score', 0.0):+.3f}**")
            for s in summary.get("stages", []):
                stages_md_lines.append(f"- Stage **{s['stage']}**: selected={s.get('selection_mode', 'balanced')}, score={s.get('selected_score', 0.0):+.3f}")
                pareto = s.get("pareto", {})
                if pareto:
                    for mode in ("safe", "balanced", "aggressive"):
                        pt = pareto.get(mode, {})
                        vals = pt.get("values") or []
                        if vals:
                            stages_md_lines.append(
                                f"  - **{mode}**: artifact={vals[0]:+.2f}, "
                                f"music={-vals[1]:+.2f}, stereo={-vals[3]:+.2f}"
                            )
                bp = s.get("selected_params", s.get("best_params", {}))
                if bp:
                    bp_s = ", ".join(f"`{k}`={v:.3g}" if isinstance(v, float) else f"`{k}`={v}" for k, v in bp.items())
                    stages_md_lines.append(f"  - {bp_s}")
            failure_log = summary.get("failure_log")
            if failure_log:
                stages_md_lines.append(f"- Failed-trial diagnostics: `{failure_log}`")
            stages_md_lines.append("")
            stages_md_lines.append("_(NSGA-II runs with flat objective weights; aggressiveness selects from the Pareto front after search.)_")

            new_state = _merge_param_state(current_state, _state_from_dataclasses(new_p, base_mp))
            return tuple([new_state] + _state_to_component_values(new_state) + ["\n".join(stages_md_lines)])

        analyze_btn.click(
            fn=do_analyze,
            inputs=[audio_in, preview_t0, preview_dur, auto_use_preview, auto_n_regions, params_state],
            outputs=[params_state] + param_outputs + [auto_report],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        refine_btn.click(
            fn=do_refine,
            inputs=[
                audio_in,
                preview_t0,
                preview_dur,
                auto_use_preview,
                auto_n_regions,
                auto_aggressiveness,
                auto_n_trials,
                auto_refine_dur,
                params_state,
            ],
            outputs=[params_state] + param_outputs + [auto_report],
        ).then(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        demo.load(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])
        run_btn.click(fn=run_once, inputs=render_inputs, outputs=result_outputs)
        full_btn.click(fn=render_full_to_files, inputs=[audio_in, params_state], outputs=[dl_out, dl_diff, dl_params])

        def _bind_preview_auto(comp: Any) -> None:
            handler = getattr(comp, "release", None)
            if callable(handler):
                handler(fn=run_once, inputs=render_inputs, outputs=result_outputs)
                return
            comp.change(fn=run_once, inputs=render_inputs, outputs=result_outputs)

        def _make_state_updater(key: str):
            def _update(value: object, state: Mapping[str, object] | None) -> dict[str, object]:
                return _merge_param_state(state, {key: value})
            return _update

        def _bind_param_auto(key: str, comp: Any) -> None:
            updater = _make_state_updater(key)
            handler = getattr(comp, "release", None)
            if callable(handler):
                handler(fn=updater, inputs=[comp, params_state], outputs=[params_state]).then(
                    fn=run_once, inputs=render_inputs, outputs=result_outputs
                )
                return
            comp.change(fn=updater, inputs=[comp, params_state], outputs=[params_state]).then(
                fn=run_once, inputs=render_inputs, outputs=result_outputs
            )

        for comp in (preview_t0, preview_dur, loop_xfade_ms, full_song_mode):
            _bind_preview_auto(comp)
        for key, comp in param_components.items():
            _bind_param_auto(key, comp)

        def save_preset_from_state(
            preset_name_v: str,
            desc: str,
            state: Mapping[str, object] | None,
        ) -> tuple[Any, str]:
            nonlocal all_presets
            clean_name = (preset_name_v or "").strip()
            if not clean_name:
                return gr.update(), "Please enter a preset name."
            current_state = _merge_param_state(state, {})
            values = {key: current_state[key] for key in PARAM_KNOB_NAMES}
            _save_user_preset(USER_PRESETS_PATH, clean_name, (desc or "").strip(), values)
            all_presets = dict(base_presets)
            all_presets.update(_load_user_presets(USER_PRESETS_PATH))
            return gr.update(choices=list(all_presets.keys()), value=clean_name), f"Saved preset to `{USER_PRESETS_PATH}`."

        preset_save.click(
            fn=save_preset_from_state,
            inputs=[preset_name, preset_user_desc, params_state],
            outputs=[preset, preset_save_status],
        ).then(fn=_preset_desc_md, inputs=[preset], outputs=[preset_desc])

        def export_realtime_params(state: Mapping[str, object] | None) -> str:
            import json

            p, mp, _ = _build_params(state)
            p.hf_zero_phase = False
            path = os.path.join(os.path.dirname(__file__), "realtime_params.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"params": asdict(p), "master_params": asdict(mp)}, f, indent=2, sort_keys=True)
            return f"Wrote `{path}`. Start `python realtime_player.py YOURFILE.wav --params {os.path.basename(path)}`"

        rt_export.click(fn=export_realtime_params, inputs=[params_state], outputs=[rt_export_status])

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
