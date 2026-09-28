"""Execution handlers for interactive preview and full export in UI."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
import sys
import time
from typing import Mapping

import numpy as np
import soundfile as sf

from ..api import process_audio, slice_with_context
from .audio import (
    _gradio_audio,
    _loop_crossfade_rotate,
    _png_bytes_to_rgb,
    _render_metrics_md,
    _slice_preview,
    _spectrogram_png_bytes,
    _to_float_audio,
)
from .state import _build_params


def run_once(
    audio_in: tuple[int, np.ndarray] | None,
    preview_t0: float,
    preview_dur: float,
    loop_xfade_ms: float,
    full_song_mode: bool,
    param_state: Mapping[str, object] | None,
    repair_only_diff: bool = False,
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
    """Execute a single preview or full song run and produce audio, spectrograms, and metrics."""
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)
    p, mp, dp = _build_params(param_state)
    x_ctx = x
    trim0, trim1 = 0, x.shape[0]
    x_seg = x
    mode_note = ""

    if not full_song_mode:
        x_seg, _, _ = _slice_preview(x, sr, preview_t0, preview_dur)
        x_ctx, target_s0, target_s1, ctx_s0 = slice_with_context(
            x, sr, preview_t0, preview_dur, params=p,
        )
        trim0, trim1 = target_s0 - ctx_s0, target_s1 - ctx_s0

    y_ctx, info = process_audio(
        x_ctx, sr, params=replace(p, delta_listen=False), master_params=mp,
        debug_params=dp, include_residuals=repair_only_diff,
    )
    processed = _to_float_audio(y_ctx)[trim0:trim1]
    removed = x_seg - processed

    if repair_only_diff:
        removed = _to_float_audio(info["residuals"]["artifact_repair_diff"])[trim0:trim1]
        mode_note += ("\n\n**Repair-only diff** — input minus a separate repair render "
                      "without L/R alignment, spectral balancing, spectral carving, or mastering.")

    out_sig = removed if p.delta_listen else processed
    aux_sig = processed if p.delta_listen else removed
    x_play, out_play, aux_play = x_seg, out_sig, aux_sig

    if not full_song_mode:
        x_play = _loop_crossfade_rotate(x_seg, sr, loop_xfade_ms)
        out_play = _loop_crossfade_rotate(out_sig, sr, loop_xfade_ms)
        aux_play = _loop_crossfade_rotate(aux_sig, sr, loop_xfade_ms)

    if full_song_mode:
        mode_note += (
            f"\n\n**Full song mode** — processed {x_seg.shape[0] / sr:.1f}s. "
            "Spectrograms are downsampled; use *Render full & generate downloads* for WAV exports."
        )

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
    repair_only_diff: bool = False,
) -> tuple[str, str, str]:
    """Render full audio track to 24-bit PCM WAV and JSON parameters on disk."""
    if audio_in is None:
        raise ValueError("Please load an audio file first.")

    sr, x = audio_in
    sr = int(sr)
    x = _to_float_audio(x)

    p, mp, dp = _build_params(param_state)

    y, info = process_audio(
        x, sr, params=replace(p, delta_listen=False), master_params=mp,
        debug_params=dp, include_residuals=repair_only_diff,
    )
    processed = _to_float_audio(y)
    removed = x - processed

    if repair_only_diff:
        removed = _to_float_audio(info["residuals"]["artifact_repair_diff"])

    out_audio = removed if p.delta_listen else processed
    diff_audio = processed if p.delta_listen else removed

    # Support testing patch on ui_gradio.__file__
    mod = sys.modules.get("ui_gradio")
    module_file = getattr(mod, "__file__", __file__) if mod is not None else __file__
    outdir = os.path.join(os.path.dirname(module_file), "ui_downloads")
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(outdir, f"output-{stamp}.wav")
    diff_path = os.path.join(outdir, f"diff-{stamp}.wav")
    params_path = os.path.join(outdir, f"params-{stamp}.json")

    sf.write(out_path, out_audio, sr, subtype="PCM_24")
    sf.write(diff_path, diff_audio, sr, subtype="PCM_24")

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({"params": asdict(p), "master_params": asdict(mp), "repair_only_diff": repair_only_diff}, f, indent=2, sort_keys=True)

    return out_path, diff_path, params_path
