"""Audio conversion, preview slicing, looping crossfade, and spectrogram rendering for UI."""

from __future__ import annotations

import io
import math
from typing import Any, Tuple

import numpy as np

from ..core import _compute_mag_spectrogram_db


def _to_float_audio(x: np.ndarray) -> np.ndarray:
    """Normalize input audio to 2D float32 in [-1.0, 1.0]."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("audio must be 1D or 2D")
    if x.dtype.kind in ("i", "u"):
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / max(1.0, maxv)
    else:
        x = x.astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0)


def _slice_preview(x: np.ndarray, sr: int, t0: float, dur: float) -> tuple[np.ndarray, int, int]:
    """Slice audio segment for preview playback."""
    n = x.shape[0]
    t0 = float(max(0.0, t0))
    dur = float(max(0.05, dur))
    s0 = int(round(t0 * sr))
    s1 = int(round((t0 + dur) * sr))
    s0 = max(0, min(n, s0))
    s1 = max(s0, min(n, s1))
    return x[s0:s1, :], s0, s1


def _spectrogram_png_bytes(x: np.ndarray, sr: int, n_fft: int, hop: int, max_frames: int, max_hz: float) -> bytes:
    """Compute spectrogram and render in-memory PNG bytes."""
    S, f, t = _compute_mag_spectrogram_db(x, sr, n_fft=n_fft, hop=hop, max_frames=max_frames, max_hz=max_hz)
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
    """Return (sample_rate, float32 audio) formatted for gr.Audio outputs."""
    x2 = _to_float_audio(x)
    if x2.shape[1] == 1:
        return int(sr), x2[:, 0]
    return int(sr), x2


def _loop_crossfade_rotate(x: np.ndarray, sr: int, crossfade_ms: float) -> np.ndarray:
    """
    Make a loop-friendly preview by rotating the segment so the wrap point is
    between consecutive samples from the original audio, and embedding a
    crossfade between tail->head at the start.
    """
    x2 = _to_float_audio(x)
    n = x2.shape[0]
    if n < 8:
        return x2

    ms = float(max(0.0, crossfade_ms))
    if ms <= 0.0:
        return x2

    f = int(round((ms / 1000.0) * float(sr)))
    f = int(max(0, min(f, n // 4)))
    if f < 8:
        return x2

    tail = x2[n - f : n, :]
    head = x2[0:f, :]
    w = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
    cf = (1.0 - w) * tail + w * head
    mid = x2[f : n - f, :]
    y = np.concatenate([cf, mid], axis=0)
    return y.astype(np.float32, copy=False)


def _png_bytes_to_rgb(png_bytes: bytes) -> np.ndarray:
    """Decode raw PNG bytes to uint8 RGB array."""
    if not png_bytes:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    try:
        from PIL import Image
    except Exception:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return np.asarray(im)


def _render_metrics_md(info: dict[str, object]) -> str:
    """Format DSP measurements, true-peak, loudness, and tonal metrics into Markdown."""
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
        "- Default diff = input - final output, including engineering and mastering; repair-only diff uses a separate render.",
        "- Preview region is processed with temporal context padding (matches full-track stateful DSP more closely)",
    ]
    tonal = info.get("tonal", {})
    before = tonal.get("before", {})
    after = tonal.get("after", {})
    lines.extend([
        "", "### Tonal self-consistency",
        f"- Status: {before.get('status', tonal.get('status', 'unavailable'))}; corrected partials: {tonal.get('changed_partials', 0)}",
        f"- Supported families: {before.get('families', 0)} → {after.get('families', 0)}",
        f"- Ambiguous partials excluded from correction: {before.get('ambiguous_partials', 0)}",
        f"- 95th-percentile internal disagreement: {fmt(before.get('p95_disagreement_cents'))} → {fmt(after.get('p95_disagreement_cents'))} cents",
        "- Measurements describe the tonal repair stage before wet/dry blending and delivery mastering.",
    ])
    return "\n".join(lines)
