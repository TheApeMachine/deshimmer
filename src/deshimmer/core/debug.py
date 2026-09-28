"""Debugging, diagnostics collection, and spectrogram plot rendering."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .params import DebugParams


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _compute_mag_spectrogram_db(
    x: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    max_frames: int,
    max_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    n_fft = max(256, n_fft)
    hop = max(1, hop)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_mask = freqs <= float(min(0.5 * sr, max_hz))
    freqs = freqs[f_mask]

    n = x.shape[0]
    est_frames = int(math.ceil(n / hop))
    if est_frames > max_frames:
        hop = int(math.ceil(n / max_frames))
        hop = max(1, hop)

    win = np.hanning(n_fft).astype(np.float32)
    eps = 1e-12

    frames = []
    times = []
    for s in range(0, n, hop):
        frame = x[s:s + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]), mode="constant")
        frame = frame * win
        spec = np.fft.rfft(frame, n=n_fft)
        mag = np.abs(spec).astype(np.float32) + eps
        mag = mag[f_mask]
        frames.append(20.0 * np.log10(mag))
        times.append((s + 0.5 * n_fft) / sr)

    S_db = np.stack(frames, axis=1) if frames else np.zeros((freqs.size, 0), dtype=np.float32)
    t = np.array(times, dtype=np.float32)
    return S_db.astype(np.float32), freqs.astype(np.float32), t


def _plot_and_save_png(fig_path: str, fig) -> None:
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


class DebugCollector:
    def __init__(
        self,
        sr: int,
        freqs: np.ndarray,
        dn_idx: np.ndarray,
        deq_idx: np.ndarray,
        sh_idx: np.ndarray,
        stride: int,
        pad_offset: int,
        duration_s: float,
    ):
        self.sr = sr
        self.freqs = freqs
        self.dn_idx = dn_idx
        self.deq_idx = deq_idx
        self.sh_idx = sh_idx
        self.stride = max(1, stride)
        self.pad_offset = pad_offset
        self.duration_s = float(duration_s)

        self.frame_i = 0

        self.t: List[float] = []
        self.w_noise: List[float] = []
        self.w_trans: List[float] = []
        self.flux_db: List[float] = []
        self.band_db: List[float] = []

        self.dn_depth: List[float] = []
        self.deq_depth: List[float] = []
        self.sh_depth: List[float] = []

        self.att_dn_db: List[np.ndarray] = []
        self.att_deq_db: List[np.ndarray] = []
        self.att_sh_db: List[np.ndarray] = []

        self.noise_psd_dn_db: List[np.ndarray] = []

    def want(self) -> bool:
        return (self.frame_i % self.stride) == 0

    def push_meta(
        self,
        s: int,
        n_fft: int,
        w_noise: float,
        w_trans: float,
        flux_db: float,
        band_db: float,
        dn_depth: float,
        deq_depth: float,
        sh_depth: float,
    ):
        t_center = (float(s) + 0.5 * float(n_fft) - float(self.pad_offset)) / float(self.sr)
        if t_center < 0.0:
            t_center = 0.0
        if t_center > self.duration_s:
            t_center = self.duration_s
        self.t.append(float(t_center))
        self.w_noise.append(float(w_noise))
        self.w_trans.append(float(w_trans))
        self.flux_db.append(float(flux_db))
        self.band_db.append(float(band_db))
        self.dn_depth.append(float(dn_depth))
        self.deq_depth.append(float(deq_depth))
        self.sh_depth.append(float(sh_depth))

    def push_att(
        self,
        att_dn_db: Optional[np.ndarray],
        att_deq_db: Optional[np.ndarray],
        att_sh_db: Optional[np.ndarray],
    ):
        if att_dn_db is not None:
            self.att_dn_db.append(att_dn_db.astype(np.float32, copy=False))
        if att_deq_db is not None:
            self.att_deq_db.append(att_deq_db.astype(np.float32, copy=False))
        if att_sh_db is not None:
            self.att_sh_db.append(att_sh_db.astype(np.float32, copy=False))

    def push_noise_psd_dn(self, noise_psd_dn_db: Optional[np.ndarray]):
        if noise_psd_dn_db is not None:
            self.noise_psd_dn_db.append(noise_psd_dn_db.astype(np.float32, copy=False))

    def step(self):
        self.frame_i += 1

    def finalize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["t"] = np.array(self.t, dtype=np.float32)
        out["w_noise"] = np.array(self.w_noise, dtype=np.float32)
        out["w_trans"] = np.array(self.w_trans, dtype=np.float32)
        out["flux_db"] = np.array(self.flux_db, dtype=np.float32)
        out["band_db"] = np.array(self.band_db, dtype=np.float32)
        out["dn_depth"] = np.array(self.dn_depth, dtype=np.float32)
        out["deq_depth"] = np.array(self.deq_depth, dtype=np.float32)
        out["sh_depth"] = np.array(self.sh_depth, dtype=np.float32)

        if len(self.att_dn_db) > 0:
            out["f_dn"] = self.freqs[self.dn_idx].astype(np.float32)
            out["att_dn_db"] = np.stack(self.att_dn_db, axis=1)
        if len(self.att_deq_db) > 0:
            out["f_deq"] = self.freqs[self.deq_idx].astype(np.float32)
            out["att_deq_db"] = np.stack(self.att_deq_db, axis=1)
        if len(self.att_sh_db) > 0:
            out["f_sh"] = self.freqs[self.sh_idx].astype(np.float32)
            out["att_sh_db"] = np.stack(self.att_sh_db, axis=1)
        if len(self.noise_psd_dn_db) > 0:
            out["noise_psd_dn_db"] = np.stack(self.noise_psd_dn_db, axis=1)

        return out


def render_debug(
    dbg_dir: str,
    dbg_data: Dict[str, Any],
    x_in: np.ndarray,
    y_repaired: np.ndarray,
    y_out: np.ndarray,
    sr: int,
    dp: DebugParams,
    summary: Dict[str, Any],
) -> None:
    dbg_dir = _ensure_dir(dbg_dir)

    if dp.save_npz:
        npz_path = os.path.join(dbg_dir, "debug_data.npz")
        arrays = dict((k, v) for k, v in dbg_data.items() if isinstance(v, np.ndarray))
        np.savez_compressed(npz_path, **arrays)

    _write_json(os.path.join(dbg_dir, "summary.json"), summary)

    if not dp.save_png:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        _write_json(os.path.join(dbg_dir, "plot_error.json"), {"error": str(e)})
        return

    S0, f0, t0 = _compute_mag_spectrogram_db(x_in, sr, dp.spec_n_fft, dp.spec_hop, dp.spec_max_frames, dp.spec_max_hz)
    S1, f1, t1 = _compute_mag_spectrogram_db(y_out, sr, dp.spec_n_fft, dp.spec_hop, dp.spec_max_frames, dp.spec_max_hz)

    T = min(S0.shape[1], S1.shape[1])
    S0 = S0[:, :T]
    S1 = S1[:, :T]
    t = t0[:T] if t0.size else t1[:T]
    f = f0 if f0.size else f1

    # input spectrogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ext: tuple[float, float, float, float] = (
        float(t[0]) if t.size else 0.0,
        float(t[-1]) if t.size else 0.0,
        float(f[0]) if f.size else 0.0,
        float(f[-1]) if f.size else 0.0,
    )
    im = ax.imshow(S0, origin="lower", aspect="auto", extent=ext)
    ax.set_title("Spectrogram (input) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_input.png"), fig)

    # output spectrogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(S1, origin="lower", aspect="auto", extent=ext)
    ax.set_title("Spectrogram (output) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_output.png"), fig)

    # diff spectrogram
    Sd = (S1 - S0).astype(np.float32)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Sd, origin="lower", aspect="auto", extent=ext)
    ax.set_title("Spectrogram (output - input) [dB]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Δ dB")
    _plot_and_save_png(os.path.join(dbg_dir, "spectrogram_diff.png"), fig)

    # Attenuation maps
    t_frames = dbg_data.get("t", None)

    def plot_att_map(name: str, f_key: str, a_key: str):
        if f_key not in dbg_data or a_key not in dbg_data or t_frames is None:
            return
        F = dbg_data[f_key]
        A = dbg_data[a_key]
        if A.size == 0:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ext_att: tuple[float, float, float, float] = (
            float(t_frames[0]), float(t_frames[-1]), float(F[0]), float(F[-1]),
        )
        im = ax.imshow(A, origin="lower", aspect="auto", extent=ext_att)
        ax.set_title(f"Attenuation map: {name} [dB]")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Attenuation (dB)")
        _plot_and_save_png(os.path.join(dbg_dir, f"atten_{name}.png"), fig)

    plot_att_map("denoise", "f_dn", "att_dn_db")
    plot_att_map("deres", "f_deq", "att_deq_db")
    plot_att_map("shimmer", "f_sh", "att_sh_db")

    # Time-series metrics
    if t_frames is not None and t_frames.size > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t_frames, dbg_data.get("w_noise", np.zeros_like(t_frames)), label="noise-likeness (0..1)")
        ax.plot(t_frames, dbg_data.get("w_trans", np.zeros_like(t_frames)), label="transient gate (0..1)")
        ax.plot(t_frames, dbg_data.get("dn_depth", np.zeros_like(t_frames)), label="denoise depth")
        ax.plot(t_frames, dbg_data.get("deq_depth", np.zeros_like(t_frames)), label="deres depth")
        ax.plot(t_frames, dbg_data.get("sh_depth", np.zeros_like(t_frames)), label="shimmer depth")
        ax.set_title("Frame metrics / gates")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=8)
        _plot_and_save_png(os.path.join(dbg_dir, "frame_metrics.png"), fig)

        # Mean attenuation vs time
        fig = plt.figure()
        ax = fig.add_subplot(111)

        def mean_att(a_key: str) -> Optional[np.ndarray]:
            if a_key not in dbg_data:
                return None
            A = dbg_data[a_key]
            if A.size == 0:
                return None
            return np.mean(A, axis=0)

        m_dn = mean_att("att_dn_db")
        m_deq = mean_att("att_deq_db")
        m_sh = mean_att("att_sh_db")
        if m_dn is not None:
            ax.plot(t_frames, m_dn, label="mean denoise att (dB)")
        if m_deq is not None:
            ax.plot(t_frames, m_deq, label="mean deres att (dB)")
        if m_sh is not None:
            ax.plot(t_frames, m_sh, label="mean shimmer att (dB)")
        ax.set_title("Mean attenuation over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dB")
        ax.legend(loc="upper right", fontsize=8)
        _plot_and_save_png(os.path.join(dbg_dir, "mean_attenuation.png"), fig)

    # Noise PSD estimate
    if "noise_psd_dn_db" in dbg_data and t_frames is not None:
        N = dbg_data["noise_psd_dn_db"]
        F = dbg_data.get("f_dn", None)
        if F is not None and N.size:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t_frames, np.median(N, axis=0))
            ax.set_title("Estimated noise floor (median over denoise band) [dB]")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("dB (power)")
            _plot_and_save_png(os.path.join(dbg_dir, "noise_floor_estimate.png"), fig)

    with open(os.path.join(dbg_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Debug outputs:\n"
            "- summary.json: key measurements + settings\n"
            "- debug_data.npz: internal arrays (attenuation maps, gates)\n"
            "- spectrogram_*.png: before/after/diff spectrograms\n"
            "- atten_*.png: per-stage attenuation maps\n"
            "- frame_metrics.png: gates/depth over time\n"
            "- mean_attenuation.png: average attenuation over time\n"
        )
