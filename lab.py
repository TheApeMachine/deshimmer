#!/usr/bin/env python3
"""
shimmer_lab.py

Analysis tool for diagnosing AI/codec-ish shimmer/swish artifacts.

Outputs:
- spectrogram_full.png (coarse overview + band overlay)
- metrics.png (time-series: mid level, band level, flatness, modulation, spike score, composite score)
- modulation_spectrum.png (FFT of band envelope)
- metrics.csv (per-frame metrics)
- prints top "worst" time regions (for easy looping in DAW)

Why these metrics?
- Spectral flatness (geo/arith mean ratio) distinguishes noise-like vs tonal content. :contentReference[oaicite:5]{index=5}
- “Birdies” are time-varying narrow spectral artifacts in audio coding. :contentReference[oaicite:6]{index=6}
- Modulation spectrum = FFT of the envelope; shows amplitude modulation frequencies. :contentReference[oaicite:7]{index=7}

This script is designed to be memory-safe on long tracks:
- it computes metrics frame-by-frame
- it stores only a decimated spectrogram for the full-track view
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    # Band of interest
    start_hz: float
    end_hz: float

    # STFT-ish framing
    n_fft: int = 4096
    hop: int = 1024

    # Spectrogram plotting
    spec_stride: int = 4
    max_freq_plot: float = 12000.0

    # For “modulation residual” style metric (IIR smoothing in log-mag)
    smooth_ms: float = 80.0

    # For reporting “worst segments”
    top_k: int = 12
    cluster_gap_sec: float = 0.75
    cluster_len_sec: float = 4.0

    # Envelope modulation spectrum
    max_mod_hz: float = 200.0
    bandpass_order: int = 4

    # Loudness proxy band for “tail / transient” context
    mid_lo: float = 200.0
    mid_hi: float = 1500.0

    # Composite score weights (you can tweak later)
    w_flat: float = 1.0
    w_mod: float = 1.0
    w_spike: float = 0.6


def cents_to_band(center_hz: float, width_cents: float) -> Tuple[float, float]:
    half = width_cents * 0.5
    ratio = 2.0 ** (half / 1200.0)
    return center_hz / ratio, center_hz * ratio


def butter_bandpass(x: np.ndarray, sr: int, lo: float, hi: float, order: int = 4) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    nyq = sr / 2.0
    lo = max(1.0, float(lo))
    hi = min(float(hi), 0.99 * nyq)
    if hi <= lo:
        raise ValueError("Bandpass hi must be > lo")
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    y = signal.sosfiltfilt(sos, x, axis=0)
    return y.astype(np.float32)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_file(x: np.ndarray, sr: int, cfg: Cfg, outdir: str) -> None:
    ensure_dir(outdir)

    if x.ndim == 1:
        x = x[:, None]
    x = x.astype(np.float32, copy=False)
    n_samples, n_ch = x.shape

    n_fft = int(cfg.n_fft)
    hop = int(cfg.hop)
    if hop <= 0 or n_fft <= 0 or hop > n_fft:
        raise ValueError("Invalid n_fft/hop")

    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyq = float(freqs[-1])

    start_hz = float(cfg.start_hz)
    end_hz = float(cfg.end_hz)
    end_hz = min(end_hz, nyq)

    band_idx = np.where((freqs >= start_hz) & (freqs <= end_hz))[0]
    if band_idx.size < 8:
        raise ValueError("Band too narrow / not enough FFT bins; try bigger n_fft or wider band.")

    mid_hi = min(float(cfg.mid_hi), start_hz - 200.0)
    if mid_hi < cfg.mid_lo + 50.0:
        mid_hi = min(float(cfg.mid_hi), nyq)
    mid_idx = np.where((freqs >= cfg.mid_lo) & (freqs <= mid_hi))[0]
    if mid_idx.size < 8:
        mid_idx = np.where((freqs >= 200.0) & (freqs <= min(5000.0, nyq)))[0]

    # Decimated spectrogram storage (for overview plot)
    max_f = min(float(cfg.max_freq_plot), nyq)
    plot_idx = np.where(freqs <= max_f)[0]

    spec_db_cols: List[np.ndarray] = []
    spec_t: List[float] = []

    # Per-frame metrics
    times = []
    mid_db = []
    band_db = []
    flatness = []
    mod = []
    flux = []
    spike_db = []

    eps = 1e-12

    # Log-mag smoothing coefficient
    tau = max(1e-4, float(cfg.smooth_ms) / 1000.0)
    a = float(np.exp(-hop / (sr * tau)))

    Ls = None
    prev_L = None

    starts = range(0, n_samples, hop)
    for fi, s in enumerate(starts):
        frame = np.zeros((n_fft, n_ch), dtype=np.float32)
        chunk = x[s:s + n_fft, :]
        frame[:chunk.shape[0], :] = chunk
        frame_w = frame * win[:, None]

        spec = np.fft.rfft(frame_w, n=n_fft, axis=0)  # (freq, ch)
        mag = np.mean(np.abs(spec), axis=1).astype(np.float32) + eps

        # Mid loudness proxy (power)
        p_mid = float(np.mean(mag[mid_idx] ** 2))
        mid_db.append(10.0 * math.log10(max(p_mid, eps)))

        # Band level (power)
        p_band = float(np.mean(mag[band_idx] ** 2))
        band_db.append(10.0 * math.log10(max(p_band, eps)))

        # Spectral flatness (power): geo/arith mean. :contentReference[oaicite:8]{index=8}
        P = (mag[band_idx] ** 2).astype(np.float32)
        geo = float(np.exp(np.mean(np.log(P + eps))))
        ar = float(np.mean(P))
        flat = geo / (ar + eps)
        flatness.append(flat)

        # Log-magnitude modulation metric
        L = np.log(mag[band_idx]).astype(np.float32)
        if Ls is None:
            Ls = L.copy()
        else:
            Ls = a * Ls + (1.0 - a) * L
        R = L - Ls
        mod.append(float(np.mean(np.abs(R))))

        if prev_L is None:
            flux.append(0.0)
        else:
            flux.append(float(np.mean(np.abs(L - prev_L))))
        prev_L = L

        # Spike metric (birdies-ish): max vs median within band, in dB.
        # This catches narrow peaks that tower over their neighborhood. :contentReference[oaicite:9]{index=9}
        med = float(np.median(mag[band_idx]))
        mx = float(np.max(mag[band_idx]))
        ratio = mx / (med + eps)
        spike_db.append(20.0 * math.log10(max(ratio, 1.0)))

        t = s / sr
        times.append(t)

        # Coarse spectrogram capture
        if fi % int(cfg.spec_stride) == 0:
            col = 20.0 * np.log10(mag[plot_idx])
            spec_db_cols.append(col.astype(np.float32))
            spec_t.append(t)

    times = np.asarray(times, dtype=np.float32)
    mid_db = np.asarray(mid_db, dtype=np.float32)
    band_db = np.asarray(band_db, dtype=np.float32)
    flatness = np.asarray(flatness, dtype=np.float32)
    mod = np.asarray(mod, dtype=np.float32)
    flux = np.asarray(flux, dtype=np.float32)
    spike_db = np.asarray(spike_db, dtype=np.float32)

    # Normalize some metrics for a composite “artifact score”
    # (This is heuristic — we’re using it to locate listening regions.)
    flat_n = np.clip((flatness - np.percentile(flatness, 10)) / (np.percentile(flatness, 90) - np.percentile(flatness, 10) + 1e-6), 0.0, 1.0)
    mod_n = np.clip((mod - np.percentile(mod, 10)) / (np.percentile(mod, 90) - np.percentile(mod, 10) + 1e-6), 0.0, 1.0)
    spike_n = np.clip((spike_db - np.percentile(spike_db, 10)) / (np.percentile(spike_db, 90) - np.percentile(spike_db, 10) + 1e-6), 0.0, 1.0)

    score = cfg.w_flat * flat_n + cfg.w_mod * mod_n + cfg.w_spike * spike_n
    score = score.astype(np.float32)

    # Write CSV
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "mid_db", "band_db", "flatness", "mod", "flux", "spike_db", "score"])
        for i in range(len(times)):
            w.writerow([float(times[i]), float(mid_db[i]), float(band_db[i]), float(flatness[i]), float(mod[i]), float(flux[i]), float(spike_db[i]), float(score[i])])

    # Print top regions (cluster nearby peaks)
    idx_sorted = np.argsort(score)[::-1]
    chosen = []
    for idx in idx_sorted:
        t = float(times[idx])
        if all(abs(t - c) > cfg.cluster_gap_sec for c in chosen):
            chosen.append(t)
        if len(chosen) >= cfg.top_k:
            break
    chosen = sorted(chosen)

    print("\nTop candidate listening regions (center times):")
    for t in chosen:
        print(f"  {t:8.3f}s  (listen ~{cfg.cluster_len_sec:.1f}s around here)")

    # Spectrogram overview plot
    if spec_db_cols:
        S = np.stack(spec_db_cols, axis=1)  # (freq_plot, time_spec)
        plt.figure(figsize=(12, 5))
        plt.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=[spec_t[0], spec_t[-1], float(freqs[plot_idx[0]]), float(freqs[plot_idx[-1]])],
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Coarse spectrogram (dB), with band overlay")
        plt.axhline(start_hz, linewidth=1)
        plt.axhline(end_hz, linewidth=1)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "spectrogram_full.png"), dpi=160)
        plt.close()

    # Metrics plot
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(times, mid_db)
    ax1.set_ylabel("Mid dB")
    ax1.set_title("Time-series metrics (use this to validate assumptions)")

    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(times, band_db)
    ax2.set_ylabel("Band dB")

    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(times, flatness)
    ax3.set_ylabel("Flatness")

    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(times, mod, label="mod")
    ax4.plot(times, spike_db, label="spike_db")
    ax4.set_ylabel("Mod / Spike")

    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    ax5.plot(times, score)
    ax5.set_ylabel("Score")
    ax5.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics.png"), dpi=160)
    plt.close()

    # Pick one ROI for modulation spectrum: use highest score center
    roi_center = float(times[int(np.argmax(score))])
    half = cfg.cluster_len_sec * 0.5
    roi0 = max(0.0, roi_center - half)
    roi1 = min(n_samples / sr, roi_center + half)

    s0 = int(roi0 * sr)
    s1 = int(roi1 * sr)
    seg = x[s0:s1, :]

    # Bandpass ROI and compute modulation spectrum (FFT of envelope) :contentReference[oaicite:10]{index=10}
    bp = butter_bandpass(seg, sr, start_hz, end_hz, order=int(cfg.bandpass_order))
    mono = np.mean(bp, axis=1)
    env = np.abs(signal.hilbert(mono)).astype(np.float32)
    env = env - float(np.mean(env))

    n = int(2 ** math.ceil(math.log2(max(1024, len(env)))))
    E = np.fft.rfft(env * np.hanning(len(env)), n=n)
    f_env = np.fft.rfftfreq(n, d=1.0 / sr)
    mag_env = np.abs(E).astype(np.float32) + eps
    mag_env_db = 20.0 * np.log10(mag_env)

    m = f_env <= float(cfg.max_mod_hz)
    plt.figure(figsize=(10, 4))
    plt.plot(f_env[m], mag_env_db[m])
    plt.xlabel("Modulation frequency (Hz)")
    plt.ylabel("Envelope mag (dB)")
    plt.title(f"Modulation spectrum of band envelope (ROI {roi0:.2f}s–{roi1:.2f}s)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "modulation_spectrum.png"), dpi=160)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze AI/codec-ish shimmer: spectrogram + flatness + modulation + spike metrics.")
    ap.add_argument("input", help="Input audio file")
    ap.add_argument("--outdir", default="shimmer_analysis", help="Output directory for PNG/CSV")

    ap.add_argument("--start-hz", type=float, default=2000.0, help="Band start Hz")
    ap.add_argument("--end-hz", type=float, default=6000.0, help="Band end Hz")

    ap.add_argument("--center-hz", type=float, default=None, help="Alternative: band center Hz")
    ap.add_argument("--width-cents", type=float, default=None, help="Alternative: band width in cents")

    ap.add_argument("--n-fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--spec-stride", type=int, default=4)
    ap.add_argument("--max-freq-plot", type=float, default=12000.0)

    ap.add_argument("--smooth-ms", type=float, default=80.0)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--cluster-gap-sec", type=float, default=0.75)
    ap.add_argument("--cluster-len-sec", type=float, default=4.0)

    ap.add_argument("--max-mod-hz", type=float, default=200.0)
    ap.add_argument("--bandpass-order", type=int, default=4)

    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=True)
    x = x.astype(np.float32, copy=False)

    start_hz = float(args.start_hz)
    end_hz = float(args.end_hz)
    if args.center_hz is not None and args.width_cents is not None:
        start_hz, end_hz = cents_to_band(float(args.center_hz), float(args.width_cents))

    cfg = Cfg(
        start_hz=start_hz,
        end_hz=end_hz,
        n_fft=int(args.n_fft),
        hop=int(args.hop),
        spec_stride=int(args.spec_stride),
        max_freq_plot=float(args.max_freq_plot),
        smooth_ms=float(args.smooth_ms),
        top_k=int(args.top_k),
        cluster_gap_sec=float(args.cluster_gap_sec),
        cluster_len_sec=float(args.cluster_len_sec),
        max_mod_hz=float(args.max_mod_hz),
        bandpass_order=int(args.bandpass_order),
    )

    analyze_file(x, sr, cfg, args.outdir)
    print(f"\nWrote outputs to: {args.outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
