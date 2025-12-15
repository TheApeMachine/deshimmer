#!/usr/bin/env python3
"""
birdie_probe.py

Pick a time region (ROI), then:
- plot band spectrogram and "residual above local median" map
- write artifact-only audio (bins whose residual > threshold)

Uses SciPy STFT/ISTFT; note Zxx last axis is time by default. :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations
import argparse, os, math
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter


def db(x, eps=1e-12):
    return 20.0 * np.log10(np.maximum(x, eps))

def band_from_center(center_hz: float, width_cents: float):
    half = width_cents * 0.5
    ratio = 2.0 ** (half / 1200.0)
    return center_hz / ratio, center_hz * ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="birdie_probe_out")
    ap.add_argument("--t0", type=float, default=23.0, help="ROI start (sec)")
    ap.add_argument("--dur", type=float, default=4.0, help="ROI duration (sec)")

    ap.add_argument("--start-hz", type=float, default=5100.0)
    ap.add_argument("--end-hz", type=float, default=7200.0)
    ap.add_argument("--center-hz", type=float, default=None)
    ap.add_argument("--width-cents", type=float, default=None)

    ap.add_argument("--n-fft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=1024)

    ap.add_argument("--freq-med-bins", type=int, default=9, help="median filter width across frequency bins")
    ap.add_argument("--thr-db", type=float, default=8.0, help="residual threshold (dB) above local median")

    ap.add_argument("--write-artifact", default="artifact.wav")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    x, sr = sf.read(args.input, always_2d=True)
    x = x.astype(np.float32, copy=False)

    if args.center_hz is not None and args.width_cents is not None:
        args.start_hz, args.end_hz = band_from_center(float(args.center_hz), float(args.width_cents))

    s0 = int(args.t0 * sr)
    s1 = int((args.t0 + args.dur) * sr)
    s0 = max(0, min(s0, x.shape[0]))
    s1 = max(0, min(s1, x.shape[0]))
    seg = x[s0:s1, :]

    n_fft = args.n_fft
    hop = args.hop
    noverlap = n_fft - hop
    win = "hann"

    # STFT per channel
    Zs = []
    for ch in range(seg.shape[1]):
        f, t, Z = signal.stft(
            seg[:, ch],
            fs=sr,
            window=win,
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        Zs.append(Z)
    Zs = np.stack(Zs, axis=-1)  # (freq, time, ch)

    band = np.where((f >= args.start_hz) & (f <= args.end_hz))[0]
    if band.size < 8:
        raise SystemExit("Band too narrow for this FFT size; increase n-fft or widen band.")

    mag = np.mean(np.abs(Zs[band, :, :]), axis=2)  # (band_f, time)
    L = np.log(np.maximum(mag, 1e-12))

    k = int(max(3, args.freq_med_bins))
    if k % 2 == 0:
        k += 1

    # Local baseline across frequency (per time)
    L_med = median_filter(L, size=(k, 1), mode="nearest")
    residual_db = (L - L_med) * (20.0 / np.log(10.0))

    mask = residual_db > float(args.thr_db)

    # Artifact-only STFT: keep only masked bins in band
    Z_art = np.zeros_like(Zs)
    for ch in range(Zs.shape[2]):
        Zb = Zs[band, :, ch]
        Z_art[band, :, ch] = Zb * mask

    # ISTFT back per channel
    ys = []
    for ch in range(Z_art.shape[2]):
        _, y = signal.istft(
            Z_art[:, :, ch],
            fs=sr,
            window=win,
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            input_onesided=True,
            boundary=False,
        )
        ys.append(y.astype(np.float32))
    y_art = np.stack(ys, axis=1)

    # Trim/pad to ROI length
    n = seg.shape[0]
    if y_art.shape[0] > n:
        y_art = y_art[:n, :]
    elif y_art.shape[0] < n:
        y_art = np.pad(y_art, ((0, n - y_art.shape[0]), (0, 0)))

    out_wav = os.path.join(args.outdir, args.write_artifact)
    sf.write(out_wav, y_art, sr)

    # Plots
    plt.figure(figsize=(12, 4))
    plt.imshow(
        db(mag),
        origin="lower",
        aspect="auto",
        extent=[t[0] + args.t0, t[-1] + args.t0, f[band[0]], f[band[-1]]],
    )
    plt.title("ROI band magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roi_band_spectrogram.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.imshow(
        residual_db,
        origin="lower",
        aspect="auto",
        extent=[t[0] + args.t0, t[-1] + args.t0, f[band[0]], f[band[-1]]],
    )
    plt.title(f"Residual above local median (dB), thr={args.thr_db:.1f} dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roi_residual_map.png"), dpi=160)
    plt.close()

    print(f"Wrote: {out_wav}")
    print(f"Wrote: {args.outdir}/roi_band_spectrogram.png")
    print(f"Wrote: {args.outdir}/roi_residual_map.png")


if __name__ == "__main__":
    main()
