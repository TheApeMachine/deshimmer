#!/usr/bin/env python3
"""
deshimmer5.py

Practical anti-shimmer processor for AI/codec-ish artifacts:
- Focus on a target band (e.g., ~5.1–7.2 kHz).
- Detect narrow, flickery outliers above a local median baseline ("birdies"-like). :contentReference[oaicite:4]{index=4}
- Suppress them with a soft-knee attenuation.
- Optional: slight random-phase blend in noise-like frames to "de-crystallize".
- Padding + trim + fade to avoid start/end clicks.
"""

from __future__ import annotations
import argparse, os, math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter


@dataclass
class Params:
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0

    n_fft: int = 2048
    hop: int = 512

    # Noise-likeness gate via spectral flatness (geo/arith of power spectrum). :contentReference[oaicite:5]{index=5}
    flat_start: float = 0.25
    flat_end: float = 0.70

    # Birdie suppression
    freq_med_bins: int = 9          # median filter width across frequency
    thr_db: float = 8.0             # residual threshold above local median (dB)
    slope: float = 0.6              # attenuation slope: att_db = slope * (residual_db - thr_db)

    # If MANY bins exceed threshold, treat as real broadband event; do less
    density_lo: float = 0.02
    density_hi: float = 0.15

    # Transient protect (based on band energy jump)
    flux_thr_db: float = 6.0
    flux_range_db: float = 8.0

    # Optional random-phase blend (only in noise-like frames)
    noise_resynth: float = 0.0      # 0..1

    # Wet/dry
    mix: float = 1.0

    # Padding & fade
    pad: bool = True
    fade_ms: float = 5.0

    seed: int = 0
    debug: bool = False


def band_from_center(center_hz: float, width_cents: float) -> Tuple[float, float]:
    half = width_cents * 0.5
    ratio = 2.0 ** (half / 1200.0)
    return center_hz / ratio, center_hz * ratio


def process(x: np.ndarray, sr: int, p: Params) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, None]
    x = x.astype(np.float32, copy=False)

    n_fft = int(p.n_fft)
    hop = int(p.hop)
    if hop <= 0 or n_fft <= 0 or hop > n_fft:
        raise ValueError("Invalid n_fft/hop")

    # Pad to avoid boundary clicks
    if p.pad:
        x0 = np.pad(x, ((n_fft, n_fft), (0, 0)), mode="constant")
    else:
        x0 = x

    n_samples, n_ch = x0.shape
    win = np.hanning(n_fft).astype(np.float32)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    nyq = float(freqs[-1])

    start_hz = float(max(0.0, p.start_hz))
    end_hz = float(min(nyq, p.end_hz))
    if end_hz <= start_hz:
        raise ValueError("end_hz must be > start_hz")

    band_idx = np.where((freqs >= start_hz) & (freqs <= end_hz))[0]
    if band_idx.size < 8:
        raise ValueError("Band too narrow for this FFT size; increase n_fft or widen band.")

    # Edge taper weights
    w_f = np.ones(band_idx.size, dtype=np.float32)
    edge = float(max(0.0, p.edge_hz))
    if edge > 0:
        fb = freqs[band_idx].astype(np.float32)
        lo, hi = start_hz, end_hz
        m = fb < lo + edge
        if np.any(m):
            rel = (fb[m] - lo) / edge
            w_f[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1))
        m2 = fb > hi - edge
        if np.any(m2):
            rel = (hi - fb[m2]) / edge
            w_f[m2] = np.minimum(w_f[m2], 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1)))

    eps = 1e-12
    rng = np.random.default_rng(int(p.seed))

    y = np.zeros((n_samples + n_fft, n_ch), dtype=np.float32)
    wsum = np.zeros(n_samples + n_fft, dtype=np.float32)

    prev_band_db = None
    processed_frames = 0
    total_frames = 0

    for s in range(0, n_samples, hop):
        total_frames += 1

        frame = np.zeros((n_fft, n_ch), dtype=np.float32)
        chunk = x0[s:s + n_fft, :]
        frame[:chunk.shape[0], :] = chunk
        frame_w = frame * win[:, None]

        spec = np.fft.rfft(frame_w, n=n_fft, axis=0)  # (freq, ch)

        # Band magnitude (mono decision)
        band_spec = spec[band_idx, :]
        mag_band = np.mean(np.abs(band_spec), axis=1).astype(np.float32) + eps

        # Flatness (power spectrum): geo/arith mean :contentReference[oaicite:6]{index=6}
        P = mag_band ** 2
        flat = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))
        w_noise = float(np.clip((flat - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))

        # Band energy / transient protect
        band_db = 10.0 * math.log10(float(np.mean(P)) + eps)
        if prev_band_db is None:
            flux = 0.0
        else:
            flux = max(0.0, band_db - prev_band_db)
        prev_band_db = band_db
        w_trans = float(np.clip((flux - p.flux_thr_db) / max(1e-6, p.flux_range_db), 0.0, 1.0))
        w_nontrans = 1.0 - w_trans

        # Local median baseline across frequency (log-mag)
        L = np.log(mag_band)
        k = int(max(3, p.freq_med_bins))
        if k % 2 == 0:
            k += 1
        L_med = median_filter(L, size=k, mode="nearest")
        residual_db = (L - L_med) * (20.0 / np.log(10.0))

        over = residual_db - float(p.thr_db)
        mask = over > 0.0
        density = float(np.mean(mask)) if mask.size else 0.0
        # If density is high, more likely broadband musical event
        w_narrow = 1.0 - float(np.clip((density - p.density_lo) / max(1e-6, (p.density_hi - p.density_lo)), 0.0, 1.0))

        w_depth = w_noise * w_nontrans * w_narrow
        if w_depth > 1e-3:
            processed_frames += 1

        # Soft-knee attenuation in dB
        att_db = np.zeros_like(over, dtype=np.float32)
        att_db[mask] = (float(p.slope) * over[mask]).astype(np.float32)
        gain = (10.0 ** (-att_db / 20.0)).astype(np.float32)

        # Apply with overall depth and edge taper
        g_eff = 1.0 - (w_depth * w_f) * (1.0 - gain)
        spec[band_idx, :] *= g_eff[:, None]

        # Optional noise resynth (random-phase blend) in noise-like frames
        nr = float(np.clip(p.noise_resynth, 0.0, 1.0))
        if nr > 0.0 and w_noise > 1e-3:
            depth = (nr * w_noise) * w_f
            phi = rng.uniform(0.0, 2.0 * np.pi, size=band_idx.size).astype(np.float32)
            zph = (np.cos(phi) + 1j * np.sin(phi)).astype(np.complex64)
            for ch in range(n_ch):
                Zb = spec[band_idx, ch]
                mag = np.abs(Zb).astype(np.float32)
                Zrand = mag * zph
                spec[band_idx, ch] = (1.0 - depth) * Zb + depth * Zrand

        # iFFT + OLA
        out = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)
        out *= win[:, None]
        y[s:s + n_fft, :] += out
        wsum[s:s + n_fft] += win ** 2

    # Normalize OLA
    wsum = np.maximum(wsum, 1e-12)
    y = y[:n_samples, :] / wsum[:n_samples, None]

    # Unpad
    if p.pad:
        y = y[n_fft:-n_fft, :]
        x_ref = x
    else:
        x_ref = x0[:y.shape[0], :]

    # Wet/dry
    mix = float(np.clip(p.mix, 0.0, 1.0))
    y = mix * y + (1.0 - mix) * x_ref

    # Tiny fade to be safe
    fade = int(sr * (float(p.fade_ms) / 1000.0))
    if fade > 1 and y.shape[0] > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        y[:fade, :] *= ramp
        y[-fade:, :] *= ramp[::-1]

    if p.debug:
        print(f"Processed frames: {processed_frames}/{total_frames} ({100.0*processed_frames/max(1,total_frames):.1f}%)")

    return y.squeeze()


def main() -> int:
    ap = argparse.ArgumentParser(description="Reduce AI/codec-ish shimmer by suppressing narrow outliers in a target band.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--start-hz", type=float, default=5100.0)
    ap.add_argument("--end-hz", type=float, default=7200.0)
    ap.add_argument("--center-hz", type=float, default=None)
    ap.add_argument("--width-cents", type=float, default=None)
    ap.add_argument("--edge-hz", type=float, default=200.0)

    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)

    ap.add_argument("--flat-start", type=float, default=0.25)
    ap.add_argument("--flat-end", type=float, default=0.70)

    ap.add_argument("--freq-med-bins", type=int, default=9)
    ap.add_argument("--thr-db", type=float, default=8.0)
    ap.add_argument("--slope", type=float, default=0.6)
    ap.add_argument("--density-lo", type=float, default=0.02)
    ap.add_argument("--density-hi", type=float, default=0.15)

    ap.add_argument("--flux-thr-db", type=float, default=6.0)
    ap.add_argument("--flux-range-db", type=float, default=8.0)

    ap.add_argument("--noise-resynth", type=float, default=0.0)
    ap.add_argument("--mix", type=float, default=1.0)

    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--fade-ms", type=float, default=5.0)

    ap.add_argument("--write-diff", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=True)
    x = x.astype(np.float32, copy=False)

    start_hz = float(args.start_hz)
    end_hz = float(args.end_hz)
    if args.center_hz is not None and args.width_cents is not None:
        start_hz, end_hz = band_from_center(float(args.center_hz), float(args.width_cents))

    p = Params(
        start_hz=start_hz,
        end_hz=end_hz,
        edge_hz=float(args.edge_hz),
        n_fft=int(args.n_fft),
        hop=int(args.hop),
        flat_start=float(args.flat_start),
        flat_end=float(args.flat_end),
        freq_med_bins=int(args.freq_med_bins),
        thr_db=float(args.thr_db),
        slope=float(args.slope),
        density_lo=float(args.density_lo),
        density_hi=float(args.density_hi),
        flux_thr_db=float(args.flux_thr_db),
        flux_range_db=float(args.flux_range_db),
        noise_resynth=float(args.noise_resynth),
        mix=float(args.mix),
        pad=(not args.no_pad),
        fade_ms=float(args.fade_ms),
        seed=int(args.seed),
        debug=bool(args.debug),
    )

    y = process(x, sr, p)
    y2 = y[:, None] if y.ndim == 1 else y

    # Prevent clipping
    peak = float(np.max(np.abs(y2))) if y2.size else 0.0
    if peak > 0.999:
        y2 = (y2 / peak * 0.999).astype(np.float32)

    sf.write(args.output, y2, sr, subtype="PCM_24")

    if args.write_diff:
        diff = (x[:y2.shape[0], :] - y2).astype(np.float32)
        sf.write(args.write_diff, diff, sr, subtype="PCM_24")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
