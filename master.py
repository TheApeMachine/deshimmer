#!/usr/bin/env python3
"""
deshimmer5.py

Practical anti-shimmer processor for AI/codec-ish artifacts:
- Focus on a target band (e.g., ~5.1–7.2 kHz).
- Detect narrow, flickery outliers above a local median baseline ("birdies"-like).
- Suppress them with a soft-knee attenuation.
- Optional: slight random-phase blend in noise-like frames to "de-crystallize".
- Padding + trim + fade to avoid start/end clicks.

Add-ons (taste-neutral "repair" stages):
- Smart spectral denoise (noise-floor management) using minimum-statistics-ish noise PSD tracking
  + Wiener-like gain + time/frequency smoothing to avoid "musical noise".
- Smart dynamic EQ (de-resonator) that tames persistent narrow resonances across a wider band.
"""

from __future__ import annotations
import argparse, math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter, uniform_filter1d


@dataclass
class Params:
    # --- Original de-shimmer band ---
    start_hz: float = 5100.0
    end_hz: float = 7200.0
    edge_hz: float = 200.0

    n_fft: int = 2048
    hop: int = 512

    # Noise-likeness gate via spectral flatness (geo/arith of power spectrum).
    flat_start: float = 0.25
    flat_end: float = 0.70

    # Birdie suppression (within start_hz..end_hz)
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

    # ----------------------------------------------------------------------
    # Smart noise-floor management: spectral denoise (taste-neutral)
    # ----------------------------------------------------------------------
    denoise: float = 0.0             # 0..1 overall strength
    dn_start_hz: float = 120.0
    dn_end_hz: float = 16000.0
    dn_edge_hz: float = 200.0

    dn_floor_db: float = -18.0       # minimum gain floor (dB), avoids dead-silent artifacts
    dn_psd_smooth_ms: float = 50.0   # smooth PSD before min-tracking
    dn_minwin_ms: float = 400.0      # minimum-statistics window for tracking minima
    dn_up_db_per_s: float = 3.0      # how fast noise estimate can rise (dB/s, amplitude dB)
    dn_attack_ms: float = 5.0
    dn_release_ms: float = 120.0
    dn_freq_smooth_bins: int = 3     # smooth gain across frequency (reduce musical noise)

    # ----------------------------------------------------------------------
    # Smart dynamic EQ: de-resonator (taste-neutral, subtractive only)
    # ----------------------------------------------------------------------
    deres: float = 0.0               # 0..1 strength
    deq_start_hz: float = 180.0
    deq_end_hz: float = 12000.0
    deq_edge_hz: float = 150.0

    deq_freq_med_bins: int = 31      # median width across frequency for baseline
    deq_thr_db: float = 6.0          # residual threshold above local median
    deq_slope: float = 0.7           # how hard to push down peaks above threshold
    deq_max_att_db: float = 8.0      # cap reduction (safety)

    # If too many bins exceed threshold, assume it's legit broadband/tonal content -> back off
    deq_density_lo: float = 0.03
    deq_density_hi: float = 0.20

    # "Smart" bit: persistence favors stationary resonances over moving harmonics
    deq_persist_ms: float = 600.0
    deq_persist_thr_db: float = 2.5

    deq_freq_smooth_bins: int = 5
    deq_tonal_boost_db: float = 6.0  # raises threshold when frame is tonal (based on flatness)


def band_from_center(center_hz: float, width_cents: float) -> Tuple[float, float]:
    half = width_cents * 0.5
    ratio = 2.0 ** (half / 1200.0)
    return center_hz / ratio, center_hz * ratio


def _edge_taper(freqs: np.ndarray, band_idx: np.ndarray, start_hz: float, end_hz: float, edge_hz: float) -> np.ndarray:
    """Cosine taper inside edge_hz of the band edges."""
    w = np.ones(band_idx.size, dtype=np.float32)
    edge = float(max(0.0, edge_hz))
    if edge <= 0.0 or band_idx.size == 0:
        return w
    fb = freqs[band_idx].astype(np.float32)
    lo, hi = float(start_hz), float(end_hz)

    m = fb < lo + edge
    if np.any(m):
        rel = (fb[m] - lo) / edge
        w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1))

    m2 = fb > hi - edge
    if np.any(m2):
        rel = (hi - fb[m2]) / edge
        w[m2] = np.minimum(w[m2], 0.5 - 0.5 * np.cos(np.pi * np.clip(rel, 0, 1)))

    return w


def _frame_coeff(hop: int, sr: int, ms: float) -> float:
    """Per-frame EMA coefficient for time constant ms."""
    tau = max(1e-4, float(ms) / 1000.0)
    return float(math.exp(-float(hop) / (float(sr) * tau)))


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
    eps = 1e-12

    def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
        lo = float(max(0.0, lo_hz))
        hi = float(min(nyq, hi_hz))
        if hi <= lo:
            return np.array([], dtype=np.int64)
        return np.where((freqs >= lo) & (freqs <= hi))[0]

    # --- Shimmer band ---
    start_hz = float(max(0.0, p.start_hz))
    end_hz = float(min(nyq, p.end_hz))
    if end_hz <= start_hz:
        raise ValueError("end_hz must be > start_hz")

    band_idx = _idx(start_hz, end_hz)
    if band_idx.size < 8:
        raise ValueError("Band too narrow for this FFT size; increase n_fft or widen band.")
    w_sh = _edge_taper(freqs, band_idx, start_hz, end_hz, p.edge_hz)

    # --- Denoise band ---
    dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
    w_dn = _edge_taper(freqs, dn_idx, float(max(0.0, p.dn_start_hz)), float(min(nyq, p.dn_end_hz)), p.dn_edge_hz)

    # --- De-resonator band ---
    deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)
    w_deq = _edge_taper(freqs, deq_idx, float(max(0.0, p.deq_start_hz)), float(min(nyq, p.deq_end_hz)), p.deq_edge_hz)

    # --- Denoise state (minimum-statistics-ish) ---
    dn_strength = float(np.clip(p.denoise, 0.0, 1.0))
    psd_sm = None
    noise_psd = None
    block_min = None
    block_ctr = 0
    minwin_frames = max(4, int((sr * (p.dn_minwin_ms / 1000.0)) / hop))
    a_psd = _frame_coeff(hop, sr, p.dn_psd_smooth_ms)
    dn_att = _frame_coeff(hop, sr, p.dn_attack_ms)
    dn_rel = _frame_coeff(hop, sr, p.dn_release_ms)
    dn_floor = float(10.0 ** (float(p.dn_floor_db) / 20.0))
    dn_freq_smooth = int(max(1, p.dn_freq_smooth_bins))
    dn_gain_sm = np.ones(dn_idx.size, dtype=np.float32) if dn_idx.size else None

    # noise PSD is power-domain; convert dB/s (amplitude dB) to a power multiplier per block
    dn_block_sec = float(minwin_frames) * float(hop) / float(sr)
    dn_up_lin = float(10.0 ** ((float(p.dn_up_db_per_s) * dn_block_sec) / 10.0))

    # --- De-resonator state ---
    deq_strength = float(np.clip(p.deres, 0.0, 1.0))
    deq_persist = np.zeros(deq_idx.size, dtype=np.float32) if deq_idx.size else None
    a_persist = _frame_coeff(hop, sr, p.deq_persist_ms)
    deq_freq_med = int(max(3, p.deq_freq_med_bins))
    if deq_freq_med % 2 == 0:
        deq_freq_med += 1
    deq_freq_smooth = int(max(1, p.deq_freq_smooth_bins))

    rng = np.random.default_rng(int(p.seed))

    y = np.zeros((n_samples + n_fft, n_ch), dtype=np.float32)
    wsum = np.zeros(n_samples + n_fft, dtype=np.float32)

    prev_flux_db = None
    processed_frames = 0
    total_frames = 0

    for s in range(0, n_samples, hop):
        total_frames += 1

        frame = np.zeros((n_fft, n_ch), dtype=np.float32)
        chunk = x0[s:s + n_fft, :]
        frame[:chunk.shape[0], :] = chunk
        frame_w = frame * win[:, None]

        spec = np.fft.rfft(frame_w, n=n_fft, axis=0)  # (freq, ch)

        # ------------------------------------------------------------------
        # Global-ish measures used for "smart" gating
        # ------------------------------------------------------------------
        psd = np.mean(np.abs(spec) ** 2, axis=1).astype(np.float32) + eps  # power per bin (mono)

        if dn_idx.size:
            mag_dn = np.mean(np.abs(spec[dn_idx, :]), axis=1).astype(np.float32) + eps
            Pdn = mag_dn ** 2
            flat_dn = float(np.exp(np.mean(np.log(Pdn + eps))) / (np.mean(Pdn) + eps))
            w_noise_full = float(np.clip((flat_dn - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))
            band_db = 10.0 * math.log10(float(np.mean(Pdn)) + eps)
        else:
            flat_all = float(np.exp(np.mean(np.log(psd + eps))) / (np.mean(psd) + eps))
            w_noise_full = float(np.clip((flat_all - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))
            band_db = 10.0 * math.log10(float(np.mean(psd)) + eps)

        if prev_flux_db is None:
            flux = 0.0
        else:
            flux = max(0.0, band_db - prev_flux_db)
        prev_flux_db = band_db
        w_trans = float(np.clip((flux - p.flux_thr_db) / max(1e-6, p.flux_range_db), 0.0, 1.0))
        w_nontrans = 1.0 - w_trans

        # ------------------------------------------------------------------
        # (A) Smart noise-floor management: spectral denoise
        # ------------------------------------------------------------------
        if dn_strength > 1e-6 and dn_idx.size:
            if psd_sm is None:
                psd_sm = psd.copy()
                noise_psd = psd.copy()
                block_min = psd.copy()
                block_ctr = 1
            else:
                psd_sm = a_psd * psd_sm + (1.0 - a_psd) * psd
                block_min = np.minimum(block_min, psd_sm)
                block_ctr += 1

            if block_ctr >= minwin_frames:
                noise_psd = np.minimum(noise_psd * dn_up_lin, block_min).astype(np.float32, copy=False)
                block_min.fill(np.inf)
                block_ctr = 0

            # Wiener-ish gain: g = snr/(snr+k). k grows with strength.
            snr = psd_sm[dn_idx] / (noise_psd[dn_idx] + eps)
            k = 1.0 + 3.0 * dn_strength
            g_inst = snr / (snr + k)
            g_inst = dn_floor + (1.0 - dn_floor) * g_inst
            g_inst = np.clip(g_inst, dn_floor, 1.0).astype(np.float32)

            # Attack/release smoothing on gain (down fast, up slow)
            down = g_inst < dn_gain_sm
            dn_gain_sm[down] = dn_att * dn_gain_sm[down] + (1.0 - dn_att) * g_inst[down]
            dn_gain_sm[~down] = dn_rel * dn_gain_sm[~down] + (1.0 - dn_rel) * g_inst[~down]

            # Smooth across frequency to reduce "musical noise"/zippery artifacts
            g_dn = uniform_filter1d(dn_gain_sm, size=dn_freq_smooth, mode="nearest")

            # Apply more when frame is noise-like; back off on transients
            depth = dn_strength * (0.5 + 0.5 * w_noise_full) * w_nontrans
            g_eff = 1.0 - (depth * w_dn) * (1.0 - g_dn)
            spec[dn_idx, :] *= g_eff[:, None]

        # ------------------------------------------------------------------
        # (B) Smart dynamic EQ: de-resonator (persistent narrow peaks)
        # ------------------------------------------------------------------
        if deq_strength > 1e-6 and deq_idx.size:
            mag_deq = np.mean(np.abs(spec[deq_idx, :]), axis=1).astype(np.float32) + eps
            L = np.log(mag_deq)
            L_med = median_filter(L, size=deq_freq_med, mode="nearest")
            residual_db = (L - L_med) * (20.0 / np.log(10.0))

            # Raise threshold in tonal frames (low flatness) to avoid chewing harmonics
            thr_eff = float(p.deq_thr_db) + float(p.deq_tonal_boost_db) * (1.0 - w_noise_full)
            over = residual_db - thr_eff
            pos = np.maximum(0.0, over).astype(np.float32)

            mask = pos > 0.0
            density = float(np.mean(mask)) if mask.size else 0.0
            w_narrow = 1.0 - float(np.clip((density - p.deq_density_lo) / max(1e-6, (p.deq_density_hi - p.deq_density_lo)), 0.0, 1.0))

            # Persistence memory: stationary resonances accumulate, moving peaks don't
            deq_persist = a_persist * deq_persist + (1.0 - a_persist) * pos
            gate = np.clip(deq_persist / max(1e-6, float(p.deq_persist_thr_db)), 0.0, 1.0)

            att_db = float(p.deq_slope) * pos * gate
            att_db = np.minimum(att_db, float(p.deq_max_att_db)).astype(np.float32)
            gain = (10.0 ** (-att_db / 20.0)).astype(np.float32)

            depth = deq_strength * w_nontrans * w_narrow
            g_eff = 1.0 - (depth * w_deq) * (1.0 - gain)
            if deq_freq_smooth > 1:
                g_eff = uniform_filter1d(g_eff, size=deq_freq_smooth, mode="nearest").astype(np.float32, copy=False)
            spec[deq_idx, :] *= g_eff[:, None]

        # ------------------------------------------------------------------
        # (C) Original shimmer suppression in target band
        # ------------------------------------------------------------------
        band_spec = spec[band_idx, :]
        mag_band = np.mean(np.abs(band_spec), axis=1).astype(np.float32) + eps

        # Flatness in shimmer band
        P = mag_band ** 2
        flat = float(np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps))
        w_noise = float(np.clip((flat - p.flat_start) / max(1e-6, (p.flat_end - p.flat_start)), 0.0, 1.0))

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
        w_narrow = 1.0 - float(np.clip((density - p.density_lo) / max(1e-6, (p.density_hi - p.density_lo)), 0.0, 1.0))

        w_depth = w_noise * w_nontrans * w_narrow
        if w_depth > 1e-3:
            processed_frames += 1

        # Soft-knee attenuation in dB
        att_db = np.zeros_like(over, dtype=np.float32)
        att_db[mask] = (float(p.slope) * over[mask]).astype(np.float32)
        gain = (10.0 ** (-att_db / 20.0)).astype(np.float32)

        # Apply with overall depth and edge taper
        g_eff = 1.0 - (w_depth * w_sh) * (1.0 - gain)
        spec[band_idx, :] *= g_eff[:, None]

        # Optional random-phase blend (only in noise-like frames)
        nr = float(np.clip(p.noise_resynth, 0.0, 1.0))
        if nr > 0.0 and w_noise > 1e-3:
            depth = (nr * w_noise) * w_sh
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
        print(f"Processed shimmer frames: {processed_frames}/{total_frames} ({100.0*processed_frames/max(1,total_frames):.1f}%)")

    return y.squeeze()


def main() -> int:
    ap = argparse.ArgumentParser(description="Reduce AI/codec-ish shimmer, plus optional smart denoise and smart de-resonating dynamic EQ.")
    ap.add_argument("input")
    ap.add_argument("output")

    # De-shimmer band
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

    # Smart denoise (noise-floor management)
    ap.add_argument("--denoise", type=float, default=0.0, help="0..1 spectral noise floor reduction")
    ap.add_argument("--dn-start-hz", type=float, default=120.0)
    ap.add_argument("--dn-end-hz", type=float, default=16000.0)
    ap.add_argument("--dn-edge-hz", type=float, default=200.0)
    ap.add_argument("--dn-floor-db", type=float, default=-18.0)
    ap.add_argument("--dn-psd-smooth-ms", type=float, default=50.0)
    ap.add_argument("--dn-minwin-ms", type=float, default=400.0)
    ap.add_argument("--dn-up-db-per-s", type=float, default=3.0)
    ap.add_argument("--dn-attack-ms", type=float, default=5.0)
    ap.add_argument("--dn-release-ms", type=float, default=120.0)
    ap.add_argument("--dn-freq-smooth-bins", type=int, default=3)

    # Smart dynamic EQ (de-resonator)
    ap.add_argument("--deres", type=float, default=0.0, help="0..1 de-resonator (dynamic EQ)")
    ap.add_argument("--deq-start-hz", type=float, default=180.0)
    ap.add_argument("--deq-end-hz", type=float, default=12000.0)
    ap.add_argument("--deq-edge-hz", type=float, default=150.0)
    ap.add_argument("--deq-freq-med-bins", type=int, default=31)
    ap.add_argument("--deq-thr-db", type=float, default=6.0)
    ap.add_argument("--deq-slope", type=float, default=0.7)
    ap.add_argument("--deq-max-att-db", type=float, default=8.0)
    ap.add_argument("--deq-density-lo", type=float, default=0.03)
    ap.add_argument("--deq-density-hi", type=float, default=0.20)
    ap.add_argument("--deq-persist-ms", type=float, default=600.0)
    ap.add_argument("--deq-persist-thr-db", type=float, default=2.5)
    ap.add_argument("--deq-freq-smooth-bins", type=int, default=5)
    ap.add_argument("--deq-tonal-boost-db", type=float, default=6.0)

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

        denoise=float(args.denoise),
        dn_start_hz=float(args.dn_start_hz),
        dn_end_hz=float(args.dn_end_hz),
        dn_edge_hz=float(args.dn_edge_hz),
        dn_floor_db=float(args.dn_floor_db),
        dn_psd_smooth_ms=float(args.dn_psd_smooth_ms),
        dn_minwin_ms=float(args.dn_minwin_ms),
        dn_up_db_per_s=float(args.dn_up_db_per_s),
        dn_attack_ms=float(args.dn_attack_ms),
        dn_release_ms=float(args.dn_release_ms),
        dn_freq_smooth_bins=int(args.dn_freq_smooth_bins),

        deres=float(args.deres),
        deq_start_hz=float(args.deq_start_hz),
        deq_end_hz=float(args.deq_end_hz),
        deq_edge_hz=float(args.deq_edge_hz),
        deq_freq_med_bins=int(args.deq_freq_med_bins),
        deq_thr_db=float(args.deq_thr_db),
        deq_slope=float(args.deq_slope),
        deq_max_att_db=float(args.deq_max_att_db),
        deq_density_lo=float(args.deq_density_lo),
        deq_density_hi=float(args.deq_density_hi),
        deq_persist_ms=float(args.deq_persist_ms),
        deq_persist_thr_db=float(args.deq_persist_thr_db),
        deq_freq_smooth_bins=int(args.deq_freq_smooth_bins),
        deq_tonal_boost_db=float(args.deq_tonal_boost_db),

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
