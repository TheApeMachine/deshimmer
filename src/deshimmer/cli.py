"""Command line interface for batch audio deshimmering, denoising, and mastering."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from .core import (
    DebugCollector,
    DebugParams,
    MasterParams,
    Params,
    _as_2d,
    _lin_to_db,
    band_from_center,
    master_post,
    measure_lufs,
    measure_rms_dbfs,
    measure_true_peak_db,
    process_stft,
    render_debug,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Repair AI/codec-ish shimmer + smart denoise + smart de-resonator with psychoacoustic bounds."
    )

    ap.add_argument("input", help="Path to input audio file")
    ap.add_argument("output", help="Path to output audio file")

    # --- De-shimmer band ---
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
    ap.add_argument(
        "--repair-only",
        action="store_true",
        help="Disable L/R alignment, spectral balancing, and spectral carving",
    )
    ap.add_argument(
        "--tonal-repair",
        type=float,
        default=Params().tonal_repair,
        help="Self-referenced partial correction amount (default on)",
    )
    ap.add_argument("--tonal-max-shift-cents", type=float, default=Params().tonal_max_shift_cents)

    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--fade-ms", type=float, default=5.0)

    # --- Smart denoise ---
    ap.add_argument("--denoise", type=float, default=0.0)
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

    # --- Smart de-resonator ---
    ap.add_argument("--deres", type=float, default=0.0)
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
    ap.add_argument("--deq-time-floor", action="store_true")
    ap.add_argument("--deq-floor-smooth-ms", type=float, default=80.0)
    ap.add_argument("--deq-floor-rise-db-per-s", type=float, default=1.0)
    ap.add_argument("--deq-floor-thr-db", type=float, default=3.0)

    ap.add_argument("--delta-listen", action="store_true")

    # --- Downward expander ---
    ap.add_argument("--expander", action="store_true")
    ap.add_argument("--exp-start-hz", type=float, default=3000.0)
    ap.add_argument("--exp-end-hz", type=float, default=8000.0)
    ap.add_argument("--exp-threshold-db", type=float, default=-45.0)
    ap.add_argument("--exp-ratio", type=float, default=2.0)
    ap.add_argument("--exp-attack-ms", type=float, default=10.0)
    ap.add_argument("--exp-release-ms", type=float, default=150.0)

    # --- HPSS-ish ---
    ap.add_argument("--hpss", action="store_true")
    ap.add_argument("--hpss-start-hz", type=float, default=3000.0)
    ap.add_argument("--hpss-end-hz", type=float, default=8000.0)
    ap.add_argument("--hpss-time-frames", type=int, default=21)
    ap.add_argument("--hpss-freq-bins", type=int, default=17)
    ap.add_argument("--hpss-no-harmonic-only", action="store_true")

    # --- Phase blur ---
    ap.add_argument("--phase-blur", type=float, default=0.0)
    ap.add_argument("--pb-start-hz", type=float, default=3000.0)
    ap.add_argument("--pb-end-hz", type=float, default=8000.0)
    ap.add_argument("--pb-no-harmonic-only", action="store_true")

    # --- HF resynthesis ---
    ap.add_argument("--hf-resynth", action="store_true")
    ap.add_argument("--hf-lp-hz", type=float, default=3000.0)
    ap.add_argument("--hf-src-lo-hz", type=float, default=1000.0)
    ap.add_argument("--hf-src-hi-hz", type=float, default=2000.0)
    ap.add_argument("--hf-drive", type=float, default=2.0)
    ap.add_argument("--hf-hp-hz", type=float, default=3000.0)
    ap.add_argument("--hf-mix", type=float, default=0.35)
    ap.add_argument("--hf-tilt-lp-hz", type=float, default=12000.0)
    ap.add_argument("--hf-zero-phase", action="store_true")
    ap.add_argument("--no-hf-zero-phase", action="store_true")

    # --- Delivery mastering ---
    ap.add_argument("--master", action="store_true")
    ap.add_argument("--hp-hz", type=float, default=20.0)
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    ap.add_argument("--target-rms-dbfs", type=float, default=-16.0)
    ap.add_argument("--norm-max-gain-db", type=float, default=12.0)
    ap.add_argument("--norm-max-atten-db", type=float, default=24.0)
    ap.add_argument("--ceiling-dbtp", type=float, default=-1.0)
    ap.add_argument("--lim-lookahead-ms", type=float, default=5.0)
    ap.add_argument("--lim-release-ms", type=float, default=100.0)
    ap.add_argument("--tp-os", type=int, default=4)

    # Output
    ap.add_argument("--subtype", type=str, default="PCM_24")

    # Debug / diagnostics
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-dir", type=str, default=None)
    ap.add_argument("--debug-stride", type=int, default=8)
    ap.add_argument("--debug-spec-nfft", type=int, default=2048)
    ap.add_argument("--debug-spec-hop", type=int, default=512)
    ap.add_argument("--debug-spec-max-frames", type=int, default=2200)
    ap.add_argument("--debug-spec-max-hz", type=float, default=20000.0)

    ap.add_argument("--write-diff", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)

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
        enhance=not args.repair_only,
        tonal_repair=float(args.tonal_repair),
        tonal_max_shift_cents=float(args.tonal_max_shift_cents),
        pad=(not args.no_pad),
        fade_ms=float(args.fade_ms),
        seed=int(args.seed),
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
        deq_time_floor=bool(args.deq_time_floor),
        deq_floor_smooth_ms=float(args.deq_floor_smooth_ms),
        deq_floor_rise_db_per_s=float(args.deq_floor_rise_db_per_s),
        deq_floor_thr_db=float(args.deq_floor_thr_db),
        delta_listen=bool(args.delta_listen),
        expander=bool(args.expander),
        exp_start_hz=float(args.exp_start_hz),
        exp_end_hz=float(args.exp_end_hz),
        exp_threshold_db=float(args.exp_threshold_db),
        exp_ratio=float(args.exp_ratio),
        exp_attack_ms=float(args.exp_attack_ms),
        exp_release_ms=float(args.exp_release_ms),
        hpss=bool(args.hpss),
        hpss_start_hz=float(args.hpss_start_hz),
        hpss_end_hz=float(args.hpss_end_hz),
        hpss_time_frames=int(args.hpss_time_frames),
        hpss_freq_bins=int(args.hpss_freq_bins),
        hpss_harmonic_only=(not bool(args.hpss_no_harmonic_only)),
        phase_blur=float(args.phase_blur),
        pb_start_hz=float(args.pb_start_hz),
        pb_end_hz=float(args.pb_end_hz),
        pb_harmonic_only=(not bool(args.pb_no_harmonic_only)),
        hf_resynth=bool(args.hf_resynth),
        hf_lp_hz=float(args.hf_lp_hz),
        hf_src_lo_hz=float(args.hf_src_lo_hz),
        hf_src_hi_hz=float(args.hf_src_hi_hz),
        hf_drive=float(args.hf_drive),
        hf_hp_hz=float(args.hf_hp_hz),
        hf_mix=float(args.hf_mix),
        hf_tilt_lp_hz=float(args.hf_tilt_lp_hz),
        hf_zero_phase=(not bool(args.no_hf_zero_phase)),
    )

    target_lufs = None if float(args.target_lufs) >= 998.0 else float(args.target_lufs)
    mp = MasterParams(
        enabled=bool(args.master),
        hp_hz=float(args.hp_hz),
        target_lufs=target_lufs,
        target_rms_dbfs=float(args.target_rms_dbfs) if args.target_rms_dbfs is not None else None,
        norm_max_gain_db=float(args.norm_max_gain_db),
        norm_max_atten_db=float(args.norm_max_atten_db),
        ceiling_dbtp=float(args.ceiling_dbtp),
        lookahead_ms=float(args.lim_lookahead_ms),
        release_ms=float(args.lim_release_ms),
        os_factor=int(args.tp_os),
    )

    dp = DebugParams(
        enabled=bool(args.debug),
        debug_dir=args.debug_dir,
        stride=int(args.debug_stride),
        spec_n_fft=int(args.debug_spec_nfft),
        spec_hop=int(args.debug_spec_hop),
        spec_max_frames=int(args.debug_spec_max_frames),
        spec_max_hz=float(args.debug_spec_max_hz),
    )

    summary: Dict[str, Any] = {
        "sr": sr,
        "channels": x.shape[1],
        "duration_s": float(x.shape[0] / sr),
        "params": asdict(p),
        "master_params": asdict(mp),
        "debug_params": asdict(dp),
    }
    meas_in = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(x)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(x, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(x)),
        "lufs": measure_lufs(x, sr),
    }
    summary["measure_in"] = meas_in

    dbg_collector = None
    if dp.enabled:
        n_fft = p.n_fft
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        nyq = float(freqs[-1])

        def _idx(lo_hz: float, hi_hz: float) -> np.ndarray:
            lo = float(max(0.0, lo_hz))
            hi = float(min(nyq, hi_hz))
            if hi <= lo:
                return np.array([], dtype=np.int64)
            return np.where((freqs >= lo) & (freqs <= hi))[0]

        sh_idx = _idx(p.start_hz, p.end_hz)
        dn_idx = _idx(p.dn_start_hz, p.dn_end_hz)
        deq_idx = _idx(p.deq_start_hz, p.deq_end_hz)

        pad_offset = n_fft if p.pad else 0
        dbg_collector = DebugCollector(
            sr=sr,
            freqs=freqs,
            dn_idx=dn_idx,
            deq_idx=deq_idx,
            sh_idx=sh_idx,
            stride=dp.stride,
            pad_offset=pad_offset,
            duration_s=float(x.shape[0] / sr),
        )

    # ---- Core processing ----
    y_repaired = process_stft(x, sr, replace(p, delta_listen=False), dbg=dbg_collector, diagnostics=summary)
    y_rep_2d = _as_2d(y_repaired)

    meas_rep = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(y_rep_2d)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(y_rep_2d, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(y_rep_2d)),
        "lufs": measure_lufs(y_rep_2d, sr),
    }
    summary["measure_after_repair"] = meas_rep

    # ---- Mastering ----
    y_out, master_info = master_post(y_repaired, sr, mp, debug=dp.enabled)
    y_out_2d = _as_2d(y_out)

    meas_out = {
        "sample_peak_dbfs": float(_lin_to_db(np.max(np.abs(y_out_2d)) + 1e-12)),
        "true_peak_dbtp": float(measure_true_peak_db(y_out_2d, os_factor=max(1, int(args.tp_os)))),
        "rms_dbfs": float(measure_rms_dbfs(y_out_2d)),
        "lufs": measure_lufs(y_out_2d, sr),
    }
    summary["measure_out"] = meas_out
    summary["master_info"] = master_info

    # ---- Output writing ----
    if p.delta_listen:
        y_out_2d = x - y_out_2d

    subtype = str(args.subtype)
    sf.write(args.output, y_out_2d.astype(np.float32, copy=False), sr, subtype=subtype)

    if args.write_diff:
        diff = (x[: y_out_2d.shape[0], :] - y_out_2d).astype(np.float32)
        sf.write(args.write_diff, diff, sr, subtype=subtype)

    # ---- Debug output generation ----
    if dp.enabled and dbg_collector is not None:
        dbg_data = dbg_collector.finalize()

        def top_freqs(f: np.ndarray, A: np.ndarray, n: int = 8) -> List[Dict[str, float]]:
            if A.size == 0:
                return []
            mean_att = np.mean(A, axis=1)
            idx = np.argsort(mean_att)[::-1][:n]
            out = []
            for i in idx:
                out.append({"hz": float(f[i]), "mean_att_db": float(mean_att[i])})
            return out

        tops: Dict[str, Any] = {}
        if "att_sh_db" in dbg_data:
            tops["shimmer_top_bins"] = top_freqs(dbg_data["f_sh"], dbg_data["att_sh_db"])
        if "att_deq_db" in dbg_data:
            tops["deres_top_bins"] = top_freqs(dbg_data["f_deq"], dbg_data["att_deq_db"])
        if "att_dn_db" in dbg_data:
            tops["denoise_top_bins"] = top_freqs(dbg_data["f_dn"], dbg_data["att_dn_db"])

        summary["top_problem_bins"] = tops

        dbg_dir = dp.debug_dir
        if dbg_dir is None or dbg_dir.strip() == "":
            base = os.path.splitext(os.path.basename(args.output))[0]
            dbg_dir = os.path.join(os.path.dirname(args.output) or ".", f"{base}_debug")
        render_debug(
            dbg_dir=dbg_dir,
            dbg_data=dbg_data,
            x_in=x,
            y_repaired=_as_2d(y_repaired),
            y_out=_as_2d(y_out),
            sr=sr,
            dp=dp,
            summary=summary,
        )

        print("=== Debug summary ===")
        print(json.dumps({
            "measure_in": meas_in,
            "measure_after_repair": meas_rep,
            "measure_out": meas_out,
            "debug_dir": dbg_dir,
        }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
