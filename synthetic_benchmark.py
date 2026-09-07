#!/usr/bin/env python3
"""
synthetic_benchmark.py

Inject known AI-ish artifact families into clean audio and score repair
pipelines against full-reference and no-reference metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import master as _m
from auto_tune import score_processed, weights_from_aggressiveness
from deshimmer_api import process_audio


@dataclass
class DegradationSpec:
    name: str
    apply: Callable[[np.ndarray, int], np.ndarray]


@dataclass
class BenchmarkResult:
    degradation: str
    score_total: float
    score_breakdown: dict[str, float]
    visqol_proxy_db: float
    notes: list[str] = field(default_factory=list)


def _to_stereo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[:, None]
    return x


def _inject_narrow_birdies(x: np.ndarray, sr: int, *, n_birdies: int = 4) -> np.ndarray:
    y = _to_stereo(x).copy()
    n = y.shape[0]
    t = np.arange(n, dtype=np.float32) / float(sr)
    freqs = np.linspace(4200.0, 7800.0, n_birdies)
    for i, f0 in enumerate(freqs):
        drift = f0 * (1.0 + 0.02 * np.sin(2.0 * math.pi * (0.3 + 0.1 * i) * t))
        phase = 2.0 * math.pi * np.cumsum(drift / float(sr))
        amp = 0.012 * (1.0 + 0.5 * np.sin(2.0 * math.pi * 0.7 * t))
        y[:, 0] += amp * np.sin(phase).astype(np.float32)
    return y


def _inject_persistent_whine(x: np.ndarray, sr: int, *, hz: float = 3550.0) -> np.ndarray:
    y = _to_stereo(x).copy()
    n = y.shape[0]
    t = np.arange(n, dtype=np.float32) / float(sr)
    tone = 0.018 * np.sin(2.0 * math.pi * hz * t).astype(np.float32)
    y += tone[:, None]
    return y


def _inject_hf_crickets(x: np.ndarray, sr: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    y = _to_stereo(x).copy()
    n = y.shape[0]
    noise = rng.standard_normal(n).astype(np.float32)
    # Band-limit-ish crickets via simple modulation
    mod = 0.5 + 0.5 * np.sin(2.0 * math.pi * 12.0 * np.arange(n, dtype=np.float32) / float(sr))
    cr = noise * mod * 0.008
    y[:, 0] += cr
    return y


def _inject_swish_tail(x: np.ndarray, sr: int) -> np.ndarray:
    y = _to_stereo(x).copy()
    n = y.shape[0]
    rng = np.random.default_rng(1)
    env = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2
    swish = rng.standard_normal(n).astype(np.float32) * env * 0.006
    y[:, 0] += swish
    return y


def default_degradations() -> list[DegradationSpec]:
    return [
        DegradationSpec("narrow_birdies", lambda x, sr: _inject_narrow_birdies(x, sr)),
        DegradationSpec("persistent_whine_3p5k", lambda x, sr: _inject_persistent_whine(x, sr)),
        DegradationSpec("hf_crickets", lambda x, sr: _inject_hf_crickets(x, sr)),
        DegradationSpec("swish_tail", lambda x, sr: _inject_swish_tail(x, sr)),
    ]


def _visqol_proxy_db(clean: np.ndarray, test: np.ndarray, sr: int) -> float:
    """Lightweight spectral-distance proxy (not ViSQOL). Higher = closer to clean."""
    from scipy.signal import stft as scipy_stft

    def mono(a: np.ndarray) -> np.ndarray:
        a2 = _to_stereo(a)
        return np.mean(a2, axis=1).astype(np.float32)

    c, t = mono(clean), mono(test)
    m = min(c.size, t.size)
    c, t = c[:m], t[:m]
    _, _, Zc = scipy_stft(c, fs=sr, nperseg=2048, noverlap=1536)
    _, _, Zt = scipy_stft(t, fs=sr, nperseg=2048, noverlap=1536)
    mc, mt = np.abs(Zc), np.abs(Zt)
    T = min(mc.shape[1], mt.shape[1])
    diff = np.mean((20.0 * np.log10(mc[:, :T] + 1e-9) - 20.0 * np.log10(mt[:, :T] + 1e-9)) ** 2)
    return float(-math.sqrt(diff))


def run_benchmark(
    clean: np.ndarray,
    sr: int,
    *,
    params: Optional[_m.Params] = None,
    master_params: Optional[_m.MasterParams] = None,
    degradations: Optional[list[DegradationSpec]] = None,
    aggressiveness: float = 0.5,
) -> list[BenchmarkResult]:
    p = params if params is not None else _m.Params()
    mp = master_params if master_params is not None else _m.MasterParams(enabled=False)
    dp = _m.DebugParams(enabled=False)
    weights = weights_from_aggressiveness(aggressiveness)
    band = (float(p.start_hz), float(p.end_hz))
    specs = degradations if degradations is not None else default_degradations()
    out: list[BenchmarkResult] = []

    for spec in specs:
        dirty = spec.apply(clean, sr)
        try:
            repaired, _info = process_audio(dirty, sr, params=p, master_params=mp, debug_params=dp)
        except Exception as exc:
            out.append(BenchmarkResult(spec.name, -999.0, {}, 0.0, notes=[str(exc)]))
            continue
        sc = score_processed(
            clean, repaired, sr, band=band,
            swish_band=(p.swish_start_hz, p.swish_end_hz), weights=weights,
        )
        proxy = _visqol_proxy_db(clean, repaired, sr)
        out.append(BenchmarkResult(
            degradation=spec.name,
            score_total=float(sc.get("total", 0.0)),
            score_breakdown={k: float(v) for k, v in sc.items() if isinstance(v, (int, float))},
            visqol_proxy_db=proxy,
        ))
    return out


def benchmark_markdown(results: list[BenchmarkResult]) -> str:
    lines = ["**Synthetic benchmark**", "", "| Degradation | Score | ViSQOL proxy (dB) |", "|---|---:|---:|"]
    for r in results:
        lines.append(f"| {r.degradation} | {r.score_total:+.2f} | {r.visqol_proxy_db:+.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser(description="Run synthetic artifact benchmark on a clean WAV.")
    ap.add_argument("input")
    ap.add_argument("--sr", type=int, default=0, help="Override sample rate")
    args = ap.parse_args()
    x, sr = sf.read(args.input, always_2d=True)
    if args.sr > 0:
        sr = int(args.sr)
    res = run_benchmark(x.astype(np.float32), int(sr))
    print(benchmark_markdown(res))
