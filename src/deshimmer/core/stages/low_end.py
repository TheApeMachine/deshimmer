"""Low-end coherence, mono bass, and boom containment stage."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..context import DSPContext
from ..params import Params
from .base import DSPStage


@dataclass
class LowEndConfig:
    bass_mono: bool = False
    mono_hz: float = 120.0
    tighten: float = 0.0
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "LowEndConfig":
        return cls(
            bass_mono=bool(getattr(p, "bass_mono", False)),
            mono_hz=float(getattr(p, "bass_mono_hz", 120.0)),
            tighten=float(getattr(p, "bass_tighten", 0.0)),
            enhance=bool(getattr(p, "enhance", True)),
        )


class LowEndStage(DSPStage):
    name = "low_end"

    def __init__(self, config: LowEndConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "LowEndStage":
        return cls(LowEndConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enhance and (self.config.bass_mono or self.config.tighten > 1e-6)

    def process(self, ctx: DSPContext) -> None:
        if not self.is_enabled():
            return

        freqs = ctx.freqs
        mono_hz = max(20.0, float(self.config.mono_hz))
        low_idx = np.where(freqs <= mono_hz * 1.5)[0]

        if low_idx.size == 0:
            return

        if self.config.bass_mono and ctx.n_ch >= 2:
            left = ctx.Z[low_idx, :, 0]
            right = ctx.Z[low_idx, :, 1]
            mid = 0.5 * (left + right)
            side = 0.5 * (left - right)

            f_sub = freqs[low_idx]
            side_scale = np.clip(f_sub / mono_hz, 0.0, 1.0) ** 2
            side_new = side * side_scale[:, None]

            ctx.Z[low_idx, :, 0] = mid + side_new
            ctx.Z[low_idx, :, 1] = mid - side_new

        if self.config.tighten > 1e-6:
            tight_str = float(np.clip(self.config.tighten, 0.0, 1.0))
            boom_idx = np.where((freqs >= 40.0) & (freqs <= 180.0))[0]
            if boom_idx.size > 0:
                # Distinguish undesirable boom overhang from sustained musical bass:
                # 1. Harmonic presence: Pitched musical notes (synth/bass guitar) possess rich 2nd/3rd harmonics.
                # 2. Temporal decay tracking: An unwanted resonance hangs over (fails to decay) during quiet
                #    decay frames where broadband energy has dropped, whereas pure sine bass notes
                #    decay naturally with the musical envelope.
                mag_mono = ctx.Mag_mono
                harmonic_mask = np.zeros(len(boom_idx), dtype=np.float32)
                for bi, b_k in enumerate(boom_idx):
                    harm2_k = int(np.round(2 * b_k))
                    if harm2_k < len(freqs):
                        ratio = float(np.mean(mag_mono[harm2_k, :]) / (np.mean(mag_mono[b_k, :]) + ctx.eps))
                        harmonic_mask[bi] = float(np.clip(ratio * 2.5, 0.0, 1.0))

                # Track post-transient decay tail:
                # An unwanted boom rings out during the decay tail following a transient hit,
                # whereas steady, sustained pure sine bass notes have no transient decay tail.
                decay_env = np.zeros_like(ctx.w_trans)
                curr_decay = 0.0
                for t_col in range(ctx.n_cols):
                    if ctx.w_trans[t_col] > 0.3:
                        curr_decay = float(ctx.w_trans[t_col])
                    else:
                        curr_decay *= 0.85
                    decay_env[t_col] = curr_decay
                overhang_tail = np.clip(decay_env - ctx.w_trans, 0.0, 1.0)[None, :]  # (1, T)

                isolated_boom = (1.0 - harmonic_mask)[:, None] * overhang_tail
                w_decay = ctx.w_nontrans[None, :]
                overhang_attenuation = 1.0 - 0.40 * tight_str * (isolated_boom * w_decay)
                att_factor = np.clip(overhang_attenuation, 0.55, 1.0).astype(np.float32)
                ctx.Z[boom_idx, :, :] *= att_factor[:, :, None]

        ctx.Mag = np.abs(ctx.Z)
        ctx.Mag_mono = np.mean(ctx.Mag, axis=2)
        ctx.PSD = ctx.Mag_mono ** 2 + ctx.eps
