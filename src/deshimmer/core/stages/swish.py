"""Swish phase repair stage using adaptive phasor smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from ..context import DSPContext
from ..hpss import _apply_swish_phase_repair
from ..params import Params
from .base import DSPStage


@dataclass
class SwishRepairConfig:
    strength: float = 0.0
    start_hz: float = 3500.0
    end_hz: float = 14000.0
    time_amt: float = 0.55
    freq_amt: float = 0.30
    time_win: int = 7
    freq_win: int = 5
    transient_protect: float = 0.85
    harmonic_protect: float = 0.45
    hpss_time_frames: int = 21
    hpss_freq_bins: int = 17

    @classmethod
    def from_params(cls, p: Params) -> "SwishRepairConfig":
        return cls(
            strength=float(getattr(p, "swish_repair", 0.0)),
            start_hz=float(getattr(p, "swish_start_hz", 3500.0)),
            end_hz=float(getattr(p, "swish_end_hz", 14000.0)),
            time_amt=float(getattr(p, "swish_time_amt", 0.55)),
            freq_amt=float(getattr(p, "swish_freq_amt", 0.30)),
            time_win=int(getattr(p, "swish_time_win", 7)),
            freq_win=int(getattr(p, "swish_freq_win", 5)),
            transient_protect=float(getattr(p, "swish_transient_protect", 0.85)),
            harmonic_protect=float(getattr(p, "swish_harmonic_protect", 0.45)),
            hpss_time_frames=int(getattr(p, "hpss_time_frames", 21)),
            hpss_freq_bins=int(getattr(p, "hpss_freq_bins", 17)),
        )


class SwishRepairStage(DSPStage):
    name = "swish_repair"

    def __init__(self, config: SwishRepairConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "SwishRepairStage":
        return cls(SwishRepairConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        sw_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        swish_amt = float(self.config.strength)
        if swish_amt <= 1e-6 or sw_idx.size == 0:
            return

        _apply_swish_phase_repair(
            ctx.Z,
            sw_idx,
            strength=swish_amt,
            time_amt=self.config.time_amt,
            freq_amt=self.config.freq_amt,
            time_win=self.config.time_win,
            freq_win=self.config.freq_win,
            w_nontrans=ctx.w_nontrans,
            transient_protect=self.config.transient_protect,
            harmonic_protect=self.config.harmonic_protect,
            Mag_mono=ctx.Mag_mono,
            hpss_time_frames=self.config.hpss_time_frames,
            hpss_freq_bins=self.config.hpss_freq_bins,
            eps=ctx.eps,
        )
