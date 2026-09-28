"""High-frequency stereo decorrelation stage."""

from __future__ import annotations

from dataclasses import dataclass
from ..context import DSPContext
from ..hpss import _apply_hf_decorrelation
from ..params import Params
from .base import DSPStage


@dataclass
class HFDecorrelationConfig:
    strength: float = 0.0
    start_hz: float = 4500.0
    end_hz: float = 16000.0

    @classmethod
    def from_params(cls, p: Params) -> "HFDecorrelationConfig":
        return cls(
            strength=float(getattr(p, "hf_decorrelate", 0.0)),
            start_hz=float(getattr(p, "hf_dec_start_hz", 4500.0)),
            end_hz=float(getattr(p, "hf_dec_end_hz", 16000.0)),
        )


class HFDecorrelationStage(DSPStage):
    name = "hf_decorrelation"

    def __init__(self, config: HFDecorrelationConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "HFDecorrelationStage":
        return cls(HFDecorrelationConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.strength > 1e-6

    def process(self, ctx: DSPContext) -> None:
        hdc_idx = ctx.idx(self.config.start_hz, self.config.end_hz)
        hdc_amt = float(self.config.strength)
        if hdc_amt <= 1e-6 or hdc_idx.size == 0 or ctx.n_ch < 2:
            return

        _apply_hf_decorrelation(ctx.Z, hdc_idx, amount=hdc_amt, rng=ctx.rng, eps=ctx.eps)
