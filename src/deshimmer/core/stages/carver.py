"""Dynamic spectral carver stage."""

from __future__ import annotations

from dataclasses import dataclass
from ..context import DSPContext
from ..params import Params
from ..spatial import dynamic_spectral_carver
from .base import DSPStage


@dataclass
class CarverConfig:
    enhance: bool = True

    @classmethod
    def from_params(cls, p: Params) -> "CarverConfig":
        return cls(enhance=bool(getattr(p, "enhance", True)))


class DynamicCarverStage(DSPStage):
    name = "dynamic_carver"

    def __init__(self, config: CarverConfig):
        self.config = config

    @classmethod
    def from_params(cls, p: Params) -> "DynamicCarverStage":
        return cls(CarverConfig.from_params(p))

    def is_enabled(self) -> bool:
        return self.config.enhance

    def process(self, ctx: DSPContext) -> None:
        if self.config.enhance:
            ctx.Z = dynamic_spectral_carver(ctx.Z, ctx.freqs)
