"""Abstract base class for all DSP processing stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from ..context import DSPContext


class DSPStage(ABC):
    name: str = "dsp_stage"

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this stage is active and should run."""
        pass

    @abstractmethod
    def process(self, ctx: DSPContext) -> None:
        """Process the audio representation within the DSP context in place."""
        pass
