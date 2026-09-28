"""Deshimmer - Audio artifact reconstruction and spectral repair toolkit."""

__version__ = "0.1.0"

from .api import preview_context_seconds, process_audio, slice_with_context
from .autotune import AnalysisReport, Region, Weights, analyze, refine
from .core import DebugParams, MasterParams, Params
from .ui import build_ui

__all__ = [
    "AnalysisReport",
    "DebugParams",
    "MasterParams",
    "Params",
    "Region",
    "Weights",
    "__version__",
    "analyze",
    "build_ui",
    "preview_context_seconds",
    "process_audio",
    "refine",
    "slice_with_context",
]
