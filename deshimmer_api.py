"""Compatibility shim forwarding to deshimmer.api."""

from deshimmer.api import *
import deshimmer.api as _api

__all__ = [
    "preview_context_seconds",
    "slice_with_context",
    "process_audio",
]
