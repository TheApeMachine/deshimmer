"""Compatibility shim forwarding to deshimmer.tonal.repair."""

from deshimmer.tonal.repair import *
import deshimmer.tonal.repair as _repair

__all__ = ["TonalRepair"]
