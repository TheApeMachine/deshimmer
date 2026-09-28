"""Compatibility shim forwarding to deshimmer.autotune."""

from deshimmer.autotune import *
import deshimmer.autotune as _autotune

__all__ = _autotune.__all__
