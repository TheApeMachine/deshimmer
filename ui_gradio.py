"""Compatibility shim forwarding to deshimmer.ui."""

from deshimmer.ui import *
import deshimmer.ui as _ui

__all__ = _ui.__all__

if __name__ == "__main__":
    raise SystemExit(_ui.main())
