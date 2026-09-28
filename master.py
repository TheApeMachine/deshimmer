"""Compatibility shim forwarding to deshimmer.core and deshimmer.cli."""

import sys
from deshimmer.cli import main
from deshimmer.core import *
import deshimmer.core as _core

_LAST_ARTIFACT_CONF_FRAMES = None
__all__ = _core.__all__ + ["main", "_LAST_ARTIFACT_CONF_FRAMES"]

if __name__ == "__main__":

    raise SystemExit(main())
