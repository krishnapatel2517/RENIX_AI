"""
RENIX 3D Engine
===============

Standard-compliant module alias for holographic_ui.3d_engine.
Allows standard `from holographic_ui import engine_3d` and
`from holographic_ui.engine_3d import Engine3D`.
"""

from __future__ import annotations
import importlib

# Dynamically import the implementation from 3d_engine
_underlying = importlib.import_module(".3d_engine", package=__package__)

for _attr in dir(_underlying):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_underlying, _attr)

__all__ = [attr for attr in dir(_underlying) if not attr.startswith("__")]
