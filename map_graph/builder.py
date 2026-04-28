from __future__ import annotations

from .service import CoordValidationResult, MapCaptureResult, MapGraphService


class MapGraphBuilder(MapGraphService):
    """Backward-compatible name used by existing assistant/snail code."""

    pass


__all__ = [
    "MapGraphBuilder",
    "MapCaptureResult",
    "CoordValidationResult",
]
