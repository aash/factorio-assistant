"""Leaf — presentation layer for the factorio-assistant overlay.

Manages scene visibility, rendering timers, and widget state for all
overlay UI elements.  Subscribes to SnailEventBus and
SnailEventBus events and translates state changes into
OverlayClient scene operations.
"""

from __future__ import annotations

from .state import LeafState

__all__ = [
    "LeafState",
]
