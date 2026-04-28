"""LeafState — encapsulates all mutable presentation-layer state.

Tracks scene visibility, render timers, and widget flags that control
what the overlay renders and at what rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

HISTORY_MAX = 10


@dataclass
class LeafState:
    """Overlay rendering state for the leaf presentation layer."""

    # ── Scene visibility ───────────────────────────────────────────────
    scene_visibility: dict[str, bool] = field(
        default_factory=lambda: {
            "history": False,
            "input": False,
            "map_composite_image": True,
            "map_composite": True,
            "map_node_mouse_hover": False,
            "map_character_marker": False,
            "ui_brect_marks": False,
        }
    )

    # ── Widget toggles ─────────────────────────────────────────────────
    show_map_overlay: bool = True
    show_map_node_hover: bool = False
    show_ui_brect_marks: bool = False
    show_history_widget: bool = False

    # ── Render timers (throttled at N fps) ─────────────────────────────
    map_node_hover_next_update_ts: float = 0.0
    character_marker_next_update_ts: float = 0.0

    # ── History ────────────────────────────────────────────────────────
    history_queue: deque | None = field(
        default_factory=lambda: deque(maxlen=HISTORY_MAX)
    )

    # ── Map composite scene cache ───────────────────────────────────────
    last_node_marker_active: bool = False
    last_node_marker_uid: str | None = None
    map_composite_scene_origin: tuple[int, int] | None = None
    map_edge_scene_verify_next_ts: float = 0.0

    def set_scene_visible(self, name: str, visible: bool) -> bool:
        """Update scene visibility and return True if it changed."""
        changed = self.scene_visibility.get(name) != visible
        self.scene_visibility[name] = visible
        return changed

    def is_scene_visible(self, name: str) -> bool:
        return self.scene_visibility.get(name, False)
