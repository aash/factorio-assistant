"""Scene visibility management for the overlay.

Provides functions to show/hide overlay scenes and track their state
through the LeafState dataclass.
"""

from __future__ import annotations

import logging
from overlay import OverlayClient
from leaf.state import LeafState


def set_scene_visible(ov: OverlayClient, leaf_state: LeafState, name: str, visible: bool) -> bool:
    """Set a scene's visibility in both the overlay and LeafState.

    Always sends the command to the overlay. Returns True.
    """
    leaf_state.scene_visibility[name] = visible
    if visible:
        ov.show_scene(name)
    else:
        ov.hide_scene(name)
    return True


def hide_map_scenes(ov: OverlayClient, leaf_state: LeafState) -> None:
    """Hide all map-related overlay scenes."""
    set_scene_visible(ov, leaf_state, "map_composite_image", False)
    set_scene_visible(ov, leaf_state, "map_composite", False)
    set_scene_visible(ov, leaf_state, "map_node_mouse_hover", False)
    set_scene_visible(ov, leaf_state, "map_character_marker", False)
    # Also clear the last-node marker
    _destroy_last_node_marker(ov)
    leaf_state.map_composite_scene_origin = None


def _destroy_last_node_marker(ov: OverlayClient) -> None:
    """Destroy the last-node marker scene if it exists."""
    try:
        ov.destroy_scene("map_last_node_marker")
    except Exception:
        pass
