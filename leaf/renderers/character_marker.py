"""Character position marker overlay.

Draws a blue double-ring and coordinate label at the player's
current map position.
"""

from __future__ import annotations

import time
from typing import Optional

from overlay import OverlayClient
from leaf.state import LeafState
from leaf.renderers.scene_visibility import set_scene_visible
from leaf.renderers.coords import map_coord_to_screen, map_scene_geometry

CHARACTER_MARKER_FPS = 60
CHARACTER_MARKER_Z = 998


def draw_character_marker(
    ov: OverlayClient,
    leaf_state: LeafState,
    track_character_coord: bool,
    character_marker_coord: Optional[list[int]],
    map_tiles: list,
    map_offsets: list[tuple[int, int]],
    force: bool = False,
) -> None:
    """Draw or hide the character position marker on the map overlay.

    Caller is responsible for throttling; this function checks
    ``leaf_state.character_marker_next_update_ts`` internally.
    """
    if not track_character_coord or character_marker_coord is None:
        set_scene_visible(ov, leaf_state, "map_character_marker", False)
        return

    now = time.perf_counter()
    if not force and now < leaf_state.character_marker_next_update_ts:
        return
    leaf_state.character_marker_next_update_ts = now + (1.0 / CHARACTER_MARKER_FPS)

    if not map_tiles or not map_offsets:
        return

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = map_scene_geometry(map_offsets, map_tiles)
    cx, cy = map_coord_to_screen(character_marker_coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)

    set_scene_visible(ov, leaf_state, "map_character_marker", True)
    with ov.scene("map_character_marker") as s:
        try:
            s.set_z(CHARACTER_MARKER_Z)
        except Exception:
            pass
        ccx, ccy = character_marker_coord
        s.text(
            cx + 20,
            cy,
            f"{ccx},{ccy}",
            color=(220, 220, 220, 255),
            font="JetBrainsMono NFM",
            size=8,
        )
        s.ellipse(
            cx - 10,
            cy - 10,
            20,
            20,
            pen_color=(0, 180, 255, 255),
            pen_width=2,
            brush_color=(0, 180, 255, 48),
        )
        s.ellipse(
            cx - 5,
            cy - 5,
            10,
            10,
            pen_color=(255, 255, 255, 255),
            pen_width=1,
            brush_color=(255, 255, 255, 96),
        )
