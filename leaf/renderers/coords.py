"""Coordinate system helpers for the map overlay.

These are pure functions that translate between game-map coordinates
and screen-pixel coordinates used by the overlay client.
"""

from __future__ import annotations

import numpy as np
from graphics import Rect
from map_graph.models import MapEdge

# Map anchor offsets (in screen pixels) for where the map origin is drawn
MAP_ANCHOR_X = 40
MAP_ANCHOR_Y = 140


def map_coord_to_screen(
    coord: tuple[int, int],
    origin_x: int,
    origin_y: int,
    min_x: int,
    min_y: int,
    tile_w: int,
    tile_h: int,
) -> tuple[int, int]:
    """Convert a game-map (dx, dy) coordinate to overlay screen pixel position."""
    dx, dy = coord
    return (
        origin_x + (dx - min_x) + tile_w // 2,
        origin_y + (dy - min_y) + tile_h // 2,
    )


def screen_to_map_coord(
    x: int,
    y: int,
    origin_x: int,
    origin_y: int,
    min_x: int,
    min_y: int,
    tile_w: int,
    tile_h: int,
) -> tuple[int, int]:
    """Convert a screen pixel position to a game-map (dx, dy) coordinate."""
    dx = int(round((x - origin_x) - (tile_w // 2) + min_x))
    dy = int(round((y - origin_y) - (tile_h // 2) + min_y))
    return dx, dy


def map_scene_geometry(
    map_offsets: list[tuple[int, int]],
    tile_size: int,
) -> tuple[int, int, int, int, int, int]:
    """Compute (origin_x, origin_y, min_x, min_y, tile_w, tile_h).

    Parameters
    ----------
    map_offsets : list[tuple[int, int]]
        (dx, dy) offsets of each map tile.
    tile_size : int
        Width/height of a single map tile in pixels (square).
    """
    min_x = min(dx for dx, _ in map_offsets) if map_offsets else 0
    min_y = min(dy for _, dy in map_offsets) if map_offsets else 0
    origin_x = 1920 // 2 + MAP_ANCHOR_X + min_x
    origin_y = (1080 * 2) // 2 + MAP_ANCHOR_Y + min_y
    tile_w = tile_size
    tile_h = tile_size
    return origin_x, origin_y, min_x, min_y, tile_w, tile_h


def canonical_edge_key(edge: MapEdge) -> tuple[str, str]:
    """Return a canonical, order-independent key for an edge (sorted by uid)."""
    from_uid = str(edge.from_uid)
    to_uid = str(edge.to_uid)
    if from_uid <= to_uid:
        return from_uid, to_uid
    return to_uid, from_uid


def sorted_line_coords(
    x1: int, y1: int, x2: int, y2: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((x1, y1), (x2, y2)) sorted so the smaller point comes first."""
    if (x1, y1) <= (x2, y2):
        return (x1, y1), (x2, y2)
    return (x2, y2), (x1, y1)


def center_square_rect(window_rect: Rect, size_px: int) -> Rect:
    """Return a Rect of size_px×size_px centered on the window rect."""
    dims = np.array([size_px, size_px])
    cent = window_rect.wh() // 2
    return Rect.from_centdims(*cent, *dims)
