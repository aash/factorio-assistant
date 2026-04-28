"""Map node hover highlight overlay.

Draws a green ring around the map node nearest to the mouse cursor.
"""

from __future__ import annotations

import time
from typing import Optional

from overlay import OverlayClient
from graphics import Rect
from leaf.state import LeafState
from leaf.renderers.scene_visibility import set_scene_visible
from leaf.renderers.coords import map_coord_to_screen, map_scene_geometry
from map_graph import MapGraphBuilder

HOVER_FPS = 15
HOVER_RADIUS_PX = 18.0
HOVER_ELLIPSE_RADIUS = 14


def get_hovered_map_node(
    ahk,
    map_graph_builder: Optional[MapGraphBuilder],
    map_offsets: list[tuple[int, int]],
    map_tiles: list,
) -> object:
    """Return the MapNode closest to the mouse cursor, or None."""
    if map_graph_builder is None:
        return None
    graph = map_graph_builder.graph
    if graph is None or not map_tiles:
        return None

    try:
        mouse_pos = ahk.get_mouse_position(coord_mode="Screen")
    except Exception:
        return None
    if mouse_pos is None:
        return None

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = map_scene_geometry(map_offsets, map_tiles)
    mouse_x, mouse_y = int(mouse_pos.x), int(mouse_pos.y)

    hovered_node = None
    hovered_dist = HOVER_RADIUS_PX
    for node in graph.nodes.values():
        if node.coord is None:
            continue
        node_x, node_y = map_coord_to_screen(node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
        dist = ((mouse_x - node_x) ** 2 + (mouse_y - node_y) ** 2) ** 0.5
        if dist <= hovered_dist:
            hovered_dist = dist
            hovered_node = node
    return hovered_node


def draw_map_node_mouse_hover(
    ov: OverlayClient,
    leaf_state: LeafState,
    ahk,
    map_graph_builder: Optional[MapGraphBuilder],
    map_offsets: list[tuple[int, int]],
    map_tiles: list,
    force: bool = False,
) -> None:
    """Draw a green ring around the hovered map node, if any."""
    if not leaf_state.show_map_node_hover:
        set_scene_visible(ov, leaf_state, "map_node_mouse_hover", False)
        return

    now = time.perf_counter()
    if not force and now < leaf_state.map_node_hover_next_update_ts:
        return
    leaf_state.map_node_hover_next_update_ts = now + (1.0 / HOVER_FPS)

    hovered = get_hovered_map_node(ahk, map_graph_builder, map_offsets, map_tiles)
    if hovered is None:
        set_scene_visible(ov, leaf_state, "map_node_mouse_hover", False)
        return

    if not map_tiles:
        return
    origin_x, origin_y, min_x, min_y, tile_w, tile_h = map_scene_geometry(map_offsets, map_tiles)
    cx, cy = map_coord_to_screen(hovered.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)

    set_scene_visible(ov, leaf_state, "map_node_mouse_hover", True)
    with ov.scene("map_node_mouse_hover") as s:
        s.ellipse(
            cx - HOVER_ELLIPSE_RADIUS,
            cy - HOVER_ELLIPSE_RADIUS,
            HOVER_ELLIPSE_RADIUS * 2,
            HOVER_ELLIPSE_RADIUS * 2,
            pen_color=(0, 255, 0, 255),
            pen_width=2,
            brush_color=(0, 60, 0, 96),
        )
