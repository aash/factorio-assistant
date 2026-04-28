"""Map composite overlay — stitched tile image, edges, crosshair markers.

Renders the map-graph composite image, accepted edges between
nodes, crosshair marks at each tile coordinate, and the
"last added node" highlight circle.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from overlay import OverlayClient
from map_graph import MapGraphBuilder
from leaf.state import LeafState
from leaf.renderers.scene_visibility import set_scene_visible
from leaf.renderers.coords import (
    map_coord_to_screen,
    map_scene_geometry,
    canonical_edge_key,
    sorted_line_coords,
)

EDGE_COLOR = (0, 255, 0, 120)
CROSSHAIR_COLOR = (255, 0, 0, 180)
CROSSHAIR_SIZE = 7
CROSSHAIR_WIDTH = 3
COORD_TEXT_COLOR = (255, 230, 120, 255)
COORD_FONT = "JetBrainsMono NFM"
COORD_FONT_SIZE = 9
LAST_NODE_Z = 999
CHAR_MARKER_Z = 998


def draw_map_composite(
    ov: OverlayClient,
    leaf_state: LeafState,
    map_tiles: list[np.ndarray],
    map_offsets: list[tuple[int, int]],
    map_composite: Optional[np.ndarray],
    map_composite_pngbytes: Optional[bytes],
    map_graph_builder: Optional[MapGraphBuilder],
    last_node_marker_active: bool,
    last_node_marker_uid: Optional[str],
) -> None:
    """Render the map composite image, edges, crosshairs, and last-node marker.

    Call this on every frame (it is cheap when scenes use delta updates).
    """
    # Sync leaf_state tracking from module globals (updated by action handlers)
    leaf_state.last_node_marker_active = last_node_marker_active
    leaf_state.last_node_marker_uid = last_node_marker_uid

    if map_composite is None:
        set_scene_visible(ov, leaf_state, "map_composite_image", False)
        set_scene_visible(ov, leaf_state, "map_composite", False)
        return

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = map_scene_geometry(map_offsets, map_tiles)
    map_shape = map_composite.shape

    # Determine last-node screen position
    last_node_screen = None
    last_node_uid: str | None = None
    if map_graph_builder is not None:
        graph = map_graph_builder.graph
        if graph.last_uid is not None:
            last = graph.nodes.get(graph.last_uid)
            if last is not None and last.coord is not None:
                last_node_screen = map_coord_to_screen(
                    last.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h
                )
                last_node_uid = graph.last_uid

    # Set Z-ordering
    try:
        ov.set_scene_z("map_composite_image", 0)
        ov.set_scene_z("map_composite", 10)
        ov.set_scene_z("map_character_marker", CHAR_MARKER_Z)
    except Exception:
        pass

    # Composite image layer
    with ov.scene_delta("map_composite_image") as s_img:
        if map_composite_pngbytes is not None:
            s_img.image(
                origin_x,
                origin_y,
                map_shape[1],
                map_shape[0],
                png_bytes=map_composite_pngbytes,
            )

    # Edges and crosshair layer
    with ov.scene_delta("map_composite") as s:
        if map_graph_builder is not None:
            edge_pairs: set[tuple[str, str]] = {
                canonical_edge_key(edge)
                for edge in map_graph_builder.graph.edges
                if edge.accepted
            }
            for from_uid, to_uid in sorted(edge_pairs):
                from_node = map_graph_builder.graph.nodes.get(from_uid)
                to_node = map_graph_builder.graph.nodes.get(to_uid)
                if from_node is None or to_node is None:
                    continue
                if from_node.coord is None or to_node.coord is None:
                    continue
                x1, y1 = map_coord_to_screen(from_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                x2, y2 = map_coord_to_screen(to_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                (nx1, ny1), (nx2, ny2) = sorted_line_coords(x1, y1, x2, y2)
                s.line(nx1, ny1, nx2, ny2, color=EDGE_COLOR, width=1)

        # Crosshair + coordinate labels at each tile
        for dx, dy in map_offsets:
            cx, cy = map_coord_to_screen((dx, dy), origin_x, origin_y, min_x, min_y, tile_w, tile_h)
            (lx1, ly1), (lx2, ly2) = sorted_line_coords(cx - CROSSHAIR_SIZE, cy, cx + CROSSHAIR_SIZE, cy)
            s.line(lx1, ly1, lx2, ly2, color=CROSSHAIR_COLOR, width=CROSSHAIR_WIDTH)
            (lyx1, lyy1), (lyx2, lyy2) = sorted_line_coords(cx, cy - CROSSHAIR_SIZE, cx, cy + CROSSHAIR_SIZE)
            s.line(lyx1, lyy1, lyx2, lyy2, color=CROSSHAIR_COLOR, width=CROSSHAIR_WIDTH)
            s.text(
                cx - 12 + 10,
                cy + 4 - 10,
                f"{dx},{dy}",
                color=COORD_TEXT_COLOR,
                font=COORD_FONT,
                size=COORD_FONT_SIZE,
            )

    # Last-node highlight
    if last_node_screen is not None and last_node_uid is not None:
        # if last_node_uid != leaf_state.last_node_marker_uid:
        draw_last_node_marker(ov, leaf_state, last_node_screen, last_node_uid)
    else:
        clear_last_node_marker(ov, leaf_state)


def draw_last_node_marker(
    ov: OverlayClient,
    leaf_state: LeafState,
    screen_coord: tuple[int, int],
    node_uid: str,
) -> None:
    """Draw a gold double-ring around the most recently added map node."""
    # Clear existing marker if it targets a different node
    if leaf_state.last_node_marker_active and leaf_state.last_node_marker_uid != node_uid:
        logging.warning(
            "last node marker scene was still active; clearing (prev=%s new=%s)",
            leaf_state.last_node_marker_uid,
            node_uid,
        )
        try:
            ov.destroy_scene("map_last_node_marker")
        except Exception as exc:
            logging.warning("failed to destroy existing last node marker: %s", exc)
    elif leaf_state.last_node_marker_active and leaf_state.last_node_marker_uid == node_uid:
        return

    cx, cy = screen_coord
    with ov.scene("map_last_node_marker") as s:
        try:
            s.set_z(LAST_NODE_Z)
        except Exception:
            pass
        s.ellipse(
            cx - 12,
            cy - 12,
            24,
            24,
            pen_color=(255, 220, 0, 255),
            pen_width=2,
            brush_color=(0, 0, 0, 0),
        )
        s.ellipse(
            cx - 4,
            cy - 4,
            8,
            8,
            pen_color=(255, 220, 0, 255),
            pen_width=1,
            brush_color=(255, 220, 0, 96),
        )

    leaf_state.last_node_marker_active = True
    leaf_state.last_node_marker_uid = node_uid


def clear_last_node_marker(ov: OverlayClient, leaf_state: LeafState) -> None:
    """Destroy the last-node marker scene and reset tracking state."""
    if leaf_state.last_node_marker_active:
        try:
            ov.destroy_scene("map_last_node_marker")
        except Exception as exc:
            logging.warning("failed to destroy last node marker: %s", exc)
        leaf_state.last_node_marker_active = False
        leaf_state.last_node_marker_uid = None
