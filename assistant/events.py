"""Event name constants for the factorio-assistant event bus architecture.

Organized by domain scope:
  - snail.character.*  — character coordinate tracking
  - snail.map_graph.*  — map graph operations
  - leaf.*             — presentation layer / overlay
"""

# ── Character scope ────────────────────────────────────────────────────

SNAIL_CHARACTER_COORD_UPDATED = "snail.character.coord_updated"
"""Emitted when character marker position changes via frame-delta tracking.
Payload: coord: tuple[int, int], source: str ("frame"|"seed"|"validation")
"""

SNAIL_CHARACTER_COORD_CORRECTED = "snail.character.coord_corrected"
"""Emitted when graph validation corrects the character coordinate.
Payload: coord: tuple[int, int], drift: float
"""

SNAIL_CHARACTER_TRACKING_ENABLED = "snail.character.tracking_enabled"
"""Emitted when character tracking is started and seed succeeds.
Payload: coord: tuple[int, int]
"""

SNAIL_CHARACTER_TRACKING_DISABLED = "snail.character.tracking_disabled"
"""Emitted when character tracking is turned off.
Payload: (none)
"""

SNAIL_CHARACTER_SEED_FAILED = "snail.character.seed_failed"
"""Emitted when initial character marker seed could not be found.
Payload: (none)
"""

# ── Map graph scope ────────────────────────────────────────────────────

SNAIL_MAP_GRAPH_NODE_ADDED = "snail.map_graph.node_added"
"""Emitted when a new map tile is captured and merged into the graph.
Payload: node_uid: str, coord: tuple[int, int], edges_count: int
"""

SNAIL_MAP_GRAPH_NODE_DELETED = "snail.map_graph.node_deleted"
"""Emitted when a map graph node is removed by the user.
Payload: node_uid: str
"""

SNAIL_MAP_GRAPH_COMPOSITE_UPDATED = "snail.map_graph.composite_updated"
"""Emitted when the map composite image is rebuilt (after add/delete/reload).
Payload: tiles: list[np.ndarray], offsets: list[tuple[int, int]],
        composite: np.ndarray | None, png_bytes: bytes | None
"""

SNAIL_MAP_GRAPH_GRAPH_DROPPED = "snail.map_graph.graph_dropped"
"""Emitted when the entire map graph is wiped.
Payload: (none)
"""

SNAIL_MAP_GRAPH_GRAPH_LOADED = "snail.map_graph.graph_loaded"
"""Emitted when the map graph is loaded from disk (startup or reload).
Payload: node_count: int, edge_count: int
"""

# ── Leaf scope (presentation layer) ────────────────────────────────────

LEAF_USER_REQUEST_CAPTURE = "leaf.user.request_capture"
"""User requests a map tile capture (e.g. Ctrl+7).
Payload: tile_size: int
"""

LEAF_USER_REQUEST_DELETE_NODE = "leaf.user.request_delete_node"
"""User requests deletion of a hovered map node (e.g. Ctrl+Alt+D).
Payload: (none — leaf determines which node is hovered)
"""

LEAF_USER_REQUEST_TOGGLE_TRACKING = "leaf.user.request_toggle_tracking"
"""User toggles character tracking (e.g. Ctrl+Alt+T).
Payload: (none)
"""

LEAF_USER_REQUEST_TOGGLE_OVERLAY = "leaf.user.request_toggle_overlay"
"""User toggles map overlay visibility (e.g. Ctrl+Alt+M).
Payload: (none)
"""

LEAF_USER_REQUEST_TOGGLE_HOVER = "leaf.user.request_toggle_hover"
"""User toggles map node hover highlight.
Payload: (none)
"""

LEAF_USER_REQUEST_TOGGLE_BRECTS = "leaf.user.request_toggle_brects"
"""User toggles UI bounding box marks.
Payload: (none)
"""

LEAF_USER_REQUEST_MOVE_TO_COORD = "leaf.user.request_move_to_coord"
"""User requests PID movement to a map coordinate.
Payload: target_x: int, target_y: int
"""

# ── Screenshot requests (handled by snail) ─────────────────────────────

SNAIL_SCREENSHOT_WINDOW = "snail.screenshot.window"
"""User requests a full-window screenshot.
Payload: (none)
"""

SNAIL_SCREENSHOT_NON_UI = "snail.screenshot.non_ui"
"""User requests a non-UI-area screenshot.
Payload: (none)
"""

SNAIL_SCREENSHOT_CENTER = "snail.screenshot.center"
"""User requests a centered square screenshot.
Payload: size: int (default 100), counter: int
"""
