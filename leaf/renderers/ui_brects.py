"""UI bounding-box overlay.

Draws coloured outlines and labels around detected game-UI widgets
(minimap, quickbar, character panel, etc.).
"""

from __future__ import annotations

from typing import Any

from common import label_brect
from graphics import Rect
from leaf.state import LeafState
from leaf.renderers.scene_visibility import set_scene_visible
from overlay import OverlayClient

UI_BRECT_PEN_COLOR = (0, 255, 0, 255)
UI_BRECT_PEN_WIDTH = 1
UI_BRECT_FONT = "JetBrainsMono NFM"
UI_BRECT_FONT_SIZE = 10


def ui_brect_label(rect: Rect, window: Rect) -> str:
    """Derive a human-readable label for a UI bounding rect."""
    labels = label_brect(rect, window)
    if not labels:
        return "ui"
    return ",".join(sorted(str(lbl) for lbl in labels))


def draw_ui_brect_marks(
    ov: OverlayClient,
    leaf_state: LeafState,
    window_rect: Rect,
    ui_brects: list[Rect],
) -> None:
    """Draw bounding boxes and labels around all detected UI regions."""
    set_scene_visible(ov, leaf_state, "ui_brect_marks", True)
    with ov.scene("ui_brect_marks") as s:
        for uir in ui_brects:
            abs_rect = uir.moved(window_rect.x0, window_rect.y0)
            x, y, w, h = map(int, abs_rect.xywh())
            s.rect(x, y, w, h, pen_color=UI_BRECT_PEN_COLOR, pen_width=UI_BRECT_PEN_WIDTH)
            s.text(
                x + 4,
                y + 20,
                ui_brect_label(abs_rect, window_rect),
                color=UI_BRECT_PEN_COLOR,
                font=UI_BRECT_FONT,
                size=UI_BRECT_FONT_SIZE,
            )
