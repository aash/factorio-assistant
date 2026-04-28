"""Command history widget overlay.

Displays the most recently executed commands in a semi-transparent
panel anchored to the bottom of the game window.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from overlay import OverlayClient
from leaf.state import LeafState
from leaf.renderers.scene_visibility import set_scene_visible

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160
HISTORY_FONT = "JetBrainsMono NFM"
HISTORY_FONT_SIZE = 10
HISTORY_TEXT_COLOR = (220, 220, 220, 255)


def draw_history(
    ov: OverlayClient,
    leaf_state: LeafState,
    input_queue: deque,
    screen_rect: tuple[int, int, int, int],
) -> None:
    """Render the command history widget at the bottom of the given screen rect."""
    x0, y0, w, h = screen_rect
    num = len(input_queue)
    if num == 0:
        set_scene_visible(ov, leaf_state, "history", False)
        return

    box_h = num * HISTORY_LINE_H + HISTORY_MARGIN * 2
    box_y = y0 + h - box_h
    set_scene_visible(ov, leaf_state, "history", True)

    with ov.scene("history") as s:
        s.rect(
            x0 + HISTORY_MARGIN,
            box_y,
            w - 2 * HISTORY_MARGIN,
            box_h,
            pen_color=None,
            brush_color=(0, 0, 0, HISTORY_BG_ALPHA),
        )
        for i, msg in enumerate(reversed(list(input_queue))):
            ty = box_y + HISTORY_MARGIN + i * HISTORY_LINE_H + HISTORY_LINE_H - 2
            s.text(
                x0 + HISTORY_MARGIN * 2,
                ty,
                msg,
                color=HISTORY_TEXT_COLOR,
                font=HISTORY_FONT,
                size=HISTORY_FONT_SIZE,
            )


def refresh_history_widget(
    ov: OverlayClient,
    leaf_state: LeafState,
    input_queue: Optional[deque],
    screen_rect: tuple[int, int, int, int],
) -> None:
    """Conditionally draw or hide the history widget based on toggle state."""
    if leaf_state.show_history_widget and input_queue is not None:
        draw_history(ov, leaf_state, input_queue, screen_rect)
    else:
        set_scene_visible(ov, leaf_state, "history", False)
