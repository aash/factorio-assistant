"""HUD overlay — FPS, elapsed time, CPU, and memory stats.

Renders a semi-transparent panel in the top-left corner of the game window.
"""

from __future__ import annotations

from collections import deque

from overlay import OverlayClient

# HUD layout constants
HUD_X = 26
HUD_Y = 54
HUD_W = 340
HUD_H = 76
HUD_BG_ALPHA = 80
HUD_TEXT_X = 40
HUD_LINE1_Y = 80
HUD_LINE2_Y = 98
HUD_LINE3_Y = 116
HUD_FONT = "JetBrainsMono NFM"
HUD_FONT_SIZE = 10
HUD_COLOR = (0, 255, 0, 255)
UNITS_PER_SECOND = 1000


def draw_hud(
    ov: OverlayClient,
    elapsed_sec: float,
    fps_deque: deque,
    assistant_cpu: float,
    overlay_cpu: float,
    assistant_mem_bytes: int,
    overlay_mem_bytes: int,
) -> None:
    """Render the HUD overlay scene.

    Parameters
    ----------
    ov : OverlayClient
    elapsed_sec : float
        Seconds since session start.
    fps_deque : deque[int]
        Rolling window of frame durations (in ms) for FPS calculation.
    assistant_cpu : float
    overlay_cpu : float
    assistant_mem_bytes : int
    overlay_mem_bytes : int
    """
    fps = len(fps_deque) * UNITS_PER_SECOND / sum(fps_deque) if sum(fps_deque) > 0 else 0.0

    with ov.scene("hud") as hud:
        hud.rect(HUD_X, HUD_Y, HUD_W, HUD_H, pen_color=None, brush_color=(0, 0, 0, HUD_BG_ALPHA))
        hud.text(HUD_TEXT_X, HUD_LINE1_Y, f"{elapsed_sec:6.3f}  FPS {fps:06.1f}", HUD_COLOR, HUD_FONT, HUD_FONT_SIZE)
        hud.text(
            HUD_TEXT_X,
            HUD_LINE2_Y,
            f"as cpu {assistant_cpu:05.1f}% mem {(assistant_mem_bytes / (1024 * 1024)):06.1f}MB",
            HUD_COLOR,
            HUD_FONT,
            HUD_FONT_SIZE,
        )
        hud.text(
            HUD_TEXT_X,
            HUD_LINE3_Y,
            f"ov cpu {overlay_cpu:05.1f}% mem {(overlay_mem_bytes / (1024 * 1024)):06.1f}MB",
            HUD_COLOR,
            HUD_FONT,
            HUD_FONT_SIZE,
        )
