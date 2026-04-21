from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget


class InvisibleToolWindow(QWidget):
    def __init__(self, *, steal_focus_interval_ms: int = 500) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(1, 1)
        self.move(-10000, -10000)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._focus_timer = QTimer(self)
        self._focus_timer.setInterval(steal_focus_interval_ms)
        self._focus_timer.timeout.connect(self._steal_focus)

    def _steal_focus(self) -> None:
        self.activateWindow()
        self.raise_()
        self.setFocus()


@contextmanager
def key_capture_window(
    *,
    grab_keyboard: bool = True,
    steal_focus_interval_ms: int = 50,
) -> Generator[tuple[QApplication, InvisibleToolWindow], None, None]:
    """
    Context manager that creates an invisible top-level tool window
    which periodically steals focus to ensure keyboard input is captured
    by the AHK InputHook running alongside.

    Yields a tuple of (app, window):
        - app: the QApplication instance (created if needed)
        - window: the invisible tool window widget

    When grab_keyboard=True (default), the window calls grabKeyboard()
    so all Qt keyboard events are routed to it regardless of focus.

    The window periodically steals focus every steal_focus_interval_ms
    milliseconds (default 50).

    Call app.exec() in the caller to start the Qt event loop.
    """
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = InvisibleToolWindow(steal_focus_interval_ms=steal_focus_interval_ms)
    window.show()
    if grab_keyboard:
        window.grabKeyboard()
    window._focus_timer.start()

    try:
        yield app, window  # type: ignore[arg-type]
    finally:
        window._focus_timer.stop()
        if grab_keyboard:
            try:
                window.releaseKeyboard()
            except RuntimeError:
                pass
        window.close()
        if owns_app:
            app.quit()