import sys
import ctypes
from ctypes import wintypes
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from OpenGL.GL import (
    glEnable,
    glBlendFunc,
    glViewport,
    glClearColor,
    glClear,
    GL_COLOR_BUFFER_BIT,
    GL_BLEND,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
)


class MARGINS(ctypes.Structure):
    _fields_ = [("cxLeftWidth", ctypes.c_int), ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int), ("cyBottomHeight", ctypes.c_int)]

WS_EX_NOREDIRECTIONBITMAP = 0x00200000
WS_EX_TRANSPARENT         = 0x00000020  # add this
GWL_EXSTYLE               = -20
WDA_EXCLUDEFROMCAPTURE    = 0x00000011

def setup_window(hwnd: int):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi")

    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE,
        ex | WS_EX_NOREDIRECTIONBITMAP | WS_EX_TRANSPARENT  # add WS_EX_TRANSPARENT
    )

    margins = MARGINS(-1, -1, -1, -1)
    dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))

    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype  = wintypes.BOOL
    user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

class GLOverlay(QOpenGLWidget):
    """OpenGL widget — renders with true per-pixel alpha via GPU."""

    def initializeGL(self):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush
        from PySide6.QtCore import QRect, QPoint

        # Clear to transparent first
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Use QPainter on the QOpenGLWidget directly
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # --- Full QPainter API is available ---

        # Shapes
        painter.setBrush(QBrush(QColor(255, 0, 0, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(100, 100, 200, 200)

        # Strokes
        painter.setPen(QPen(QColor(0, 255, 0, 220), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRect(350, 100, 150, 100))

        # Text
        painter.setPen(QColor(255, 255, 255, 230))
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        painter.drawText(QPoint(100, 80), "Overlay text")

        # Lines
        painter.setPen(QPen(QColor(255, 255, 0, 200), 2))
        painter.drawLine(0, 0, self.width(), self.height())

        # Pixmaps / images
        # painter.drawPixmap(0, 0, QPixmap("image.png"))

        painter.end()


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        # NO WA_TranslucentBackground — that would force WS_EX_LAYERED
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gl = GLOverlay(self)
        layout.addWidget(self.gl)

        self.resize(800, 600)

    def showEvent(self, event):
        super().showEvent(event)
        setup_window(int(self.winId()))


if __name__ == "__main__":
    # Enable alpha in the OpenGL surface format
    fmt: QSurfaceFormat = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    fmt.setRedBufferSize(8)
    fmt.setGreenBufferSize(8)
    fmt.setBlueBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    win = OverlayWindow()
    win.show()
    sys.exit(app.exec())