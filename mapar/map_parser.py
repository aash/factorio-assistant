import ahk
import ctypes
import ctypes.wintypes
import win32gui  # ty:ignore[unresolved-import]
from common import Rect


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class RECT(ctypes.Structure):
    _fields_ = [
        ("left",   ctypes.c_long),
        ("top",    ctypes.c_long),
        ("right",  ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

user32 = ctypes.windll.user32

class MapParser:

    RESOLUTION1 = {'width': 1920, 'height': 1080}
    RESOLUTION2 = {'width': 1280, 'height': 720}

    def __init__(self, window_name = 'Factorio', ahk = ahk.AHK(version='v2')):
        self.ahk = ahk


    @classmethod
    def get_factorio_client_rect(cls, ahk: ahk.AHK, window_name: str) -> Rect | None:
        window = ahk.find_window(title=window_name)
        if window is None:
            return None
        window_id = int(window.id)
        window.activate()

        client_area_zero = win32gui.ClientToScreen(window_id, (0,0))
        cr = win32gui.GetClientRect(window_id)

        # Equivalent to win32gui.ClientToScreen(window_id, (0, 0))
        # pt = POINT(0, 0)
        # user32.ClientToScreen(window_id, ctypes.byref(pt))
        # client_area_zero = (pt.x, pt.y)

        # Equivalent to win32gui.GetClientRect(window_id)
        # rect = RECT()
        # user32.GetClientRect(window_id, ctypes.byref(rect))
        # cr = (rect.left, rect.top, rect.right, rect.bottom)

        client_rect_dict = Rect(client_area_zero[0], client_area_zero[1], cr[2], cr[3])
        return client_rect_dict
