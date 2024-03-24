import ahk
import win32gui
# import win32ui
# import win32con
# import win32api
import numpy as np
from common import DataObject


class MapParser:

    RESOLUTION1 = {'width': 1920, 'height': 1080}
    RESOLUTION2 = {'width': 1280, 'height': 720}

    def __init__(self, window_name = 'Factorio', ahk = ahk.AHK()):
        self.ahk = ahk


    @classmethod
    def get_factorio_client_rect(cls, ahk: ahk.AHK, window_name: str) -> dict:
        window = ahk.find_window(title=window_name)
        window_id = int(window.id, 16)
        window.activate()
        client_area_zero = win32gui.ClientToScreen(window_id, (0,0))
        cr = win32gui.GetClientRect(window_id)
        client_rect_dict = {
            'x': client_area_zero[0],
            'y': client_area_zero[1],
            'width': cr[2],
            'height': cr[3]
        }
        return client_rect_dict


    @classmethod
    def get_factorio_gui_state(cls, ahk, window_name):
        pass
    
    @classmethod
    def get_window_snapshot(cls, window_id):
        return None, #take_screenshot_of_window_to_numpy(window_id)

    @classmethod
    def get_main_window_caption(cls, window_id):
        pass


# def take_screenshot_of_window_to_numpy(hwnd):
        
#     # Make sure the window is in the foreground and restored
#     win32gui.SetForegroundWindow(hwnd)
#     win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

#     # Get the window's client area dimensions
#     left, top, right, bot = win32gui.GetClientRect(hwnd)
#     width = right - left
#     height = bot - top

#     # # Adjust dimensions for GetWindowRect
#     # left, top, right, bot = win32gui.GetWindowRect(hwnd)
#     # width = right - left
#     # height = bot - top

#     # Get window context and create a compatible DC
#     hwndDC = win32gui.GetWindowDC(hwnd)
#     mfcDC = win32ui.CreateDCFromHandle(hwndDC)
#     saveDC = mfcDC.CreateCompatibleDC()

#     # Create a bitmap object
#     saveBitMap = win32ui.CreateBitmap()
#     saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
#     saveDC.SelectObject(saveBitMap)

#     # Blit (copy) the client area of the window into our DC and bitmap
#     result = saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

#     # Convert the bitmap to a NumPy array
#     signedIntsArray = saveBitMap.GetBitmapBits(True)
#     img = np.frombuffer(signedIntsArray, dtype='uint8').reshape(height, width, 4).copy()

#     # Cleanup
#     win32gui.DeleteObject(saveBitMap.GetHandle())
#     saveDC.DeleteDC()
#     mfcDC.DeleteDC()
#     win32gui.ReleaseDC(hwnd, hwndDC)

#     return img
