
from typing import List, Tuple
from scipy.ndimage import center_of_mass
import logging
import ahk as autohotkey
from mapar import MapParser
from d3dshot import D3DShot, CaptureOutputs
from common import DataObject, get_palette, Rect, is_inside, label_brect, UiLocation, erode, dilate
import cv2 as cv
import numpy as np
import time
from enum import Enum

WIDGET_MINIMUM_AREA = 50 * 50


def get_bounding_rects(f0, f1) -> List[Rect]:
    '''
    From two consecutive images of UI try to deduce which parts are stationary,
    then get it's bounding boxes and label them by adjacency to screen edge.

    That labels can be used to deduce UI behind them. For example top right corner is
    a map and tooltip. Bottom center is quick bar.  
    '''
    r = np.bitwise_xor(f0, f1)
    r = cv.cvtColor(r, cv.COLOR_RGB2GRAY)
    r = ((r > 0) * 255).astype(np.uint8)
    ds = 3
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    r = cv.dilate(r, element)
    r = ((r == 0) * 255).astype(np.uint8)
    r = cv.dilate(r, element)
    contour, _ = cv.findContours(r, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    res = []
    for i, con in enumerate(contour):
        brect = cv.boundingRect(con)
        rect = Rect(*brect)
        if rect.width() * rect.height() > WIDGET_MINIMUM_AREA:
            res.append(rect)
    return res


class WidgetType(Enum):

    CENTRAL = {UiLocation.HCENTER, UiLocation.VCENTER}
    MINIMAP = {UiLocation.TOP, UiLocation.RIGHT}
    CHARACTER = {UiLocation.BOTTOM, UiLocation.LEFT}
    QUICKBAR = {UiLocation.BOTTOM, UiLocation.HCENTER}

    def __str__(self):
        return self.name.lower()


class Snail:

    FACTORIO_WINDOW_NAME = 'Factorio'
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (0, 255, 0)

    def __init__(self):
        window_name = self.FACTORIO_WINDOW_NAME
        self.ahk = autohotkey.AHK()
        self.ahk.set_coord_mode('Mouse', 'Client')
        self.window = self.ahk.find_window(title=window_name)
        self.window_id = int(self.window.id, 16)
        self.window.activate()
        robj = DataObject(MapParser.get_factorio_client_rect(self.ahk, window_name))
        self.window_rect = Rect(robj.x, robj.y, robj.width, robj.height)
        pass

    def __enter__(self):
        logging.info('Starting snail')
        self.d3d = D3DShot(capture_output=CaptureOutputs.NUMPY)
        self.d3d.capture(target_fps=30, region=self.window_rect.xyxy())
        return self

    def __exit__(self, *exc_details):
        logging.info('Stopping snail')
        self.d3d.stop()
        del self.ahk

    def get_diff_image(self, action, initialize = None, finalize = None, roi = None):

        '''
        Get two consequtive images of a window, taken before action and after,
        with the option of initialization and finalization.
        '''
        if initialize:
            initialize()
        im1, _ = self.d3d.wait_next_frame()
        action()
        im2, _ = self.d3d.wait_next_frame()
        if finalize:
            finalize()
        return im1, im2

    def get_widget_brects(self) -> Rect:
        it_count = 3
        sleep_time = 0.2
        def initialize():
            self.window.activate()
            self.ahk.mouse_move(1, 1)
            time.sleep(sleep_time)
        def action():
            self.ahk.send_input(f'{{WheelUp {it_count}}}', blocking=True)
            time.sleep(sleep_time)
        def finalize():
            self.ahk.send_input(f'{{WheelDown {it_count}}}', blocking=True)
        im0, im1 = self.get_diff_image(action, initialize, finalize)
        brects = get_bounding_rects(im0, im1)
        return brects, im0, im1

    def filter_brects(self, rects: List[Rect], query: WidgetType = WidgetType.CENTRAL) -> Rect:
        win = Rect(0, 0, *self.window_rect.wh())
        f = filter(lambda x: query.value == label_brect(x, win), rects)
        l = list(map(lambda x: label_brect(x, win), rects))
        brect = next(f, None)
        return brect
    
    def wait_next_frame(self) -> np.ndarray:
        f, *_ = self.d3d.wait_next_frame()
        return f

    def wait_next_frame_with_time(self) -> np.ndarray:
        f, t = self.d3d.wait_next_frame()
        return f, t
    
    def ensure_next_frame(self):
        im = None
        while im is None:
            im = self.wait_next_frame()
        return

    def ensure_zoom_level(self):
        animation_duration_sec = 1.0
        self.ahk.send_input('{F9}')
        # wait the animation
        time.sleep(animation_duration_sec)


    
    def non_ui_rect(self, r):
        char_brect = self.filter_brects(r, WidgetType.CHARACTER)
        qbar_brect = self.filter_brects(r, WidgetType.QUICKBAR)
        mmap_brect = self.filter_brects(r, WidgetType.MINIMAP)
        w = self.window_rect.w - mmap_brect.w
        h = self.window_rect.h - max(char_brect.h, qbar_brect.h)
        non_ui_rect = Rect(0, 0, w, h)
        return non_ui_rect

    
    def find_grid_nodes(cls, img: np.ndarray, grid_color = COLOR_BLACK):
        out = img
        out = cv.inRange(out, grid_color, grid_color)
        kernel_size = 11
        kernel_v = np.zeros((kernel_size, kernel_size)) 
        kernel_h = np.copy(kernel_v) 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        kernel_v /= kernel_size 
        kernel_h /= kernel_size 
        outv = cv.filter2D(out, -1, kernel_v) 
        outv = cv.inRange(outv, 200, 255)
        outh = cv.filter2D(out, -1, kernel_h) 
        outh = cv.inRange(outh, 200, 255)
        out = cv.bitwise_or(outh, outv)
        out = erode(out, 1, cv.MORPH_RECT)
        out = erode(out, 1, cv.MORPH_ELLIPSE)
        y, x = np.where(out != 0)
        c = list(sorted(zip(x, y)))
        return c
    
    def get_char_coords(cls, img: np.ndarray, entity_color = COLOR_GREEN):
        out = img
        out = cv.inRange(out, entity_color, entity_color)
        contour, _ = cv.findContours(out, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cc = set()
        for con in contour:
            r = cv.boundingRect(con)
            c = r[0] + r[2]//2, r[1] + r[3]//2
            cc.add(c)
        h, w = out.shape
        img_center = np.array((w // 2, h // 2))
        locations = [(c, np.linalg.norm(c - img_center)) for c in cc]
        x = min(locations, key=lambda x: x[1])
        return x[0]




# class ZoomInZoomOut:
#     '''
#             def action():
#                 # for i in range(it_count):
#                 self.ahk.send(f'{{WheelUp {it_count}}}')
#                 sleep(sleep_time)
#             def action_cleanup():
#                 # for i in range(it_count):
#                 self.ahk.send(f'{{WheelDown {it_count}}}')
#             im, im1 = self.get_diff_image(init, action)
#             action_cleanup()
#             r, *_ = get_bounding_rect(im, im1)
#             return r
#     '''

#     def __init__(self, snail: Snail, it_count = 4, sleep_time = 0.05):
#         self.it_count = it_count
#         self.sleep_time = sleep_time
#         self.snail = snail
#         self.cx = self.snail.client_rect[0]
#         self.cy = self.snail.client_rect[1]
#         self.snail.window.activate()
#         self.snail.ahk.mouse_move(self.cx, self.cy)

#     def __enter__(self):
#         self.snail.ahk.send(f'{{WheelUp {self.it_count}}}')
#         time.sleep(self.sleep_time)

#     def __exit__(self, *exc_details):
#         pass

