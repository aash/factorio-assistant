from PIL import Image
from typing import List, Tuple, Optional, NoReturn
#from scipy.ndimage import center_of_mass
import logging
import ahk as autohotkey
import dxcam
from common import * 
import cv2 as cv
import numpy as np
import time
from enum import Enum
import itertools
import yaml
import os
from box import Box
from pathlib import Path
import win32gui  # ty:ignore[unresolved-import]


from contextlib import contextmanager
WIDGET_MINIMUM_AREA = 50 * 50


def get_factorio_client_rect(ahk: autohotkey.AHK, window_name: str) -> Rect | None:
    window = ahk.find_window(title=window_name)
    if window is None:
        return None
    window_id = int(window.id)
    client_area_zero = win32gui.ClientToScreen(window_id, (0,0))
    cr = win32gui.GetClientRect(window_id)
    client_rect_dict = Rect(client_area_zero[0], client_area_zero[1], cr[2], cr[3])
    return client_rect_dict


def wh2str(wh: Tuple[int, int]) -> str:
    return f'{wh[0]}x{wh[1]}'

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

def log_and_raise(msg: str, exc: type[RuntimeError] = RuntimeError) -> NoReturn:
    logging.error(msg)
    raise exc(msg)


class WidgetType(Enum):

    CENTRAL = {UiLocation.HCENTER, UiLocation.VCENTER}
    MINIMAP = {UiLocation.TOP, UiLocation.RIGHT}
    CHARACTER = {UiLocation.BOTTOM, UiLocation.LEFT}
    QUICKBAR = {UiLocation.BOTTOM, UiLocation.HCENTER}

    def __str__(self):
        return self.name.lower()


class SnailWindowMode(Enum):
    WINDOWED = 1
    FULL_SCREEN = 2

class Snail:

    CONFIG_FILE = 'config.yaml'
    FACTORIO_WINDOW_NAME = 'Factorio'
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (0, 255, 0)

    def __init__(self, window_mode = SnailWindowMode.WINDOWED):
        self.ahk = autohotkey.AHK(version='v2')
        self.window_mode = window_mode
        if self.window_mode == SnailWindowMode.WINDOWED:
            self.ahk.set_coord_mode('Mouse', 'Screen')
            self.window = self.ahk.find_window(title=self.FACTORIO_WINDOW_NAME)
            if self.window is None:
                log_and_raise('game window is not found')
            rect = get_factorio_client_rect(self.ahk, self.FACTORIO_WINDOW_NAME)
            if rect is None:
                log_and_raise(f'could not get `{self.window.get_title()}` window client area rectangle')
            self.window_id = int(self.window.id)
            self.window_rect = rect
            self.window.activate()
        else:
            raise RuntimeError('mode is not supported')
            self.ahk.set_coord_mode('Mouse', 'Screen')
            self.window = None
            self.window_id = None
            self.window_rect = get_screen_rect()
        cfg_file = Path(self.CONFIG_FILE)
        if not cfg_file.exists():
            cfg_file.touch()
        self.config = Box.from_yaml(filename=self.CONFIG_FILE)
        self._dxgi_backend = "dxcam"
        self.dxgi_fps = 120
        self.debug_ui_rect = None

    def __enter__(self):
        logging.info('Starting snail')
        self.dxgi_dxcam_device = dxcam.create(
            backend="dxgi", # default Desktop Duplication backend
            processor_backend="cv2", # default OpenCV processor
            output_color="BGR",
            region=self.window_rect.xyxy(),
        )
        self.dxgi_dxcam_device.start(target_fps=self.dxgi_fps)
        logging.info(f'snail started {self.window_rect}')
        self.ensure_next_frame()
        
        if self.window_rect != Rect.from_str(self.config.get('prev_win_resolution')) or \
            not hasattr(self.config, 'prev_non_ui_rect'):
            r, _, _ = self.get_widget_brects(default_delay=0.6)
            self.non_ui_rect = self.get_non_ui_rect(r)
        else:
            self.non_ui_rect = Rect.from_str(self.config.prev_non_ui_rect)
        logging.info(f'non ui rect: {self.non_ui_rect}')
        non_ui_img = self.wait_next_frame()
        ents = self.get_entity_coords(non_ui_img)
        logging.info(ents)
        if len(ents) > 0:
            self.entity_positions_enabled = True
            self.char_offset = self.get_char_coords(non_ui_img)
            logging.info(f'character offset: {self.char_offset}')
        else:
            self.entity_positions_enabled = False
            if self.window_rect.wh() == (3840, 2160):
                self.char_offset = np.array((1921, 1081))
            elif self.window_rect.wh() == (1920, 1080):
                self.char_offset = np.array((960, 541))
            else:
                self.char_offset = None
        self.ahk.start_hotkeys()
        time.sleep(0.1)
        return self

    def __exit__(self, *exc_details):
        self.ahk.clear_hotkeys()
        self.ahk.stop_hotkeys()
        time.sleep(0.1)
        logging.info('Stopping snail')
        self.dxgi_dxcam_device.stop()
        self.config.prev_win_resolution = str(self.window_rect)
        self.config.prev_non_ui_rect = str(self.non_ui_rect)
        self.config.char_location = {'3840x2160': '1921,1081',
            '1920x1080': '960,541'}
        self.config.to_yaml(self.CONFIG_FILE)
        logging.info('write config to disk')

    def get_diff_image(self, action, initialize = None, finalize = None, roi = None):

        '''
        Get two consequtive images of a window, taken before action and after,
        with the option of initialization and finalization.
        '''
        if initialize:
            initialize()
        im1 = self.wait_next_frame()
        action()
        im2 = self.wait_next_frame()
        if finalize:
            finalize()
        return im1, im2

    def get_widget_brects(self, default_delay=0.2) -> Tuple[List[Rect], np.ndarray, np.ndarray]:
        it_count = 2
        sleep_time = default_delay
        def initialize():
            time.sleep(sleep_time)
            # logging.info('init 111')
            self.ahk.mouse_move(*self.window_rect.center())
            time.sleep(sleep_time)
        def action():
            # self.ahk.send_input(f'{{WheelUp {it_count}}}', blocking=True)
            # logging.info('action 111')
            self.ahk.send_input(r'^{F9}', blocking=True)
            time.sleep(sleep_time)
        def finalize():
            # self.ahk.send_input(f'{{WheelDown {it_count}}}', blocking=True)
            # logging.info('finalize 111')
            self.ahk.send_input(r'{F9}', blocking=True)
        im0, im1 = self.get_diff_image(action, initialize, finalize)
        self.ui_brects = get_bounding_rects(im0, im1)
        # cv2.imwrite('brects_im0.png', im0)
        # cv2.imwrite('brects_im1.png', im1)
        logging.info(f'{self.ui_brects}')
        return self.ui_brects, im0, im1
    
    def get_widget_brects1(self):
        self.ahk.mouse_move(1, 1)
        im1 = self.wait_next_frame()
        with zoom_and_restore(self.ahk, 2, 0.2):
            im2 = self.wait_next_frame()
        brects = get_bounding_rects(im1, im2)

    def filter_brects(self, rects: List[Rect], query: WidgetType = WidgetType.CENTRAL) -> Rect:
        win = Rect(0, 0, *self.window_rect.wh())
        f = filter(lambda x: query.value == label_brect(x, win), rects)
        l = list(map(lambda x: label_brect(x, win), rects))
        brect = next(f, None)
        return brect
    
    def wait_next_frame(self, roi: Optional[Rect] = None) -> np.ndarray:
        if roi is not None:
            raise RuntimeError('unsupported argument')
        f = self.dxgi_dxcam_device.get_latest_frame()
        if f is not None:
            return f
        else:
            raise RuntimeError('no frame data')

    def wait_next_frame_with_time(self, roi: Optional[Rect] = None) -> Tuple[np.ndarray, float]:
        if roi is not None:
            raise RuntimeError('unsupported argument')
        r = self.dxgi_dxcam_device.get_latest_frame(with_timestamp=True)
        if r is not None:
            return r
        else:
            raise RuntimeError('no frame data')
    
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


    
    def get_non_ui_rect(self, r):
        char_brect = self.filter_brects(r, WidgetType.CHARACTER)
        qbar_brect = self.filter_brects(r, WidgetType.QUICKBAR)
        mmap_brect = self.filter_brects(r, WidgetType.MINIMAP)
        w = self.window_rect.w - mmap_brect.w
        h = self.window_rect.h - max(char_brect.h, qbar_brect.h)
        non_ui_rect = Rect(0, 0, w, h)
        return non_ui_rect

    @classmethod
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
        return np.array(x[0])

    def get_entity_coords(cls, img: np.ndarray, entity_color = COLOR_GREEN):
        out = img
        out = cv.inRange(out, entity_color, entity_color)
        contour, _ = cv.findContours(out, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cc = set()
        for con in contour:
            x, y, w, h = cv.boundingRect(con)
            if w * h > 25 and abs(w - h) < 2:
                c = x + w//2, y + h//2
                cc.add(c)
        return cc


def get_grid_rect(grid_wh, grid_node, char_offs):
    #False_True
    right_left = grid_node[0] <= char_offs[0]
    #False_True
    bottom_top = grid_node[1] <= char_offs[1]
    d = {
        (True, True): 'top_left',
        (True, False): 'top_right',
        (False, True): 'bottom_left',
        (False, False): 'bottom_right',
    }
    m = getattr(Rect, f'from_{d[(bottom_top, right_left)]}')
    return m(*grid_node, *grid_wh)



def chk(img, p):
    v = np.array([[0, +1], [0, -1]])
    h = np.array([[+1, 0], [-1, 0]])
    # e = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
    # ep =list(map(lambda x: img[*(x + p)] == 0, e))
    p = np.array(p)
    vp = list(map(lambda x: img[*(x + p)] == 255, v))
    nvp = list(map(lambda x: img[*(x + p)] == 0, v))
    hp = list(map(lambda x: img[*(x + p)] == 255, h))
    nhp = list(map(lambda x: img[*(x + p)] == 0, h))
    p, q, r, s = all(vp), all(hp), img[*p], True
    if all(vp) and all(hp) and r:
        return 'big'
    if all(vp) and all(nhp) and r:
        return 'vert'
    if all(hp) and all(nvp) and r:
        return 'horz'
    if all(nvp) and all(nhp) and r:
        return 'small'
    return 'none'

# def get_grid(im, threshold = 100, grid_color = (0,0,0)):
#     def get_trilples(lst):
#         triples = []
#         if len(lst) >= 3:
#             for i in range(len(lst)-2):
#                 if lst[i] + 1 == lst[i+1] and lst[i+1] + 1 == lst[i+2]:
#                     triples.append((lst[i], lst[i+1], lst[i+2]))
#         return triples
#     def filter_triples(lst):
#         l = lst
#         ml = []
#         for t in get_trilples(lst):
#             ml.append(t[1])
#             l.remove(t[0])
#             l.remove(t[2])
#         return l, ml
#     def get_diffs(l):
#         return list(map(lambda x: x[0] - x[1], zip(l[1:], l[:-1])))
#     def get_most_frequent_value(l):
#         counts = np.bincount(l)
#         sorted_values = np.argsort(-counts)
#         most_frequent_value = sorted_values[0]
#         return most_frequent_value
#     def find_a_for_max_count(arr, w):
#         max_count = 0
#         best_a = arr.min()  
#         for a in arr:
#             count = np.count_nonzero([(val - a) % w == 0 for val in arr])
#             if count > max_count:
#                 max_count = count
#                 best_a = a
#         return best_a

#     out = im
#     h, w, *_ = out.shape
#     out = cv.inRange(out, grid_color, grid_color)
#     # cv.imwrite('grid.png', out)
#     r = []
#     for ax, mx in zip([0, 1], [h, w]):
#         s = np.sum(out/255, axis=ax)
#         lines = list(np.where(np.abs(s - mx) <= threshold)[0])
#         l, ml = filter_triples(lines)
#         r.append((l, ml))
    
#     # get lengths between consequtive grid lines
#     # logging.info(f'vl: {r[0][0]}\nhl: {r[1][0]}')
#     vdiffs = get_diffs(r[0][0])
#     hdiffs = get_diffs(r[1][0])
#     # most probable grid size
#     # logging.info(f'vdiffs: {vdiffs}\nhdiffs: {hdiffs}')
#     grid_size = get_most_frequent_value(vdiffs + hdiffs)
#     # find the most probable starting point
#     vstart = find_a_for_max_count(np.array(r[0][0]), grid_size)
#     hstart = find_a_for_max_count(np.array(r[1][0]), grid_size)
#     vn = list(range(vstart, w, grid_size))
#     hn = list(range(hstart, h, grid_size))

#     # logging.info(f'vstart: {vstart}, hstart: {hstart}, grid_size: {grid_size}')
#     # logging.info(f'vn: {vn}\nhn: {hn}')



#     return vn, hn, r[0][1], r[1][1], grid_size
