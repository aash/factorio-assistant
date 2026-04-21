from typing import List, Tuple, Optional, NoReturn
import logging
import ahk as autohotkey
import dxcam
import cv2
import numpy as np
import time
from enum import Enum
from box import Box
from pathlib import Path
import win32gui  # ty:ignore[unresolved-import]
from common import Rect, UiLocation, get_screen_rect, label_brect

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
    r = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
    r = ((r > 0) * 255).astype(np.uint8)
    ds = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    r = cv2.dilate(r, element)
    r = ((r == 0) * 255).astype(np.uint8)
    r = cv2.dilate(r, element)
    contour, _ = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    res = []
    for i, con in enumerate(contour):
        brect = cv2.boundingRect(con)
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
    CACHE_FILE = 'data/cache.yaml'
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
        self.window_rect_key = str(self.window_rect)
        cfg_file = Path(self.CONFIG_FILE)
        if not cfg_file.exists():
            cfg_file.touch()
        self.config = Box.from_yaml(filename=self.CONFIG_FILE)
        self.cache_path = Path(self.CACHE_FILE)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_path.exists():
            self.cache_path.write_text('{}\n', encoding='utf-8')
        self.cache = Box.from_yaml(filename=self.CACHE_FILE) or Box()
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

        cache_window_rect = self.cache.get('window_rect')
        cache_non_ui_rect = self.cache.get('non_ui_rect')
        cache_ui_brects = self.cache.get('ui_brects')

        if cache_window_rect == self.window_rect_key and cache_non_ui_rect and cache_ui_brects:
            self.non_ui_rect = Rect.from_str(cache_non_ui_rect)
            self.ui_brects = [Rect.from_str(s) for s in cache_ui_brects]
            logging.info('loaded ui layout from cache')
        else:
            ui_brects, _, _ = self.get_widget_brects(default_delay=0.6)
            self.non_ui_rect = self.get_non_ui_rect(ui_brects)
            self.ui_brects = ui_brects
            self._save_cache()
        logging.info(f'non ui rect: {self.non_ui_rect}')
        self.ahk.start_hotkeys()
        time.sleep(0.1)
        return self

    def __exit__(self, *exc_details):
        self.ahk.clear_hotkeys()
        self.ahk.stop_hotkeys()
        time.sleep(0.1)
        logging.info('Stopping snail')
        self.dxgi_dxcam_device.stop()
        self._save_cache()

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
        sleep_time = default_delay
        def initialize():
            time.sleep(sleep_time)
            self.ahk.mouse_move(*self.window_rect.center())
            time.sleep(sleep_time)
        def action():
            self.ahk.send_input(r'^{F9}')
            time.sleep(sleep_time)
        def finalize():
            self.ahk.send_input(r'{F9}')
        im0, im1 = self.get_diff_image(action, initialize, finalize)
        self.ui_brects = get_bounding_rects(im0, im1)
        return self.ui_brects, im0, im1

    def _save_cache(self):
        self.cache.window_rect = self.window_rect_key
        self.cache.non_ui_rect = str(self.non_ui_rect)
        self.cache.ui_brects = [str(r) for r in getattr(self, 'ui_brects', [])]
        self.cache.to_yaml(self.CACHE_FILE)

    def filter_brects(self, rects: List[Rect], query: WidgetType = WidgetType.CENTRAL) -> Rect | None:
        win = Rect(0, 0, *self.window_rect.wh())
        f = filter(lambda x: query.value == label_brect(x, win), rects)
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
        assert char_brect is not None
        assert qbar_brect is not None
        assert mmap_brect is not None
        w = self.window_rect.w - mmap_brect.w
        h = self.window_rect.h - max(char_brect.h, qbar_brect.h)
        non_ui_rect = Rect(0, 0, w, h)
        return non_ui_rect

    def get_char_coords(cls, img: np.ndarray, entity_color = COLOR_GREEN):
        out = img
        out = cv2.inRange(out, entity_color, entity_color)  # ty:ignore[no-matching-overload]
        contour, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cc = set()
        for con in contour:
            r = cv2.boundingRect(con)
            c = r[0] + r[2]//2, r[1] + r[3]//2
            cc.add(c)
        h, w = out.shape
        img_center = np.array((w // 2, h // 2))
        locations = [(c, np.linalg.norm(c - img_center)) for c in cc]
        x = min(locations, key=lambda x: x[1])
        return np.array(x[0])

    def get_entity_coords(cls, img: np.ndarray, entity_color = COLOR_GREEN):
        out = img
        out = cv2.inRange(out, entity_color, entity_color)  # ty:ignore[no-matching-overload]
        contour, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cc = set()
        for con in contour:
            x, y, w, h = cv2.boundingRect(con)
            if w * h > 25 and abs(w - h) < 2:
                c = x + w//2, y + h//2
                cc.add(c)
        return cc
