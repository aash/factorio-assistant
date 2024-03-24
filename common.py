import numpy as np
import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from typing import List

class DataObject:
    def __init__(self, data_dict):
        self.__dict__ = data_dict

'''
h: 0-179
s: 0-255
v: 0-255
'''
def hsv2rgb(hsv):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0][0] = (np.array(hsv) * np.array([179, 255, 255])).astype(np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return tuple(map(int, (rgb[0][0]*255).astype(np.uint8)))

def get_palette(size: int):
    assert 0 < size <= 42
    f = 1/size
    for c in range(size):
        print(c*f)
    return [hsv2rgb((c*f, 0.99, 0.99)) for c in range(size)]

def bits(n: int):
    l = [n >> i & 1 for i in range(n.bit_length())]
    ll = list(enumerate(l))
    lll = set(map(lambda x: x[1] << x[0], ll))
    return lll

class UiLocation(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4
    HCENTER = 5
    VCENTER = 6

    def __str__(self):
        return self.name.split('.')[-1].lower()


@dataclass
class Rect:
    x0: int
    y0: int
    w: int
    h: int

    def top(self) -> int:
        return self.y0
    
    def bottom(self) -> int:
        return self.y0 + self.h
    
    def left(self) -> int:
        return self.x0
    
    def right(self) -> int:
        return self.x0 + self.w
    
    def top_left(self):
        return (self.x0, self.y0)

    def top_right(self):
        return (self.x0 + self.w, self.y0)
    
    def bottom_left(self):
        return (self.x0, self.y0 + self.h)

    def bottom_right(self):
        return (self.x0 + self.w, self.y0 + self.h)
    
    def left_segment(self):
        return Segment(self.y0, self.y0 + self.h)

    def top_segment(self):
        return Segment(self.x0, self.x0 + self.w)
    
    def xywh(self):
        return (self.x0, self.y0, self.w, self.h)
    
    def xyxy(self):
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
    
    def width(self):
        return self.w
    
    def height(self):
        return self.h
    
    def wh(self):
        return (self.w, self.h)
    
    def xy(self):
        return (self.x0, self.y0)
    
    @classmethod
    def from_xyxy(cls, _x0: int, _y0: int, _x1: int, _y1: int) -> 'Rect':
        x0 = min(_x0, _x1)
        y0 = min(_y0, _y1)
        x1 = max(_x0, _x1)
        y1 = max(_y0, _y1)
        return Rect(x0, y0, x1 - x0, y1 - y0)
    
    @classmethod
    def from_top_left(cls, _x: int, _y: int, _w: int, _h: int) -> 'Rect':
        return Rect(_x, _y, _w, _h)

    @classmethod
    def from_bottom_left(cls, _x: int, _y: int, _w: int, _h: int) -> 'Rect':
        return Rect(_x, _y - _h, _w, _h)

    @classmethod
    def from_bottom_right(cls, _x: int, _y: int, _w: int, _h: int) -> 'Rect':
        return Rect(_x - _w, _y - _h, _w, _h)

    @classmethod
    def from_top_right(cls, _x: int, _y: int, _w: int, _h: int) -> 'Rect':
        return Rect(_x - _w, _y, _w, _h)
    
@dataclass
class Segment:
    left: int
    right: int

class BoundingRect:

    def __init__():
        pass

def is_inside(p: Segment, q: Segment, threshold: int = 1):
    '''
    check if `q` segment is inside another segment `p` with threshold `threshold`
    '''
    return q.left - p.left > threshold and p.right - q.right > threshold
    
def label_brect(rect: Rect, window: Rect, threshold: int = 1):
    lbls = set()
    if rect.left() - window.left() < threshold:
        lbls.add(UiLocation.LEFT)
    if window.right() - rect.right() < threshold:
        lbls.add(UiLocation.RIGHT)
    if rect.top() - window.top() < threshold:
        lbls.add(UiLocation.TOP)
    if window.bottom() - rect.bottom() < threshold:
        lbls.add(UiLocation.BOTTOM)
    if is_inside(window.left_segment(), rect.left_segment(), threshold):
        lbls.add(UiLocation.VCENTER)
    if is_inside(window.top_segment(), rect.top_segment(), threshold):
        lbls.add(UiLocation.HCENTER)
    return lbls

# def crop_image(img: np.ndarray, r: Rect) -> np.ndarray:
#     b = r.xyxy()
#     return img[b[1]:b[3], b[0]:b[2]].copy()

def crop_image(img: np.ndarray, r: Rect, debug = False) -> np.ndarray:
    """
    Crops a part of an image using a rectangle defined by the top-left corner, width, and height.
    If the rectangle goes beyond the image boundaries, it will be truncated.
    """
    x0 = max(0, r.x0)
    y0 = max(0, r.y0)
    x1 = min(img.shape[1], r.x0 + r.w)
    y1 = min(img.shape[0], r.y0 + r.h)
    if debug:
        return img[y0:y1, x0:x1], (x0, y0), (x1, y1)
    else:
        return img[y0:y1, x0:x1]

def erode(img: np.ndarray, sz: int, shape):
    el = cv.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv.erode(img, el)

def dilate(img: np.ndarray, sz: int, shape):
    el = cv.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv.dilate(img, el)

class MoveDirectionSimple(Enum):
    UP     = 0b0001
    DOWN   = 0b0010
    LEFT   = 0b0100
    RIGHT  = 0b1000

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class MoveDirectionComposite(Enum):
    UP_LEFT = MoveDirectionSimple.UP.value | MoveDirectionSimple.LEFT.value
    UP_RIGTH = MoveDirectionSimple.UP.value | MoveDirectionSimple.RIGHT.value
    DOWN_LEFT = MoveDirectionSimple.DOWN.value | MoveDirectionSimple.LEFT.value
    DOWN_RIGHT = MoveDirectionSimple.DOWN.value | MoveDirectionSimple.RIGHT.value

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class MoveDirection(Enum):
    UP     = MoveDirectionSimple.UP.value
    DOWN   = MoveDirectionSimple.DOWN.value
    LEFT   = MoveDirectionSimple.LEFT.value
    RIGHT  = MoveDirectionSimple.RIGHT.value
    UP_LEFT = MoveDirectionComposite.UP_LEFT.value
    UP_RIGTH = MoveDirectionComposite.UP_RIGTH.value
    DOWN_LEFT = MoveDirectionComposite.DOWN_LEFT.value
    DOWN_RIGHT = MoveDirectionComposite.DOWN_RIGHT.value

    def simplify(self) -> List[MoveDirectionSimple]:
        bb = bits(self.value).difference({0})
        return [MoveDirectionSimple(m) for m in bb]

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class KeyState(Enum):
    PRESS = 0
    RELEASE = 1

def wrap(s: str, c: str) -> str:
    d = {
        '{': ('{', '}'),
        '}': ('{', '}'),
        '[': ('[', ']'),
        ']': ('[', ']'),
        '(': ('(', ')'),
        ')': ('(', ')'),
    }
    if c in d:
        return d[c][0] + s + d[c][1]
    else:
        raise RuntimeError('unreachable')

def get_ahk_sequence(dir: MoveDirection, key_state: KeyState) -> str:
    d2k = {
        MoveDirectionSimple.UP: 'w',
        MoveDirectionSimple.DOWN: 's',
        MoveDirectionSimple.LEFT: 'a',
        MoveDirectionSimple.RIGHT: 'd',
    }
    ks2s = {
        KeyState.RELEASE: 'up',
        KeyState.PRESS: 'down',
    }
    s = ks2s[key_state]
    k = [d2k[_k] for _k in dir.simplify()]
    ss = [wrap(f'{_k} {s}', '{') for _k in k]
    return ''.join(ss)

