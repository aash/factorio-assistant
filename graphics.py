from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from typing import Tuple

@dataclass
class Segment:
    left: int
    right: int

@dataclass
class Rect:
    x0: int
    y0: int
    w: int
    h: int

    def __hash__(self):
        return hash((self.x0, self.y0, self.w, self.h))
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.xywh() == other
        elif isinstance(other, list):
            return list(self.xywh()) == other
        elif isinstance(other, Rect):
            return (self.x0, self.y0, self.w, self.h) == (other.x0, other.y0, other.w, other.h)
        else:
            return False

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return ','.join(map(str, self.xywh()))

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
    
    def xywh(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.w, self.h)
    
    def xyxy(self):
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
    
    def width(self):
        return self.w
    
    def height(self):
        return self.h
    
    def wh(self) -> np.ndarray:
        return np.array((self.w, self.h))
    
    def xy(self):
        return np.array((self.x0, self.y0))
    
    def sub_rect(self, sub: 'Rect'):
        return Rect(self.x0 + sub.x0, self.y0 + sub.y0, *sub.wh())
    
    def moved(self, dx: int, dy: int):
        return Rect(self.x0 + dx, self.y0 + dy, self.w, self.h)
    
    def __add__(self, other: np.ndarray):
        return Rect(self.x0 + other[0], self.y0 + other[1], self.w, self.h)
    
    def center(self) -> np.ndarray:
        return np.array((self.x0 + self.w // 2, self.y0 + self.h // 2))

    @classmethod
    def from_centdims(cls, cx, cy, w, h) -> 'Rect':
        """ Constructs Rect from center point (cx, cy) and dimensions (w, h)
        """
        return Rect(cx - w // 2, cy - h // 2, w, h)

    @classmethod
    def from_str(cls, s:str) -> 'Rect':
        """ Constructs Rect from string representation 'x, y, w, h' or 'Rect(x, y, w, h)'
        """
        assert isinstance(s, str), 'parameter should be str(ing)'
        if s.startswith('Rect'):
            # TODO: implement parsing of Rect(a, b, c ,d) strings
            return Rect(0, 0, 0, 0)
        return Rect(*map(int, s.split(',')))

    @classmethod
    def from_ptdm(cls, pt: Tuple, dim: Tuple) -> 'Rect':
        """ Constructs Rect from top left point pt and dimensions dim
        """
        return Rect(*pt, *dim)
    
    @classmethod
    def from_xyxy(cls, x0: int, y0: int, x1: int, y1: int) -> 'Rect':
        """ Constructs Rect from unordered diagonal points (x0, y0) and (x1, y1)
        """
        p0, p1 = sorted((x0, x1))
        q0, q1 = sorted((y0, y1))
        return Rect(p0, q0, p1 - p0, q1 - q0)
    
    @classmethod
    def from_top_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from top left point (x, y) and dimensions (w, h)
        """
        return Rect(x, y, w, h)

    @classmethod
    def from_bottom_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from bottom left point (x, y) and dimensions (w, h)
        """
        return Rect(x, y - h, w, h)

    @classmethod
    def from_bottom_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from bottom right point (x, y) and dimensions (w, h)
        """
        return Rect(x - w, y - h, w, h)

    @classmethod
    def from_top_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from top right point (x, y) and dimensions (w, h)
        """
        return Rect(x - w, y, w, h)


def crop_image(img: np.ndarray, r: Rect) -> np.ndarray:
    """
    Crops a part of an image using a rectangle defined by the top-left corner, width, and height.
    If the rectangle goes beyond the image boundaries, it will be truncated.
    """

    r = deepcopy(r)

    x0 = max(0, r.x0)
    y0 = max(0, r.y0)
    x1 = min(img.shape[1], r.x0 + r.w)
    y1 = min(img.shape[0], r.y0 + r.h)
    return img[y0:y1, x0:x1].copy()
