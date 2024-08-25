import cv2 as cv
import numpy as np
from typing import Callable, Tuple
from graphics import Rect, crop_image

class npext:
    def __init__(self, array: np.ndarray):
        self.array = array
        
    def __or__(self, operation: Callable[['npext'], 'npext']) -> 'npext':
        return operation(self)

    def __array__(self):
        return self.array

    def __repr__(self):
        return repr(self.array)

def resize(fx: float, fy: float) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        resized = cv.resize(arr, None, fx=fx, fy=fy)
        return npext(resized)
    return operation

def bin_threshold(thr_val: int, high_value: int) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        _, msk = cv.threshold(arr, thr_val, high_value, cv.THRESH_BINARY)
        return npext(msk)
    return operation

def to_gray() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        arr = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
        return npext(arr)
    return operation

def dilate() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        
        arr = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
        return npext(arr)
    return operation

def crop(r: Rect) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        arr = crop_image(arr, r)
        return npext(arr)
    return operation

def nz(r: Rect) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        arr = np.count_nonzero(arr)
        return npext(arr)
    return operation

def nz_mask() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        if arr.ndim == 3:
            msk = ((arr != (0, 0, 0)) * 255).astype(np.uint8)
        elif arr.ndim == 2:
            msk = ((arr != 0) * 255).astype(np.uint8)
        else:
            raise RuntimeError('unexpected number of dimensions')
        return npext(msk)
    return operation
    
def bgr2rgb() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        assert arr.ndim == 3
        out = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
        return npext(out)
    return operation

def gray2rgb() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        assert arr.ndim == 2
        out = cv.cvtColor(arr, cv.COLOR_GRAY2RGB)
        return npext(out)
    return operation
