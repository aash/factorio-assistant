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

def nz() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        arr = np.count_nonzero(arr)
        return npext(arr)
    return operation

def nz_mask() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        arr = ext.array
        if arr.ndim == 3:
            msk = np.where(np.any(arr != (0, 0, 0), axis=-1), 255, 0).astype(np.uint8)
        elif arr.ndim == 2:
            msk = np.where(np.any(arr != 0, axis=-1), 255, 0).astype(np.uint8)
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

def posterize(level: int, preserve_black: bool = True) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        image = ext.array
        assert image.ndim == 3
        step = 256 // max(2, level)
        posterized_img = (image // step) * step + step // 2
        if preserve_black:
            black_mask = np.all(image == [0, 0, 0], axis=-1)
            posterized_img[black_mask] = [0, 0, 0]
        return npext(posterized_img)
    return operation

def bitwise_xor(other: npext) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        return npext(cv.bitwise_xor(ext.array, other.array))
    return operation

def apply_mask(mask: npext) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        return npext(cv.bitwise_or(ext.array, ext.array, mask=mask.array))
    return operation

def dilate(el, sz, it=1) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        cross_kernel = cv.getStructuringElement(el, (sz, sz))
        return npext(cv.dilate(ext.array, cross_kernel, iterations=it))
    return operation

def erode(el, sz, it=1) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        cross_kernel = cv.getStructuringElement(el, (sz, sz))
        return npext(cv.erode(ext.array, cross_kernel, iterations=it))
    return operation

def gaussian_blur(sz) -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        return npext(cv.GaussianBlur(ext.array, (sz,sz), 0, 0))
    return operation

def to_float32() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        return npext(ext.array.astype(np.float32))
    return operation

def to_uint8() -> Callable[[npext], npext]:
    def operation(ext: npext) -> npext:
        return npext(ext.array.astype(np.uint8))
    return operation
