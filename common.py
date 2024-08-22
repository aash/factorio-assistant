import numpy as np
import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from copy import deepcopy
import time
import contextlib
import itertools
import queue
import ahk as autohotkey
import logging
import win32api
from sklearn.cluster import KMeans
from npext import *
from graphics import *

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


def time_range(dur: float):
    t0 = time.time()
    i = 0
    grid = [t0]
    while True:
        if len(grid) > 20:
            grid.pop(0)
        t = time.time()
        grid.append(t)
        if len(grid) > 1:
            diffs = [b-a for b, a in list(zip(grid[1:], grid[:-1]))]
            avg_time = sum(diffs) / len(diffs)
        i += 1
        if time.time() - t0 > dur:
            break
        fps = 0.0 if len(grid) < 2 or avg_time == 0 else 1 / avg_time
        yield t, fps, i

class timer_unit(Enum):
    SECOND = 1
    MILLISECOND = 2

@contextlib.contextmanager
def timer(unit: timer_unit = timer_unit.SECOND):
    if unit is timer_unit.SECOND:
        t0 = time.time()
        yield lambda : time.time() - t0
    elif unit is timer_unit.MILLISECOND:
        t0 = int(1000*time.time())
        yield lambda : int(1000*time.time()) - t0

@contextlib.contextmanager
def timer_sec():
    t0 = time.time()
    yield lambda : time.time() - t0

@contextlib.contextmanager
def timer_ms():
    t0 = int(1000*time.time())
    yield lambda : int(1000*time.time()) - t0

def cart_prod(x, y):
    return list(itertools.product(x, y))

def grid(vl: np.ndarray, hl: np.ndarray) -> np.ndarray:
    return np.array([[(v, h) for h in hl] for v in vl])

def hstack(imgs):
    maxh = max([i.shape[0] for i in imgs])
    out_imgs = []
    for img in imgs:
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img = np.vstack((img, np.zeros((maxh - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)))
        out_imgs.append(img)
    return np.hstack(out_imgs)


@contextlib.contextmanager
def exit_hotkey(key = '^q', ahk = None):
    q = queue.Queue()
    if ahk is None:
        ahk = autohotkey.AHK()
    ahk.add_hotkey(key, lambda: q.put('exit'), logging.info('exit hotkey handler'))
    ahk.start_hotkeys()
    def get_command():
        if not q.empty():
            return q.get()
        return None
    yield get_command
    ahk.stop_hotkeys() 

@contextlib.contextmanager
def hotkey_handler(key, cmd):
    q = queue.Queue()
    ahk = autohotkey.AHK()
    logging.info(f'adding new hotkey {key} {cmd}')
    ahk.add_hotkey(key, lambda: q.put(cmd), logging.info(f"{cmd} command triggered"))
    ahk.start_hotkeys()
    def get_command():
        if not q.empty():
            cmd = q.get()
            logging.info(f'hotkey triggered {cmd}')
            return cmd
        return None
    yield get_command
    ahk.stop_hotkeys()

@dataclass
class point2d:
    xy: np.ndarray
    def __call__(self, inv: bool = False):
        if inv:
            return np.array((self.xy[1], self.xy[0]))
        return self.xy
    @classmethod
    def fromndarray(cls, arr: np.ndarray):
        assert arr.shape == (2, )
        return cls(arr) 
    @classmethod
    def fromxy(cls, x: int, y: int):
        return cls(np.array((x, y)))

@dataclass
class cell_loc:
    xy: np.ndarray
    def __call__(self, inv: bool = False):
        if inv:
            return np.array((self.xy[1], self.xy[0]))
        return self.xy
    def from_char_loc(loc: point2d, grid_width: int):
        return cell_loc(loc() // grid_width)

@contextlib.contextmanager
def timeout(tsec: float):
    t0 = time.time()
    def is_not_timeout():
        return time.time() - t0 < tsec
    yield is_not_timeout
def mixin(dst: np.ndarray, src: np.ndarray, alpha: float) -> np.ndarray:
    assert 0 < alpha <= 1.0
    dst = cv.addWeighted(dst, alpha, src, 1.0 - alpha, 0, dst)
    return dst

def get_midpoint(im: np.ndarray) -> point2d:
    return point2d.fromxy(im.shape[1] // 2, im.shape[0] // 2)

def strip_zeros_2d(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")
    
    if image.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")
    
    # Find rows and columns that are completely zero
    non_zero_rows = np.where(image.sum(axis=1) != 0)[0]
    non_zero_cols = np.where(image.sum(axis=0) != 0)[0]
    
    # If all rows or all columns are zero
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return np.array([[]])  # Return an empty 2D array
    
    # Determine the first and last non-zero row and column
    first_non_zero_row = non_zero_rows[0]
    last_non_zero_row = non_zero_rows[-1]
    first_non_zero_col = non_zero_cols[0]
    last_non_zero_col = non_zero_cols[-1]
    
    # Slice the array to remove zero rows and columns
    stripped_image = image[first_non_zero_row:last_non_zero_row + 1, first_non_zero_col:last_non_zero_col + 1]
    
    return stripped_image


def is_entity_tile(tile):
    entity_color = (0, 255, 0)
    out = cv.inRange(tile, entity_color, entity_color)
    ent_color_num = cv.countNonZero(out)
    return ent_color_num > 5

def get_closest(mvl, v):
    if len(mvl) < 1:
        return None
    dist = map(lambda x: tuple([x[0], abs(x[1] - v)]), enumerate(mvl))
    closest = min(dist, key=lambda x: x[1])
    return closest[0]


def margin(original_image, margin_width = 10):
    original_height, original_width = original_image.shape[:2]
    new_width = original_width + 2 * margin_width
    new_height = original_height + 2 * margin_width
    if len(original_image.shape) == 2:
        new_image = np.zeros((new_height, new_width), dtype=np.uint8)
    else:
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_image[margin_width:margin_width + original_height, margin_width:margin_width + original_width] = original_image
    return new_image

def flood_fill_contour(img, seed_point, mark_width = 15):
    sp = np.array(seed_point)
    ofs = np.array((mark_width, mark_width))
    loc = crop_image(img, Rect(*(sp - ofs), *(ofs*2)))
    img = loc
    h, w = img.shape[:2]
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    num, im, mask, rect = cv.floodFill(img, flood_fill_mask, ofs, 255)
    mask = (mask * 255).astype(np.uint8)
    mask[0,:] = 0
    mask[-1,:] = 0
    mask[:,0] = 0
    mask[:,-1] = 0
    y, x = np.where(mask > 0)
    sx, sy = sp
    sp_inv = np.array((sx,sy))
    contour_pixels = np.array(list(zip(x, y)))
    return np.array([p + sp_inv-ofs - (1,1) for p in contour_pixels])



def entity_pos(im, mark_size = 14):
    py, px = np.where(im > 0)
    pts = set(zip(px, py))
    ents = []
    while pts:
        sp = pts.pop()
        cp = flood_fill_contour(im, sp)
        r = Rect(*cv.boundingRect(cp))
        if r.w * r.h < (mark_size-1)**2:
            continue
        for x, y in cp:
            if (x, y) in pts:
                pts.remove((x, y))
        
        ents.append(r.center())
    return ents

def get_grid(im, threshold = 100, grid_color = (0,0,0)):
    def get_trilples(lst):
        triples = []
        if len(lst) >= 3:
            for i in range(len(lst)-2):
                if lst[i] + 1 == lst[i+1] and lst[i+1] + 1 == lst[i+2]:
                    triples.append((lst[i], lst[i+1], lst[i+2]))
        return triples
    def filter_triples(lst):
        l = lst
        ml = []
        for t in get_trilples(lst):
            ml.append(t[1])
            l.remove(t[0])
            l.remove(t[2])
        return l, ml
    def get_diffs(l):
        return list(map(lambda x: x[0] - x[1], zip(l[1:], l[:-1])))
    def get_most_frequent_value(l):
        counts = np.bincount(l)
        sorted_values = np.argsort(-counts)
        most_frequent_value = sorted_values[0]
        return most_frequent_value
    def find_a_for_max_count(arr, w):
        max_count = 0
        best_a = arr.min()  
        for a in arr:
            count = np.count_nonzero([(val - a) % w == 0 for val in arr])
            if count > max_count:
                max_count = count
                best_a = a
        return best_a

    out = im
    h, w, *_ = out.shape
    out = cv.inRange(out, grid_color, grid_color)
    # cv.imwrite('grid.png', out)
    r = []
    for ax, mx in zip([0, 1], [h, w]):
        s = np.sum(out/255, axis=ax)
        lines = list(np.where(np.abs(s - mx) <= threshold)[0])
        l, ml = filter_triples(lines)
        r.append((l, ml))
    
    # get lengths between consequtive grid lines
    # logging.info(f'vl: {r[0][0]}\nhl: {r[1][0]}')
    vdiffs = get_diffs(r[0][0])
    hdiffs = get_diffs(r[1][0])
    # most probable grid size
    # logging.info(f'vdiffs: {vdiffs}\nhdiffs: {hdiffs}')
    grid_size = get_most_frequent_value(vdiffs + hdiffs)
    # find the most probable starting point
    vstart = find_a_for_max_count(np.array(r[0][0]), grid_size)
    hstart = find_a_for_max_count(np.array(r[1][0]), grid_size)
    vn = list(range(vstart, w, grid_size))
    hn = list(range(hstart, h, grid_size))

    # logging.info(f'vstart: {vstart}, hstart: {hstart}, grid_size: {grid_size}')
    # logging.info(f'vn: {vn}\nhn: {hn}')



    return vn, hn, r[0][1], r[1][1], grid_size


class Grid:
    
    def get_grid_cell(absolute_x: int, absolute_y: int):
        pass


class MarkDirection(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4

def detect_mark_direction(cell) -> MarkDirection:
    if len(cell.shape) == 2:
        h, w = cell.shape
    else:
        h, w, _ = cell.shape
    # w = min(w, h)
    p = w // 2
    rs = [Rect(0, 0, p, p),
          Rect(0, h-1-p, p, p),
          Rect(w-1-p, 0, p, p),
          Rect(w-1-p,h-1-p, p, p)]
    s = [crop_image(cell, r) for r in rs]
    bf = tuple([np.count_nonzero(s_) < 2 for s_ in s])
    if bf.count(True) != 1:
        return None
    dct = {
        (False, True, False, False): MarkDirection.TOP_RIGHT,
        (False, False, True, False): MarkDirection.BOTTOM_LEFT,
        (False, False, False, True): MarkDirection.TOP_LEFT,
        (True, False, False, False): MarkDirection.BOTTOM_RIGHT
    }
    return dct[bf]


def get_marks(im1, im2):
    r = cv.bitwise_xor(im1, im2)
    r = cv.cvtColor(r, cv.COLOR_BGR2GRAY)
    _, r = cv.threshold(r, 0, 255, cv.THRESH_BINARY)
    h, w = r.shape
    cv.rectangle(r, (0,0), (w, h), 0, 5)
    i = 1
    cross_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    r = cv.erode(r, cross_kernel, iterations=i)
    # lab_img = cv.cvtColor(im1, cv.COLOR_BGR2LAB)
    # lower_red = np.array([20, 150, 150])  # Example values for lower bound
    # upper_red = np.array([255, 255, 255])  # Example values for upper bound
    # mask = cv.inRange(lab_img, lower_red, upper_red)
    # mask = cv.bitwise_and(mask, r)
    return r

def is_single_cell_entity(cell: np.ndarray):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cell, connectivity=4)
    if num_labels != 5:
        return False
    md = set()
    for c in range(1, num_labels):
        l = ((labels == c) * 255).astype(np.uint8)
        r = cv.boundingRect(l)
        cimg = crop_image(cell, Rect(*r))
        md.add(detect_mark_direction(cimg))
    return len(md) == 4


def get_ccs(img, with_br:bool = False):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=4)
    ccs = []
    for c in range(1, num_labels):
        l = ((labels == c) * 255).astype(np.uint8)
        r = cv.boundingRect(l)
        cimg = crop_image(img, Rect(*r))
        if with_br:
            ccs.append((cimg, r))
        else:
            ccs.append(cimg)
    return ccs

@dataclass
class entity:
    top_left: np.ndarray
    size: np.ndarray

def get_entity_coords_from_marks(mask: np.ndarray, im1, grid_color=(0,0,0)):
    assert mask.dtype in [np.uint8, np.int8]
    assert len(mask.shape) == 2
    vl, hl, mvl, mhl, cell_width = get_grid(im1, grid_color=grid_color)
    g = grid(vl, hl)
    def get_cell(im, g, gi, cell_width):
        return crop_image(im, Rect(*g[gi], cell_width, cell_width))
    gw, gh, _ = g.shape
    non_empty_cells = dict()
    for i in range(gw):
        for j in range(gh):
            cell = get_cell(mask, g, (i,j), cell_width)
            if np.count_nonzero(cell) != 0:
                non_empty_cells[(i,j)] = cell
                
    ents = []
    entities = []

    for k, v in non_empty_cells.items():
        x, y = g[k]
        # dis(v)
        if np.count_nonzero(v) < 20:
            continue
        if is_single_cell_entity(v):
            ents.append(np.array([x + cell_width // 2, y + cell_width // 2]))
            e = entity(k, np.array([1,1]))
            entities.append(e)
            # print(e)
        else:
            ccs = get_ccs(v)
            if len(ccs) == 1:
                d = detect_mark_direction(ccs[0])
                if d == MarkDirection.TOP_LEFT:
                    # then look into the cell on the right,
                    # check it's TOP_RIGHT orientation
                    i, j = k
                    ii = 1
                    while (i+ii, j) not in non_empty_cells:
                        if i+ii >= gw:
                            break
                        ii += 1
                    if (i+ii, j) in non_empty_cells:
                        rgt = non_empty_cells[(i+ii, j)]
                        rgt = strip_zeros_2d(rgt)
                        rgt_dir = detect_mark_direction(rgt)
                        if rgt_dir != MarkDirection.TOP_RIGHT:
                            logging.info(f'found marking is not expected rgt_dir={rgt_dir}')
                        else:
                            trf = True
                    else:
                        logging.info('could not find top right marking')
                        continue
                    jj = 1
                    while (i, j+jj) not in non_empty_cells:
                        if j + jj >= gh:
                            break
                        jj += 1
                    if (i, j+jj) in non_empty_cells:
                        btm = non_empty_cells[(i, j+jj)]
                        btm = strip_zeros_2d(btm)
                        btm_dir = detect_mark_direction(btm)
                        if btm_dir != MarkDirection.BOTTOM_LEFT:
                            logging.info(f'found marking is not expected btm_dir={btm_dir}')
                        else:
                            blf = True
                    else:
                        logging.info('could not find bottom left marking')
                        continue
                    if trf and blf:
                        if (i+ii, j+jj) not in non_empty_cells:
                            logging.info(f'diagonal mark is not present')
                            continue
                        diag = non_empty_cells[(i+ii, j+jj)]
                        diag = strip_zeros_2d(diag)
                        diag_dir = detect_mark_direction(diag)
                        if diag_dir != MarkDirection.BOTTOM_RIGHT:
                            logging.info(f'diagonal marking is unexpected diag_dir={diag_dir}')
                            continue
                    ents.append(np.array([x + cell_width * (ii + 1) // 2, y + cell_width * (jj + 1) // 2]))
                    e = entity(k, np.array([ii, jj]))
                    entities.append(e)
            elif len(ccs) == 2:
                dirs = set([detect_mark_direction(c) for c in ccs])
                if dirs == {MarkDirection.TOP_LEFT, MarkDirection.TOP_RIGHT}:
                    ents.append(np.array([x + cell_width//2, y + cell_width//2]))
                    e = entity(k, np.array([1,2]))
                    entities.append(e)
                if dirs == {MarkDirection.TOP_LEFT, MarkDirection.BOTTOM_LEFT}:
                    ents.append(np.array([x + cell_width//2, y + cell_width//2]))
                    e = entity(k, np.array([2,1]))
                    entities.append(e)
    return ents, entities

from PIL import Image as PILImage  # For converting to PIL Image for display
from IPython.display import display
def dis(*imgs):
    l = list(map(PILImage.fromarray, imgs))
    display(*l)


def putOutlinedText(im, s, p, c = (255,255,255), sz = 1):
    cv.putText(im, s, p, cv.FONT_HERSHEY_SIMPLEX, sz, (0,0,0), thickness=4, lineType=cv.LINE_AA)
    cv.putText(im, s, p, cv.FONT_HERSHEY_SIMPLEX, sz, c, thickness=1, lineType=cv.LINE_AA)
 
def get_screen_rect():
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)
    return Rect(0, 0, screen_width, screen_height)

@contextlib.contextmanager
def zoom_and_restore(ahk, n: int, sleep_time: float):
    for i in range(n):
        ahk.send('{WheelUp}')
        time.sleep(sleep_time)
    yield
    for i in range(n):
        ahk.send('{WheelDown}')
        time.sleep(sleep_time)

def get_cell_at(img: np.ndarray, p: np.ndarray, w: int) -> np.ndarray:
    return crop_image(img, Rect(*p, w, w))


def get_dominant_colors(image, k=5):
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape((-1, 3))

    # Apply KMeans to find the top k colors
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    # The cluster centers are our dominant colors.
    colors = kmeans.cluster_centers_

    return colors, kmeans.labels_

def recreate_image(centroids, labels, w, h):
    '''Recreate the (compressed) image from the cluster centers & labels'''
    d = centroids.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = centroids[labels[label_idx]]
            label_idx += 1
    return image


def get_reduced_img(img1, k = 6):
    dominant_colors, labels = get_dominant_colors(img1, k)

    # Recreate the image using the dominant colors
    w, h, _ = img1.shape
    new_image = recreate_image(dominant_colors, labels, w, h)

    # Convert the image back to BGR
    # new_image_bgr = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return new_image.astype(np.uint8)


def posterize(image, level):
    """
    Posterize an image to reduce the number of colors.

    :param image: Input image (numpy array).
    :param level: Level of posterization (number of different shades per channel).
    :return: Posterized image.
    """
    level = max(2, level)  # Ensure level is at least 2 to avoid zero-division
    # Calculate the quantization step
    step = 256 // level

    # Apply the quantization to each channel
    posterized_img = (image // step) * step + step // 2

    return posterized_img

def posterize_blk(image, level):
    """
    Posterize an image while preserving black colors.

    :param image: Input image (numpy array).
    :param level: Level of posterization (number of different shades per channel).
    :return: Posterized image with preserved black colors.
    """
    # Create a mask for the black regions
    black_mask = np.all(image == [0, 0, 0], axis=-1)

    # Calculate the quantization step
    step = 256 // max(2, level)  # Ensure level is at least 2 to avoid zero-division

    # Apply the quantization to each channel
    posterized_img = (image // step) * step + step // 2

    # Reapply the black color to the regions indicated by the mask
    posterized_img[black_mask] = [0, 0, 0]

    return posterized_img

def count_gray_pixels(image, threshold=1):
    
    # If the image is not read correctly
    if image is None:
        raise ValueError("Image not found or unable to read the image.")
    
    # Split the image into its color channels
    b, g, r = cv.split(image)
    
    # Calculate the absolute differences between the channels
    max_diff = np.max(np.array([b, g, r]), axis=0) - np.min(np.array([b, g, r]), axis=0)
    
    # Create a mask where the difference is below the threshold
    gray_mask = max_diff <= threshold
    
    # Count the number of gray pixels
    number_of_gray_pixels = np.sum(gray_mask)
    
    return number_of_gray_pixels

def count_gray_pixels1(image, threshold=10):
    # If the image is not read correctly
    if image is None:
        raise ValueError("Image not found or unable to read the image.")
    
    # Split the image into its color channels
    b, g, r = cv.split(image)
    
    # Calculate the absolute differences between the channels
    max_diff = np.max(np.array([b, g, r]), axis=0) - np.min(np.array([b, g, r]), axis=0)
    
    # Create a mask where the difference is below the threshold
    gray_mask = max_diff <= threshold
    
    # Count the number of gray pixels
    number_of_gray_pixels = np.sum(gray_mask)
    
    # Convert boolean mask to 8-bit format (0 or 255)
    gray_mask_uint8 = (gray_mask * 255).astype(np.uint8)
    
    return number_of_gray_pixels, gray_mask_uint8

def get_prevalent_color(image):
    reshaped_array = image.reshape(-1, image.shape[-1])

    # Get unique colors and their counts
    unique_colors, counts = np.unique(reshaped_array, axis=0, return_counts=True)

    # Create the dictionary with counts
    color_counts_dict = {tuple(color): count for color, count in zip(unique_colors, counts)}
    cc = max(color_counts_dict.items(), key=lambda x: x[1])
    # print(cc)
    return cc

map_corner_to_dir = {
    'top-left-ccw': 'down',
    'top-left-cw': 'right',
    'top-right-ccw': 'left',
    'top-right-cw': 'down',
    'bottom-left-ccw': 'right',
    'bottom-left-cw': 'up',
    'bottom-right-ccw': 'up',
    'bottom-right-cw': 'left',
}

def corner_to_dir(c):
    return map_corner_to_dir[c]

def find_contour(binary_img):
    # Find contours in the binary image
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the bracket
    max_contour = max(contours, key=cv.contourArea)
    
    return max_contour

def get_cell_to_entity_map(entities):
    map_c_to_e = {}
    for e in entities:
        map_c_to_e[e.top_left] = e.size
    return map_c_to_e

def classify_turn(img, iii, iset):
    img = img.astype(np.uint8)
    h, w, _ = img.shape
    ww = 30
    rect = Rect(0, 0, ww, ww)
    cimg = (npext(img) | crop(Rect(w//2 - ww//2, w//2-ww//2, ww,ww))).array
    www = 12
    dd = {
        Rect(rect.x0, rect.y0, www, www): 'top-left',
        Rect(rect.x0 + rect.w - www, rect.y0, www, www): 'top-right',
        Rect(rect.x0, rect.y0 + rect.h - www, www, www): 'bottom-left',
        Rect(rect.x0 + rect.w - www, rect.y0 + rect.h - www, www, www): 'bottom-right'
    }

    def nz_mask(img):
        msk = np.where(np.any(img != (0, 0, 0), axis=-1), 255, 0).astype(np.uint8)
        return msk

    def classify_ints(integers):
        def max_dev(integers):
            sorted(integers)
            m = np.mean(integers)
            d = np.abs(integers - m)
            return np.max(d)
        ints = sorted(integers) 
        md3 = max_dev(ints[1:])
        md = max_dev(ints)
        if md3 < 30 and md > 20:
            return 'corner'
        else:
            return 'straight'

    rects = cimg.copy()
    for k, v in dd.items():
        cv.rectangle(rects, k.xy(), np.array(k.xy()) + k.wh(), (255, 0, 0), 1)


    cnts = [cv.countNonZero(nz_mask(crop_image(cimg, k))) for k, v in dd.items()]

    if iii in iset:
        print(cnts)

    if classify_ints(cnts) == 'straight':
        return None, rects, cnts
    else:
        tpl = min(dd.items(), key=lambda tpl: cv.countNonZero(nz_mask(crop_image(cimg, tpl[0]))))
        return tpl[1], rects, cnts

def classify_image(img, iii = None, iset = {}):
    assert len(img.shape) == 3
    w = 32
    t, _, _ = classify_turn(img, iii, iset)
    p = 20
    q = 20 // 4
    def cmp(img, r1, r2, s1, s2):
        i1 = npext(img) | crop(r1) | to_gray() | bin_threshold(60, 255)
        i2 = npext(img) | crop(r2) | to_gray() | bin_threshold(60, 255)
        return s1 if cv.countNonZero(i1.array) < cv.countNonZero(i2.array) else s2
    img_orig = img.copy()     
    if t is None:
        
        # get a subimage of 12x12 size inside original, because on the
        # edges it can contain noise
        ww = 12
        r = Rect(w//2-ww//2, w//2-ww//2, ww, ww)
        img = crop_image(img, r)
        # img_post = posterize_blk(img, 6)
        
        # get most common color and paint it black
        cc, cnt = get_prevalent_color(img)
        # img = img_post
        msk = np.all(img == cc, axis=-1)
        img[msk] = (0, 0, 0)
        
        # flatten image into binary mask, it should coninatin
        # only arrow by which we deduce belt direction
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        _, img = cv.threshold(img, 30, 255, cv.THRESH_BINARY)
        
        contour = find_contour(img)
        if iii in iset:
            dis(img, img_orig)

        # depending on dimensions of arrow bounding box decide
        # the direction of belt
        brect = cv.boundingRect(contour)

        if brect[2] > brect[3]:
            r1 = Rect(w//2 - p//2, 1, p, q-1)
            r2 = Rect(w//2 - p//2, w-q, p, q-1)
            return cmp(img_orig, r1, r2, 'down', 'up')
        else:
            r1 = Rect(1, w//2 - p//2, q-1, p)
            r2 = Rect(w-q, w//2 - p//2, q-1, p)
            return cmp(img_orig, r1, r2, 'right', 'left')
    else:
        if t == 'top-left':
            r1 = Rect(w//2 - p//2, w - q, p, q-1)
            r2 = Rect(w - q, w//2 - p//2, q-1, p)
            return cmp(img_orig, r1, r2, 'top-left-cw', 'top-left-ccw')
        if t == 'top-right':
            r1 = Rect(w//2 - p//2, w - q, p, q-1)
            r2 = Rect(1, w//2 - p//2, q-1, p)
            return cmp(img_orig, r1, r2, 'top-right-ccw', 'top-right-cw')
        if t == 'bottom-right':
            r1 = Rect(1, w//2-p//2, q-1, p)
            r2 = Rect(w//2 - p//2, 1, p, q-1)
            return cmp(img_orig, r1, r2, 'bottom-right-ccw', 'bottom-right-cw')
        if t == 'bottom-left':
            r1 = Rect(w-q, w//2-p//2, q-1, p)
            r2 = Rect(w//2 - p//2, 1, p, q-1)
            return cmp(img_orig, r1, r2, 'bottom-left-cw', 'bottom-left-ccw')

def analyze_bracket_orientation(contour):
    # Calculate the bounding box and rotation
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.intp(box)
    
    # Calculate the extent of the contour (to differentiate between '(' and ')')
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # Center of mass
    M = cv.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return aspect_ratio, (cX, cY), box

def classify_bracket(aspect_ratio, center, box):
    # Basic heuristic: 
    # '(' tends to have its mass center slightly on the right,
    # ')' tends to have its mass center slightly on the left.
    delta_x = center[0] - np.mean(box[:, 0])
    delta_y = center[1] - np.mean(box[:, 1])
    # print(center[0], np.mean(box[:, 0]), center[1], np.mean(box[:, 1]))
    # print(delta_y)
    if abs(delta_x) < 0.1:
        return delta_y > 0
    else:
        return delta_x > 0

def qweqwe(img11):
    img = img11.astype(np.uint8)
    img = posterize_blk(img, 3)
    cc, cnt = get_prevalent_color(img)
    msk = np.all(img == cc, axis=-1)
    img[msk] = (0, 0, 0)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, img = cv.threshold(img, 20, 255, cv.THRESH_BINARY)

    contour = find_contour(img)
    aspect_ratio, center, box = analyze_bracket_orientation(contour)
    imgc = img.copy()
    imgc = cv.cvtColor(imgc, cv.COLOR_GRAY2RGB)
    imgc[center] = (255,0,0)
    dis(imgc)
    return classify_bracket(aspect_ratio, center, box)


def draw_lines_between_points(image, points, color=(0, 255, 0), thickness=1):
    """
    Draw lines between a sequence of points on an image.

    :param image: The image on which to draw the lines.
    :param points: A list of (x, y) coordinates.
    :param color: The color of the lines in BGR format.
    :param thickness: The thickness of the lines.
    """
    # Draw lines between consecutive points
    # pts = points + [points[0]]
    pts = points.tolist() + [points[0].tolist()]
    # print(pts)
    for i in range(len(pts) - 1):
        cv.line(image, pts[i], pts[i+1], color, thickness)

def midpoint(p1, p2):
    """
    Calculate the midpoint of the line segment between points p1 and p2.

    :param p1: Tuple (x1, y1) representing the first point.
    :param p2: Tuple (x2, y2) representing the second point.
    :return: Tuple (mx, my) representing the midpoint.
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def distance(p1, p2):
    """
    Calculate the distance between points p1 and p2.

    :param p1: Tuple (x1, y1) representing the first point.
    :param p2: Tuple (x2, y2) representing the second point.
    :return: Distance between p1 and p2.
    """
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def find_dividing_line(rect):
    """
    Find the line dividing the rectangle into two equal parts across the thin side.

    :param rect: List of four tuples representing the vertices of the rectangle in order.
    :return: Tuple of two tuples representing the midpoints of the shorter sides.
    """
    # Extract the vertices
    A, B, C, D = rect
    
    # Calculate the lengths of the sides
    side_lengths = [
        (distance(A, B), (A, B)),
        (distance(B, C), (B, C)),
        (distance(C, D), (C, D)),
        (distance(D, A), (D, A))
    ]
    
    # Sort the sides by length
    side_lengths.sort(key=lambda x: x[0], reverse=True)
    
    # The first two elements are the shorter sides
    shorter_side1 = side_lengths[0][1]
    shorter_side2 = side_lengths[1][1]
    
    # Calculate the midpoints of the shorter sides
    midpoint1 = midpoint(shorter_side1[0], shorter_side1[1])
    midpoint2 = midpoint(shorter_side2[0], shorter_side2[1])
    
    return midpoint1, midpoint2

def rotate_image(image, angle):
    # Get the image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply the rotation
    rotated_image = cv.warpAffine(image, M, (w, h))
    return rotated_image


directions = {
    'r': (0, 1),
    'l': (0, -1),
    'u': (-1, 0),
    'd': (1, 0)
}

reverse_directions = {
    'r': (0, -1),
    'l': (0, 1),
    'u': (1, 0),
    'd': (-1, 0)
}

# Remove leading and trailing whitespace, split by lines, and replace '_' with ' ' for empty spaces
def get_strmap_to_grid(map_str: str):
    lines = map_str.strip().split('\n')
    grid = [list(line.replace('_', ' ')) for line in lines]
    return grid

# Function to check if a cell has a belt
def is_belt(grid, row, col):
    return 0 <= row < len(grid) and 0 <= col < len(grid[row]) and grid[row][col] in directions

# Function to find the sink for a given source
def find_sink(grid, row, col):
    while is_belt(grid, row, col):
        direction = grid[row][col]
        dr, dc = directions[direction]
        new_row, new_col = row + dr, col + dc
        if not is_belt(grid, new_row, new_col):
            break
        row, col = new_row, new_col
    return row, col


def get_source_sink(grid):
    sources = []

    # Traverse the grid to find all sources
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] in directions:
                has_incoming = False

                # Check for incoming belts
                for dir_key, (dr, dc) in reverse_directions.items():
                    incoming_row, incoming_col = row + dr, col + dc
                    if is_belt(grid, incoming_row, incoming_col) and grid[incoming_row][incoming_col] in directions:
                        direction_into_current = grid[incoming_row][incoming_col]
                        if directions[direction_into_current] == (-dr, -dc):
                            has_incoming = True
                            break

                if not has_incoming:
                    sources.append((row, col))

    source_sink_pairs = []

    # Find all (source, sink) pairs
    for source in sources:
        sink = find_sink(grid, *source)
        source_sink_pairs.append((source, sink))
    
    return source_sink_pairs

def follow(grid, p):
    p = np.array(p)
    g = np.array(grid)
    d = g[*p]
    dd = directions[d]
    p_ = np.array(p) + dd
    while g[*p_] == d:
        p_ += dd
    if g[*p_] != ' ':
        return [p] + follow(grid, p_)
    else:
        return [p, p_ - dd]

def strmap_to_paths(map_str):
    gr = get_strmap_to_grid(map_str)
    paths = []
    for pair in get_source_sink(gr):
        ga = np.array(gr)
        lst = follow(ga, pair[0])
        paths.append(lst)
    return paths


def get_foreground(bg, comp):
    im = comp.copy()
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    comp = comp.astype(np.float32)
    bg = bg.astype(np.float32)
    fg = (comp - bg)
    fg = np.clip(fg, 0, 255).astype(np.uint8)
    # fg = cv.cvtColor(fg, cv.COLOR_BGR2RGB)
    fg1 = posterize_blk(fg, 5)
    return fg1, fg

def get_cell_to_entity_map(entities):
    map_c_to_e = {}
    for e in entities:
        map_c_to_e[e.top_left] = e.size
    return map_c_to_e


def get_belt_map(vl, hl, gcw, map_c_to_e, fg):
    bmap = np.array([['_'] * len(hl)] * len(vl))
    w, ww = gcw, 20
    rr = Rect(w//2-ww//2,w//2-ww//2, ww, ww)
    for i, (x, y) in enumerate(itertools.product(vl, hl)):
        ii, jj = i // len(hl), i % len(hl)
        
        c = get_cell_at(fg, (x,y), gcw)
        if c.shape != (32, 32, 3):
            continue
        # cv.rectangle(fg, (x,y), (x+32, y+32), (255,0,0), 1)
        cw = crop_image(c, rr)
        if cw.shape != (20, 20, 3):
            continue
        msk = cv.cvtColor(cw, cv.COLOR_RGB2GRAY)
        # cw.emp
        _, msk = cv.threshold(msk, 20, 255, cv.THRESH_BINARY)
        # skip cell if there's no significant pixels
        if cv.countNonZero(msk) < 32 * 1:
            continue
        
        # putOutlinedText(fg, f'{i}', (x, y + 16), sz=0.35)
        col, cnum = get_prevalent_color(cw)
        # skip cell if gray is not prevalent color
        if i == 385:
            dis(c)
            cc = dilate(c, 1, cv.MORPH_ELLIPSE)
            dis(cc)
            print(col, cnum)
        if col != (25,25,25):
            continue
        # skip cell if not enough pixels are gray
        if cnum < 275:
            continue

        # skip cell if it not on cell to entity map
        if (ii, jj) not in map_c_to_e:
            continue
        sz = map_c_to_e[(ii, jj)]
        # skip cell if it corresponds to entity not of 1x1 size
        if sz.tolist() != [1,1]:
            continue

        # after all the filtering real cell processing code comes here
        d = classify_image(c)
        if d in map_corner_to_dir:
            d = corner_to_dir(d)
        bmap[jj][ii] = d[0]
        # t = classify_turn(c)
        # if t is None:
            # d = 'straight'
        # else:
            # d = t
        # putOutlinedText(fg, f'{d[0]}', (x+2, y + 16), sz=0.35)
    return bmap

def build_graph_from_map(bmap: str):

    bmap = bmap.strip('\n \t').split('\n')
    moves = {
        'u': (-1, 0),
        'd': (1, 0),
        'r': (0, 1),
        'l': (0, -1),
        '_': (0, 0)
    }

    allowed_moves = {
        'u': ('l', 'r', 'u'),
        'd': ('l', 'r', 'd'),
        'r': ('r', 'd', 'u'),
        'l': ('l', 'd', 'u'),
    }

    def is_valid(x, y):
        return 0 <= x < len(bmap) and 0 <= y < len(bmap[0]) and bmap[x][y] != '_'

    graph = {}
    for i in range(len(bmap)):
        for j in range(len(bmap[0])):
            if bmap[i][j] == '_':
                continue
            graph[(i, j)] = []
            i_, j_ = moves[bmap[i][j]]
            i1 = i + i_
            j1 = j + j_
            if not is_valid(i1, j1):
                continue
            if bmap[i1][j1] in allowed_moves[bmap[i][j]]:
                graph[(i,j)].append((i1, j1))


    return graph

def find_all_paths(graph):
    def dfs(node, path):
        # Append current node to the path
        path.append(node)
        
        # Check if the node has no outgoing edges (i.e., end of path)
        if node not in graph or not graph[node]:
            all_paths.append(path.copy())
        else:
            for next_node in graph[node]:
                dfs(next_node, path)
        
        # Remove the current node to backtrack
        path.pop()

    all_paths = []
    
    # Call DFS from each node that is not in the middle of a path
    # A node is a start node if it does not appear as a destination in the graph
    start_nodes = set(graph.keys()) - {node for edges in graph.values() for node in edges}
    for start_node in start_nodes:
        dfs(start_node, [])

    # Filter out subpaths
    non_subpaths = []
    
    for path in all_paths:
        is_subpath = False
        for other_path in all_paths:
            if path != other_path and path == other_path[:len(path)]:
                is_subpath = True
                break
        if not is_subpath:
            non_subpaths.append(path)
    
    return non_subpaths

def collapse_paths(smap, paths):
    smap = smap.strip('\n \t').split('\n')
    out = []
    for path in paths:
        out1 = [path[0]]
        pp = path[0]
        cnt = 1
        for p in path[1:]:
            a = smap[p[0]][p[1]]
            b = smap[pp[0]][pp[1]]
            if a != b:
                cnt = 1
                pp = p
                out1.append(p)
            else:
                cnt += 1
        if cnt > 1:
            out1.append(path[-1])
        out.append(out1)
    return out

