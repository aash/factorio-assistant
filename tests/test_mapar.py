from packaging import version
import logging
from mapar import *
from mapar.snail import *
import ahk as autohotkey
import d3dshot_stub as d3dshot
import time
import datetime
import numpy as np
import cv2
import cv2 as cv
import sys, inspect
from copy import deepcopy
import pytest
from pytest import fail
#from scipy.ndimage import center_of_mass
from common import *
import zmq
import contextlib
import json
import threading
import multiprocessing, subprocess
import runpy
import os, signal
import queue

FACTORIO_WINDOW_NAME = 'Factorio'
AHK_BINARY_PATH = 'D:/portable/ahk/AutoHotkeyU64.exe'
D3DSHOT_1_0_0 = version.parse('1.0.0')

def millis_now():
    return int(time.time() * 1000)


def dump_images(img_var_names: List[str] | str, image_format: str = 'bmp', postfix: str = None):
    if type(img_var_names) == str:
        img_var_names = list(map(str.strip, img_var_names.split(',')))

    
    for var_name in img_var_names:
        dump_image(var_name, image_format, postfix)

def dump_image(img_var_name: str, image_format:str = 'bmp', postfix: str = None):
    assert type(img_var_name) == str
    assert image_format in ['bmp', 'png', 'jpg', 'jpeg']
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    if caller_name == 'dump_images':
        caller_name = inspect.currentframe().f_back.f_back.f_code.co_name
        caller_locals = caller_frame.f_back.f_locals
    else:
        caller_locals = caller_frame.f_locals
    assert img_var_name in caller_locals
    img = caller_locals[img_var_name]
    if not postfix:
        postfix = f'{img_var_name}'
    else:
        postfix = f'{img_var_name}_{postfix}'
    if type(img) == np.ndarray:
        cv.imwrite(f'tmp/{caller_name}_{postfix}.{image_format}', img)
    elif type(img) == npext:
        cv.imwrite(f'tmp/{caller_name}_{postfix}.{image_format}', img.array)
    else:
        raise RuntimeError(f'unexpected type of variable: type({img_var_name}) == {type(img)}')

def clear_images():
    import os, glob
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    for fn in glob.glob(f'tmp/{caller_name}*'):
        os.remove(fn)
    

@pytest.mark.skip('outdated')
def test_can_run_and_stop_dxgi_capture_binary():
    from subprocess import PIPE, Popen
    from time import sleep, time
    from os import linesep, path
    from psutil import pid_exists
    exe = D3DShot.EXECUTABLE_PATH
    exe = f'{path.abspath(exe)}'

    p = Popen([exe], stdin=PIPE, stdout=PIPE, text=True)
    #logging.info(f'start cmd /k {exe}')
    sleep(75.2)
    quit_command = 'q' + linesep
    p.stdin.write(quit_command)
    p.stdin.flush()
    o, e = p.communicate()
    logging.info(f'{o}')
    t0 = time()
    timeout = 5.0
    while p.poll() == None:
        if (time() - t0) > timeout:
            # cv.circle(non_ui_img, (vl[9], hl[4]), 3, (0,0,255), 1)
            raise RuntimeError('timeout')
        sleep(0.1)
    # s = p.stdout.read()
    assert p.poll() == 0
    assert not pid_exists(p.pid)
    # logging.info('process outout:')
    # logging.info(s)

@pytest.mark.skip('outdated')
def test_can_run_and_stop_dxgi_capture_binary1():
    from subprocess import PIPE, Popen
    from time import sleep, time
    from os import linesep, path
    from psutil import pid_exists
    import zmq
    exe = d3dshot.D3DShot.EXECUTABLE_PATH
    exe = f'{path.abspath(exe)}'
    p = Popen([exe], stdin=PIPE, stdout=PIPE, text=True)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")
    exit_command = 'exit'

    socket.send_string('echo')
    reply = socket.recv_string()
    assert reply == 'echo'
    sleep(0.5)

    socket.send_string(exit_command)
    reply = socket.recv_string()
    assert reply == exit_command
    socket.close()
    context.term()
    timeout = 5.0
    p.wait(timeout)
    s, e = p.communicate()
    logging.info(f'exitcode: {p.returncode}')
    assert p.returncode == 0
    assert not pid_exists(p.pid)
    logging.info('process outout:')
    logging.info(s)

def test_get_client_rect():
    ahk = autohotkey.AHK(executable_path='D:/tools/python310/Scripts/AutoHotkey.exe')
    r = MapParser.get_factorio_client_rect(ahk, FACTORIO_WINDOW_NAME)
    logging.info(f'client rect: {r}')
    logging.info(f'client rect: {tuple(r.values())}')
    def non_null_rect(r):
        r = DataObject(r)
        return r.width > 0 and r.height > 0
    assert non_null_rect(r)

def test_dxgi():
    import dxgi_screen_capture
    logging.info(dxgi_screen_capture)
    logging.info(dir(dxgi_screen_capture))
    x = dxgi_screen_capture.dxgisc()
    x.Init()


def test_get_d3dshot_version():
    logging.info(f'ver: {d3dshot.__version__}')
    assert version.parse(d3dshot.__version__)

@pytest.mark.skip('outdated')
def test_dxgi_capture_stop():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    tuple(r.values())
    robj = DataObject(r)
    d3d = d3dshot.D3DShot(capture_output=d3dshot.CaptureOutputs.NUMPY, fps=30, roi=Rect(*tuple(r.values())))
    d3d.capture()
    time.sleep(1)
    d3d.stop()

def test_dxgi_capture_get_next_frame_stop():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    tuple(r.values())
    robj = DataObject(r)
    d3d = d3dshot.D3DShot(capture_output=d3dshot.CaptureOutputs.NUMPY, fps=30, roi=Rect(*tuple(r.values())))
    try:
        logging.info('start get next frame')
        img, t = d3d.wait_next_frame()
        logging.info('end get next frame')
        dump_image('img')
        #cv.imwrite('tmp/test_dxgi_capture_get_next_frame_stop.png', img)
        logging.info(f'frame number: {t}')
    except Exception as e:
        fail(e)
    finally:
        d3d.stop()

def test_mapparser_getrate_capture():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    d3d = d3dshot.D3DShot(capture_output=d3dshot.CaptureOutputs.NUMPY, fps=60, roi=Rect(*tuple(r.values())))
    d3d.capture()
    n = 300
    t0 = millis_now()
    for i in range(n):
        img, t = d3d.wait_next_frame()
        logging.info(f'{i} new frame no: {t}')
        assert img.shape
        #cv2.imwrite(f'frame{i:06d}.bmp', img)
    dt = millis_now() - t0
    logging.info(f'time per frame: {dt / n} ms')
    logging.info(f'fps: {1000 * (n / dt)}')
    d3d.stop()

def test_snail():
    with Snail() as s:
        time.sleep(1)
        assert isinstance(s, Snail)

def test_coord_mode():
    with Snail() as s:
        logging.info(s.ahk.get_coord_mode('Mouse'))
        s.ahk.set_coord_mode('Mouse', 'Client')
        s.window.activate()
        s.ahk.mouse_move(1, 1)
        logging.info(s.ahk.get_coord_mode('Mouse'))


def test_d3dshot_wait_next_frame():
    with Snail() as s:
        s.window.activate()
        s.ahk.send_input('{F9}', blocking=True)
        # time.sleep(2)
        s.ahk.mouse_move(20,20)
        f0, t0 = s.d3d.wait_next_frame()
        logging.info(f'frame number: {t0}')
        cnt = 3
        for i in range(cnt):
            s.ahk.send_input('{WheelUp 1}', blocking=True)
            time.sleep(0.005)
        f1, t1 = s.d3d.wait_next_frame()
        logging.info(f'frame number: {t1}')
        for i in range(cnt):
            s.ahk.send_input('{WheelDown 1}', blocking=True)
            time.sleep(0.005)
        r = np.bitwise_xor(f0, f1)
        dump_image('r')
        dump_image('f0')
        dump_image('f1')
        logging.info(f'{t1 - t0}')
        assert True

def test_get_widgets():
    from mapar.snail import get_bounding_rects
    from common import Rect, label_brect
    with Snail() as s:
        s.window.activate()
        s.ahk.send_input('{F9}', blocking=True)
        #time.sleep(2)
        s.ahk.mouse_move(1, 1)
        f0, t0 = s.d3d.wait_next_frame()
        s.ahk.send_input('{WheelUp 1}', blocking=True)
        time.sleep(0.1)
        f1, t1 = s.d3d.wait_next_frame()
        s.ahk.send_input('{WheelDown 1}', blocking=True)
        dump_image('f0')
        dump_image('f1')
        c_composite = f0.copy()
        cols = get_palette(11)
        rects = get_bounding_rects(f0, f1)
        for i, brect in enumerate(rects):
            cv.rectangle(c_composite, brect.xywh(), cols[i])
            win = Rect(0, 0, *s.window_rect.wh())
            lbls = label_brect(brect, win)
            desc = str(i) + '-' + ','.join(map(str, lbls))
            cv.putText(c_composite, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=4, lineType=cv.LINE_AA)
            cv.putText(c_composite, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
        dump_image('c_composite')
        assert True


def test_get_central_widget():
    from common import Rect, label_brect, crop_image
    with Snail() as s:
        s.ahk.send_input('{F9}', blocking=True)
        time.sleep(1)
        r, f0, f1 = s.get_widget_brects()
        dump_image('f0')
        dump_image('f1')
        xored = cv.bitwise_xor(f0, f1)
        dump_image('xored')
        cols = get_palette(11)
        labels = f0.copy()
        for i, brect in enumerate(r):
            cv.rectangle(labels, brect.xywh(), cols[i%11])
            win = Rect(0, 0, *s.window_rect.wh())
            lbls = label_brect(brect, win)
            desc = str(i) + '-' + ','.join(map(str, lbls))
            cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=4, lineType=cv.LINE_AA)
            cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
        dump_image('labels')
        central_widget_brect = s.filter_brects(r)
        assert central_widget_brect
        logging.info(central_widget_brect.xywh())
        cwidget = crop_image(f1, central_widget_brect)
        dump_image('cwidget')
        h, w = cwidget.shape[:2]
        assert (w, h) == central_widget_brect.wh()

def test_get_non_ui_screen():
    from common import Rect, label_brect, crop_image
    from mapar.snail import WidgetType
    with Snail() as s:
        # s.ahk.send_input('{F9}', blocking=True)
        time.sleep(1)
        r, f0, f1 = s.get_widget_brects()
        char_brect = s.filter_brects(r, WidgetType.CHARACTER)
        qbar_brect = s.filter_brects(r, WidgetType.QUICKBAR)
        mmap_brect = s.filter_brects(r, WidgetType.MINIMAP)
        logging.info(f'character bar brect: {char_brect}')
        logging.info(f'quick bar brect: {qbar_brect}')
        logging.info(f'minimap brect: {mmap_brect}')
        w = s.window_rect.w - mmap_brect.w
        h = s.window_rect.h - max(char_brect.h, qbar_brect.h)
        non_ui_rect = Rect(0, 0, w, h)
        non_ui_img = crop_image(f0, non_ui_rect)
        dump_image('non_ui_img')
        assert w < s.window_rect.w and h < s.window_rect.h

def test_extract_grid_nodes():
    with Snail() as s:
        # nodes = get_grid(non_ui_img)
        non_ui_img = s.wait_next_frame(s.non_ui_rect)
        vl, hl, mvl, mhl, ds = get_grid(non_ui_img)
        w, h = s.non_ui_rect.wh()
        assert w / ds <= len(vl) + 1
        assert w / ds >= len(vl) - 1
        assert h / ds <= len(hl) + 1
        assert h / ds >= len(hl) - 1
        # logging.info(f'{nodes}')
        # logging.info(f'{ds}')
        dump_image('non_ui_img')


from common import grid
import random

@dataclass
class frame_desc:
    vl: np.ndarray
    hl: np.ndarray
    mvl: np.ndarray
    mhl: np.ndarray
    grid: np.ndarray
    image: np.ndarray
    edge: np.ndarray
    grid_width: int
    char_loc_cell=point2d.fromxy(0, 0)
    def get_nodes(self):
        return (self.vl, self.hl, self.mvl, self.mhl, self.grid_width)

def get_frame_desc(frame: np.ndarray) -> frame_desc:
    vl, hl, mvl, mhl, ds = get_grid(frame)
    edge = cv.Canny(frame, 150, 205)
    g = grid(vl, hl)
    return frame_desc(vl, hl, mvl, mhl, g, frame, edge, ds)

def get_cell_at(img: np.ndarray, p: np.ndarray, w: int) -> np.ndarray:
    return crop_image(img, Rect(*p, w, w))

def char_cells(gw, gh):
    st = [[0, 0], [0, -1], [1, 0], [1, -1], [-1, 0], [-1, -1], [-1, -2], [0, -2], [2, 0], [2, -1]]
    t = np.array((gh, gw)) // 2
    t = set(map(lambda x: tuple(t + x), st))
    return t

def match_two_frames(frame0: frame_desc, frame1: frame_desc, ani_mask: np.ndarray=None, base_node=None, frind: int = 0):
    assert frame0.grid_width == frame1.grid_width
    w = frame0.grid_width
    cell_char_coord = frame0.char_loc_cell.xy
    vl0, hl0, mvl0, mhl0, ds0 = frame0.get_nodes()
    vl1, hl1, mvl1, mhl1, ds1 = frame1.get_nodes()
    # select furthest cell across movement direction
    def is_in_rectangle(p: np.ndarray, r: Rect) -> bool:
        return (r.x0 <= p[0] <= r.x0 + r.w) and (r.y0 <= p[1] <= r.y0 + r.h)
    
    d = list(map(np.array, itertools.product([-1, 0, 1], [-1, 0, 1])))
    iw = len(vl0)
    ih = len(hl0)
    a = range(1, iw - 3)
    b = range(1, ih - 3)
    ind = list(map(np.array, itertools.product(a, b)))
    gw, gh = len(frame0.vl), len(frame0.hl)
    cc = char_cells(gw, gh)
    fh, fw, *_ = frame0.image.shape
    cc_ij = get_char_cell(frame0, point2d.fromxy(fw//2, fh//2))

    def is_animated_cell(amap, ij):
        amh, amw, *_ = amap.shape
        midp = np.array((amh, amw)) // 2
        ji = np.array([ij[1], ij[0]])
        return amap[*(midp + ji)] == 255

    if frame0.vl == frame1.vl and frame0.hl == frame1.hl:
        nzc_threshold = 0.1
        xored = cv.bitwise_xor(frame0.image, frame1.image)
        mask = np.all(xored != [0, 0, 0], axis=-1).astype(np.uint8) * 255
        am = np.zeros(shape=(gh, gw), dtype=np.uint8)
        for (ix, x) in enumerate(frame0.vl):
            for (iy, y) in enumerate(frame0.hl):
                if (ix, iy) in cc:
                    continue
                c = get_cell_at(mask, np.array((x, y)), w)
                nz = np.count_nonzero(c)
                am[iy][ix] = 255 if nz > nzc_threshold * w * w else 0
        # logging.info(f'frame grid shape: {am.shape}')
        # dump_image('am', f'_am_{frind:03d}')
        amh, amw, *_ = ani_mask.shape
        midp = np.array((amh, amw)) // 2
        am1 = ani_mask[midp[0]:midp[0] + gh, midp[1]:midp[1] + gw]
        sep = np.zeros(shape=(3,3), dtype=np.uint8)
        out_am = hstack([am, sep, am1])
        # dump_image('out_am', f'_am_{frind:03d}')


    # logging.info(f'cell char coord: {frind} {cell_char_coord}')

    # filter out animated cells
    ind_filtered = []
    for ij in ind:
        if not is_animated_cell(ani_mask, ij - cc_ij + cell_char_coord) and (tuple(ij) not in cc):
            ind_filtered.append(ij)

    random.shuffle(ind_filtered)
    if base_node is None or not is_in_rectangle(base_node, Rect.from_xyxy(a[0], b[0], a[-1], b[-1])):
        candidates = ind_filtered[:10]
    else:
        candidates = ind_filtered[:9] + [base_node]

    def is_cell_match(cell0, cell1, w):
        diff = cv.bitwise_xor(cell0, cell1)
        diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        c = cv.countNonZero(diff)
        return c / (w * w) <= 0.2
    
    canvas = np.zeros_like(frame0.image)
    for ij in candidates:
        p = frame0.grid[*ij]
        cv.rectangle(canvas, p, p + [w, w], (128,128,0), -1)
    im0 = frame0.image.copy()
    

    # im0 = cv.addWeighted(im0, 0.7, canvas, 0.3, 0, im0)
    # out_img = hstack([im0, frame1.image])
    # dump_image('out_img', f'_out_img_{frind:03d}')
         

    while len(candidates):
        ij0 = np.array(candidates.pop())
        cell0 = get_cell_at(frame0.image, frame0.grid[*ij0], w)
        for _d in d:
            ij1 = ij0 + _d
            cell1 = get_cell_at(frame1.image, frame1.grid[*ij1], w)
            # xored = cv.bitwise_xor(cell0, cell1)
            # img_out = np.concatenate((cell0, cell1, xored), axis=1)
            # cv.imwrite(f'tmp/cells_{ij0[0]}{ij0[1]}_{ij1[0]}{ij1[1]}.bmp', img_out)
            if is_cell_match(cell0, cell1, w):
                p0 = frame0.grid[*ij0]
                p1 = frame1.grid[*ij1]
                diff = p1 - p0
                # validate diff
                dcheck = True
                for ij in candidates[-3:]:
                    p = frame0.grid[*ij]
                    c0 = get_cell_at(frame0.image, p, w)
                    c1 = get_cell_at(frame1.image, p + diff, w)
                    if not is_cell_match(c0, c1, w):
                        # logging.info(f'failed double check {ij0} {ij1}')
                        r = hstack([c0, c1])
                        # r = np.concatenate((c0, c1), axis=1)
                        # dump_image('r', f'_r_fdcheck_{frind}_{hash(tuple(p))}')
                        dcheck = False
                        break
                if dcheck:
                    return ij0, ij1
    logging.info(f'started with {base_node} and ran out of cadidates')
    return None, None


def test_extract_grid_and_match():
    from mapar.snail import WidgetType, get_grid_rect, chk
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    # with Snail() as s:
    im0 = cv.imread('tmp/1/frame0_000085.bmp')
    im1 = cv.imread('tmp/1/frame1_000085.bmp')
    f0 = get_frame_desc(im0)
    f1 = get_frame_desc(im1)
    # logging.info(f'{len(f0.vl} {f0.vl} {f0.hl}')
    with timer_ms() as elapsed:
        ij0, ij1 = match_two_frames(f0, f1, np.array((10, 4)))
        if ij0 is None and ij1 is None:
            fail('no match')
        p00, p01 = f0.grid[*ij0], f0.grid[*(ij0 + (1,1))]
        p10, p11 = f1.grid[*ij1], f1.grid[*(ij1 + (1,1))]
        logging.info(f'{p00},{p01}')
        cv.rectangle(im0, p00, p01, (0,0,255), 1)
        cv.rectangle(im1, p10, p11, (0,0,255), 1)
    logging.info(f'elapsed {elapsed()}')
    qweqwe = f0.image
    dump_image('qweqwe')
    dump_image('im0')
    dump_image('im1')

"""                 cmv1 = next(filter(lambda x: abs(x[1] - char_offs[0]) < 15, enumerate(mvl1)), None)
                cmh1 = next(filter(lambda x: abs(x[1] - char_offs[1]) < 15, enumerate(mhl1)), None)
                if cmv1:
                    cmv0 = (cmv1[0], mvl0[cmv1[0]])
                    # logging.info(f'{char_offs[0]}, {list(enumerate(mvl0))}')
                    # logging.info(f'{char_offs[0]}, {list(enumerate(mvl1))}')
                        
                    if (cmv0[1] <= char_offs[0] <= cmv1[1]):
                        if i - cfidx[0] > 40:
                            gnindex[0] -= 1
                            logging.info(f'crossed from right to left')
                            cfidx[0] = i
                            cfidx[1] = 0
                    if (cmv1[1] <= char_offs[0] <= cmv0[1]):
                        if i - cfidx[1] > 40:
                            gnindex[0] += 1
                            logging.info(f'crossed from left to right')
                            cfidx[1] = i
                            cfidx[0] = 0
                    # logging.info(f'cmv: {cmv1}')
                if cmh1:
                    # cmh0 = next(filter(lambda x: abs(x[1] - char_offs[1]) < 10, enumerate(mhl0)), None)
                    cmh0 = (cmh1[0], mhl0[cmh1[0]])
                    if (cmh0[1] <= char_offs[1] <= cmh1[1]):
                        if i - cfidx[2] > 40:
                            gnindex[1] -= 1
                            logging.info(f'crossed from bottom to top')
                            cfidx[2] = i
                            cfidx[3] = 0
                    if (cmh1[1] <= char_offs[1] <= cmh0[1]):
                        if i - cfidx[3] > 40:
                            gnindex[1] += 1
                            logging.info(f'crossed from top to bottom')                    
                            cfidx[3] = i
                            cfidx[2] = 0
                    # logging.info(f'cmh: {cmh1}') 
 """


def test_xxx():
    with overlay_client() as ovl_show_img, Snail() as s:
        t0 = time.time()
        im = s.wait_next_frame(s.non_ui_rect)
        while True:
            pim = im
            im = s.wait_next_frame(s.non_ui_rect)
            xim = cv.bitwise_xor(pim, im)
            xim = (xim != (0,0,0)).astype(np.uint8) * 255
            ovl_show_img(xim)
            if time.time() - t0 > 20:
                break
            time.sleep(0.01)

def test_ahk_hotkeys():
    t0 = time.time()
    with exit_hotkey(key='+q') as get_command:
        while True:
            if get_command() == 'exit':
                break
            if time.time() - t0 > 20:
                break
            time.sleep(0.01)
            # logging.info(f'elapsed: {time.time() - t0}')

def test_extract_grid_and_match_continuous():
    with overlay_client() as ovl_show_img, Snail() as s, exit_hotkey(ahk=s.ahk) as cmd_get:
        fb = []
        t0 = time.time()
        dd = set()
        non_ui_img_prev = s.wait_next_frame(s.non_ui_rect)
        pij = None
        char_coords = np.array((0, 0))
        non_ui_img = s.wait_next_frame(s.non_ui_rect)
        char_offs = s.char_offset
        i = 0
        fb1 = []
        gnindex = np.array((0, 0))
        cfidx = [0, 0, 0, 0]
        minc = 10000000
        maxc = 0
        # Scale a value from one range to another range
        def scale_value(value, from_min, from_max, to_min, to_max):
            return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

        roi = Rect( *(s.char_offset - (200, 200)), 400, 400)
        im = crop_image(non_ui_img_prev, roi)
        edges = cv.cvtColor(cv.Canny(im, 120, 255), cv.COLOR_GRAY2BGR)
        while True:
            if cmd_get() == 'exit':
                break
            with timer_ms() as elapsed:
                pim = im
                im = s.wait_next_frame(roi)
                # cv.imwrite('tmp/1/frame_{i:06d}.bmp', im)
                # i += 1
                fd = get_frame_desc(im)
                # for x in fd.vl:
                    # cv.line(im, (x, 0), (x, roi.h), (0,0,255), 1)
                # for y in fd.hl:
                    # cv.line(im, (0, y), (roi.w, y), (0,0,255), 1)
                pedges = edges
                edges = cv.cvtColor(cv.Canny(im, 150, 205), cv.COLOR_GRAY2BGR)
                xored = cv.bitwise_xor(pim, im)
                edges_xored = cv.bitwise_xor(pedges, edges)
                # g = grid(fd.vl, fd.hl)
                out = im.copy()

                with timer_ms() as felapsed:
                    for x1, x0 in zip(fd.vl[1:], fd.vl[:-1]):
                        for y1, y0 in zip(fd.hl[1:], fd.hl[:-1]):
                            # assert x1-x0 == fd.width
                            # assert y1-y0 == fd.width
                            # logging.info(f'{x0} {x1} {y0} {y1}')
                            # break
                            cell = crop_image(xored, Rect.from_xyxy(x0, y0, x1, y1))
                            
                            cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                            nz = cv.countNonZero(cell)
                            hc, wc, *_ = cell.shape
                            # assert hc == wc == fd.width
                            if nz < 0.1 * fd.grid_width * fd.grid_width:
                                
                                cv.rectangle(out, (x0+3, y0+3), (x1-3, y1-3), (0,255,0), 1)
                            else:
                                pass
                                # cv.imwrite(f'tmp/1/{x0:06d}_{y0:06d}.bmp', cell)
                            cv.putText(out, str(nz), (x0+1, y0 + 10), 0, 0.25, (0,0,255), 1)
                    logging.info(f'felapsed: {felapsed()}')
                    
                logging.info(f'elapsed: {elapsed()}')
                composite_out = np.concatenate((out, edges, xored), axis=1)
                ovl_show_img(composite_out)
            time.sleep(0.030)
            if time.time() - t0 > 1000:
                break
            non_ui_img_prev = non_ui_img
            i += 1
        logging.info(f'maxcontrast: {maxc}, mincontrast: {minc}')


@dataclass
class cell_desc:
    coord: np.ndarray
    img: np.ndarray
    anim: bool

def is_char_moving(frame0: frame_desc, frame1: frame_desc) -> bool:
    xored = cv.bitwise_xor(frame0.image, frame1.image)
    nz = np.count_nonzero(xored == (0, 0, 0))
    h, w, *_ = frame0.image.shape
    return nz / (w * h) > 0.95

@dataclass
class world_map:
    cell_map: dict
    top_left: np.ndarray
    explored: np.ndarray

    def qwaqwa():
        pass 

def explore(pframe: frame_desc, frame: frame_desc):
    # starting position is (0, 0)
    # mark all surroundings
    # if too close to the edge add current screen to the map
    xored = np.bitwise_xor(frame.image, pframe.image)

    
    pass

def get_char_cell(frame1: frame_desc, char_offs: point2d) -> point2d:
    gw = len(frame1.vl)
    gh = len(frame1.hl)
    ij = [0, 0]
    if frame1.vl[gw // 2 - 2] <= char_offs.xy[0] < frame1.vl[gw // 2 - 1]:
        ij[0] = gw // 2 - 2
    elif frame1.vl[gw // 2 - 1] <= char_offs.xy[0] < frame1.vl[gw // 2]:
        ij[0] = gw // 2 - 1
    else:
        ij[0] = gw // 2
    if frame1.hl[gh // 2 - 2] <= char_offs.xy[1] < frame1.hl[gh // 2 - 1]:
        ij[1] = gh // 2 - 2
    elif frame1.hl[gh // 2 - 1] <= char_offs.xy[1] < frame1.hl[gh // 2]:
        ij[1] = gh // 2 - 1
    else:
        ij[1] = gh // 2
    return ij
    # return point2d.fromxy(frame1.vl[ij[0]], frame1.hl[ij[1]])

def test_get_player_cell_loc():
    with overlay_client() as ovl_show_img, Snail() as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
        timeout(100) as is_not_timeout:
        frame_size = 600
        frame_mid_point = point2d.fromxy(frame_size // 2, frame_size // 2)
        roi = Rect( *(s.char_offset - frame_mid_point()), frame_size, frame_size)
        im = s.wait_next_frame(roi)
        frame1 = get_frame_desc(im)
        w = frame1.grid_width
        i = 0
        # assert w == 32
        char_offs = frame_mid_point
        while is_not_timeout():
            if cmd_get() == 'exit':
                break
            with timer_ms() as elapsed:
                frame0 = frame1
                im = s.wait_next_frame(roi)
                frame1 = get_frame_desc(im)
                overlay = np.zeros_like(frame1.image)
                # p = np.array(frame1.grid[*(10, 10)])
                gw = len(frame1.vl)
                gh = len(frame1.hl)
                ij = get_char_cell(frame1, char_offs)
                p = frame1.grid[*ij]
                
                cv.rectangle(overlay, p, p + [w, w], (255, 0, 0), -1)
                out_img = frame1.image.copy()
                # cv.line(out_img, (frame1.vl[gw//2], 0), (frame1.vl[gw//2],frame_size), (255,255,255), 1)
                # cv.line(out_img, (frame1.vl[gw//2-1], 0), (frame1.vl[gw//2-1],frame_size), (255,255,255), 1)
                # cv.line(out_img, (0, frame1.hl[gh//2]), (frame_size, frame1.hl[gh//2]), (255,255,255), 1)
                # cv.line(out_img, (0, frame1.hl[gh//2-1]), (frame_size, frame1.hl[gh//2-1]), (255,255,255), 1)
                 
                cv.line(out_img, (frame_size//2, 0), (frame_size//2,frame_size), (0,255,0), 1)
                cv.line(out_img, (0, frame_size//2), (frame_size, frame_size//2), (0,255,0), 1)
                # cv.circle(out_img, s.char_offset, 3, (255, 255, 255), 1)
                mixin(out_img, overlay, 0.7)
                # out_img = cv.addWeighted(out_img, 0.7, overlay, 0.3, 0, out_img)
                composite_out = hstack([out_img])
                ovl_show_img(composite_out)
            time.sleep(0.010)
            i += 1


@contextlib.contextmanager
def track_dist(l, nzc_threshold = 0.1, grid_size = 32):
    ''' keeps track of animated cells, only update if character offset is larger than l
    animation mask for a cell with coordinates P contains 255 if that cell does not have
    animated portions, in other words it is static, 0 otherwise 
    '''
    prev_char_coords = cell_loc(None)
    map_size = 2048
    ani_map = np.zeros(shape=(map_size, map_size), dtype=np.uint8)
    mp = np.zeros(shape=(map_size * grid_size, map_size * grid_size), dtype=np.uint8)
    h, w, *_ = mp.shape
    midp_mp = np.array((h//2, w//2)) 
    h, w, *_ = ani_map.shape
    midp = point2d.fromxy(h//2, w//2)
    # t = frame_mid_point() // w
    st = [[0, 0], [0, -1], [1, 0], [1, -1], [-1, 0], [-1, -1], [-1, -2], [0, -2], [2, 0], [2, -1]]
    # st = list(map(lambda x: t + x, st))
 
    def g(f0: frame_desc, f1: frame_desc, char_coords: cell_loc):
        ''' f0, f1 -- consecutive frames
        char_coords -- character coordinates in cells (char_coordinates // cell_width)
        '''
        nonlocal prev_char_coords
        if prev_char_coords() is None:
            dist = 10 ** 10
        else:
            dist = np.linalg.norm(prev_char_coords.xy - char_coords.xy)
            # logging.info(f'{char_coords.xy} / {prev_char_coords.xy} / {dist} / {prev_char_coords.xy - char_coords.xy}')
        if dist > l:
            xored = np.bitwise_xor(f0.image, f1.image)
            mask = np.all(xored != [0, 0, 0], axis=-1).astype(np.uint8) * 255
            # logging.info(f'mask shape {mask.shape}')
            # dump_image('xored')
            # dump_image('mask')
            # xored_gray = cv.cvtColor(xored, cv.COLOR_BGR2GRAY)
            nz = np.count_nonzero(mask)
            h, w, *_ = xored.shape
            cc_ij = point2d.fromxy(*get_char_cell(f1, point2d.fromxy(w//2, h//2)))
            t = np.array((w // 2, h // 2)) // f1.grid_width
            t = set(map(lambda x: tuple(t + x), st))
            if f0.vl != f1.vl or f0.hl != f1.hl:
                return False
            fm = np.zeros(shape=(len(f1.hl), len(f1.vl)), dtype=np.uint8)
            w = f1.grid_width
            im = f1.image.copy()
            canvas = np.zeros_like(im)
            logging.info(f'update ani map {prev_char_coords()} {char_coords()}')
            for ix, x in enumerate(f1.vl[:-1]):
                for iy, y in enumerate(f1.hl[:-1]):
                    if (ix, iy) in t:
                        continue
                    cell = get_cell_at(mask, np.array((x, y)), w)
                    nz = np.count_nonzero(cell)
                    # logging.info(f'{nz}')
                    fm[iy][ix] = 255 if nz > nzc_threshold * w * w else 0
                    # if fm[iy][ix] == 255:
                        # cv.rectangle(tmp, (x+5, y+5), (x + w - 6, y + w - 6), (255,0,0), 1, cv.LINE_4)
                    cell_coords = point2d.fromxy(iy, ix)
                    ani_map[*(midp() - cc_ij(True) + cell_coords() + char_coords(True))] |= 255 if nz > nzc_threshold * w * w else 0
                    c1 = get_cell_at(f1.image, np.array((x, y)), w)
                    c1 = cv.cvtColor(c1, cv.COLOR_BGR2GRAY)
                    p = midp_mp + (cell_coords() + char_coords(True) - cc_ij(True)) * w
                    # logging.info(f'qwe {p}')
                    mp[p[0]:p[0]+w, p[1]:p[1]+w] = c1[:,:]
                    if nz > nzc_threshold * w * w:
                        p = f1.grid[*(ix, iy)]
                        cv2.rectangle(canvas, p, p + [w,w], (128,128,128), -1)
            for _t in t:
                p = f1.grid[*_t]
                cv2.rectangle(canvas, p, p + [w,w], (128,128,0), -1)
            im = cv.addWeighted(im, 0.7, canvas, 0.3, 0, im)
            char_loc_str = f'{char_coords.xy} / {prev_char_coords.xy}'
            cv.putText(im, char_loc_str, np.array((10,80)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=4, lineType=cv.LINE_AA)
            cv.putText(im, char_loc_str, np.array((10,80)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
             
            # outt = hstack([im, mask])
            # dump_image('outt', f'_outt_{-char_coords.xy[0]:+03d}_{-char_coords.xy[1]:+03d}')
            # dump_image('fm', f'_fm_{char_coords.xy[0]:03d}_{char_coords.xy[1]:03d}')
            # dump_image('ani_map')
            # logging.info('qweqweqwe')
            prev_char_coords.xy = char_coords.xy
            return True
        return False
    def get_ani_map():
        return ani_map, mp
    yield g, get_ani_map

def test_get_char_anim_cells():
    with overlay_client() as ovl_show_img, Snail() as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
         track_dist(5) as (tr, get_ani_map):
        frame_size = 400
        frame_mid_point = point2d.fromxy(frame_size // 2, frame_size // 2)
        roi = Rect( *(s.char_offset - frame_mid_point()), frame_size, frame_size)
        im = s.wait_next_frame(roi)
        frame = get_frame_desc(im)
        w = frame.grid_width
        w * w
        character_offset = cell_loc.from_char_loc(point2d.fromxy(0, 0), w)
        
        while True:
            if cmd_get() == 'exit':
                break
            im = s.wait_next_frame(roi)
            pframe = frame
            frame = get_frame_desc(im)
            # assert pframe.vl == frame.vl and pframe.hl == frame.hl
            # mark
            # select cell candidates
            tr(pframe, frame, character_offset)
            ani_map = get_ani_map()
            out = hstack((im, ani_map))
            ovl_show_img(out)
            time.sleep(0.010)

def test_extract_grid_and_match_continuous1():
    with overlay_client() as ovl_show_img, Snail() as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
         track_dist(3.0) as (tr, get_ani_map), timeout(100) as is_not_timeout:
        char_loc = point2d.fromxy(0, 0)

        frame_size = 600
        frame_mid_point = point2d.fromxy(frame_size // 2, frame_size // 2)
        roi = Rect( *(s.char_offset - frame_mid_point()), frame_size, frame_size)
        im = s.wait_next_frame(roi)
        frame1 = get_frame_desc(im)
        w = frame1.grid_width
        # default zoom level (F9)
        assert w == 32
        mcellw = 1024
        assert mcellw == 1024
        ij0 = ij1 = None
        i = 0
        t = frame_mid_point() // w
        st = [[0, 0], [0, -1], [1, 0], [1, -1], [-1, 0], [-1, -1], [-1, -2], [0, -2], [2, 0], [2, -1]]
        st = list(map(lambda x: t + x, st))
        char_loc_cell = point2d.fromxy(0, 0)
        char_loc_mcell = point2d.fromxy(0, 0)
        midpoint = np.array(roi.wh()) // 2
        ij = get_char_cell(frame1, point2d.fromxy(*midpoint))
        p = frame1.grid[*ij]
        char_loc = point2d.fromxy(*(midpoint - p))
        canvas = np.zeros_like(im)
        while is_not_timeout():
            if cmd_get() == 'exit':
                break
            with timer_ms() as elapsed:
                frame0 = frame1
                im = s.wait_next_frame(roi)
                frame1 = get_frame_desc(im)
                ani_map, *_ = get_ani_map()
                
                # check if we crossed major line
                mcellx_changed = False
                mcelly_changed = False
                if frame1.mvl and frame0.mvl:
                    mvli = get_closest(frame1.mvl, frame_mid_point.xy[0])
                    if abs(frame1.mvl[mvli] - frame_mid_point.xy[0]) < w:
                        if frame0.mvl[mvli] < frame_mid_point.xy[0] <= frame1.mvl[mvli]:
                            mcellx_changed = True
                            xdir = False
                            char_loc_mcell.xy[0] -= 1
                        elif frame1.mvl[mvli] < frame_mid_point.xy[0] <= frame0.mvl[mvli]:
                            mcellx_changed = True
                            xdir = True
                            char_loc_mcell.xy[0] += 1
                if frame1.mhl and frame0.mhl:
                    mhli = get_closest(frame1.mhl, frame_mid_point.xy[1])
                    if abs(frame1.mhl[mhli] - frame_mid_point.xy[1]) < w:
                        if frame0.mhl[mhli] < frame_mid_point.xy[1] <= frame1.mhl[mhli]:
                            mcelly_changed = True
                            ydir = False
                            char_loc_mcell.xy[1] -= 1
                        elif frame1.mhl[mhli] < frame_mid_point.xy[1] <= frame0.mhl[mhli]:
                            mcelly_changed = True
                            ydir = True
                            char_loc_mcell.xy[1] += 1
                 
                
                ij0, ij1 = match_two_frames(frame0, frame1, ani_map, ij1, i)
                overlay = canvas.copy()
                for _t in st:
                    p = frame1.grid[*_t]
                    cv2.rectangle(overlay, p, p + [w,w], (255,255,0), -1)
                # cv.addWeighted()

                if ij0 is not None and ij1 is not None:
                    p0 = frame0.grid[*ij0]
                    p1 = frame1.grid[*ij1]
                    diff = p0 - p1
                    if mcellx_changed:
                        xofs = 0 if xdir else 1
                        xdiff = frame1.mvl[mvli] - frame_mid_point.xy[0]
                        char_loc.xy[0] = mcellw * (char_loc_mcell.xy[0] + xofs) + xdiff
                    else:
                        char_loc.xy[0] += diff[0]

                    if mcelly_changed:
                        yofs = 0 if ydir else 1
                        ydiff = frame1.mhl[mhli] - frame_mid_point.xy[1]
                        char_loc.xy[1] = mcellw * (char_loc_mcell.xy[1] + yofs) + ydiff
                    else:
                        char_loc.xy[1] += diff[1]
                    char_loc_cell.xy = char_loc.xy // frame1.grid_width
                else:
                    logging.info(f'cant match {ij0} {ij1}')
                
                frame1.char_loc_cell.xy = char_loc_cell.xy
                
                # logging.info(f'info {i}')
                tr(frame0, frame1, char_loc_cell)
                out_img = im.copy()
                cv.circle(out_img, frame_mid_point() - char_loc(), 5, (255, 0, 0), 2)
                char_loc_str = f'{char_loc()} / {char_loc_cell()} / {char_loc_mcell()}'
                cv.putText(out_img, char_loc_str, np.array((10,80)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=4, lineType=cv.LINE_AA)
                cv.putText(out_img, char_loc_str, np.array((10,80)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1, lineType=cv.LINE_AA)
                ani_map, mp = get_ani_map()
                mixin(out_img, overlay, 0.7)
                # out_img = cv.addWeighted(out_img, 0.7, overlay, 0.3, 0, out_img)
                # mp = cv.resize(mp, (512,512))
                mp_midpoint = get_midpoint(mp)
                mp_view_wh = np.array([512, 512])
                p0 = mp_midpoint.xy + char_loc.xy - mp_view_wh // 2
                mp_view_rect = Rect.from_xyxy(*p0, *(p0 + mp_view_wh))
                mp_view = crop_image(mp, mp_view_rect)

                # am_view = crop_image(ani_map, map_view)
                # mp = np.zeros_like(frame1.image)

                composite_out = hstack([out_img, mp_view])
                ovl_show_img(composite_out)
            # need to sleep a bit, because if we call overlay api to often it will hiccup
            # time.sleep(0.010)
            i += 1
        ani_map, mpmp = get_ani_map()
        logging.info(f'map size: {mpmp.shape}')
        mp = strip_zeros_2d(mpmp)
        ani_map = strip_zeros_2d(ani_map)
        dump_image('ani_map')
        dump_image('mp')


def test_get_grid_continuous():
    from mapar.snail import WidgetType, get_grid_rect, get_grid
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence, cart_prod
    with Snail() as s:
        s.ensure_next_frame()
        r, f0, f1 = s.get_widget_brects()
        time.sleep(0.3)
        non_ui_rect = s.non_ui_rect(r)
        logging.info(f'non ui rect: {non_ui_rect}')

        i = 0
        fb = []
        t0 = time.time()
        dd = set()
        while True:
            t = time.time()
            fr = s.wait_next_frame()
            non_ui_img = crop_image(fr, non_ui_rect)
            with timer_ms() as elapsed:
                vl, hl, mvl, mhl, ds = get_grid(non_ui_img)
                g = cart_prod(vl, hl)
                dd.update(ds)
                logging.info(f'elapsed: {elapsed()}')

            # for n in g: 
            #     if n[1] in mhl and n[0] in mvl:
            #         cv.circle(non_ui_img, n, 5, (0, 0, 255), 2)
            #     elif n[1] in mhl:
            #         cv.rectangle(non_ui_img, (n[0], n[1] - 3), (n[0], n[1] + 3), (255, 0, 0), 2)
            #     elif n[0] in mvl:
            #         cv.rectangle(non_ui_img, (n[0] - 3, n[1]), (n[0] + 3, n[1]), (255, 0, 0), 2)
            #     else:
            #         cv.circle(non_ui_img, n, 2, (0, 255, 255), 1)

                # non_ui_img[*n] = (255, 255, 255)
            fb.append(non_ui_img)
            while time.time() - t < 0.017:
                time.sleep(0.005)
            if time.time() - t0 > 10:
                break
        logging.info(f'{dd}')
        logging.info(f'write frames to disk')
        for i, f in enumerate(fb):
            cv.imwrite(f'tmp/non_ui_img_{i:06d}.bmp', f)
            # dump_image('f', f'_non_ui_img_{i:06d}')



def test_get_tooltip():
    WIDGET_MINIMUM_AREA = 40 * 40
    from mapar.snail import get_bounding_rects
    from common import get_palette, Rect
    with Snail() as s:
        s.ahk.send_input('{F9}', blocking=True)
        sleep_time = 0.1
        def initialize():
            s.ahk.mouse_move(577, 220)
            time.sleep(sleep_time)
        def action():
            s.ahk.mouse_move(577, 252)
            time.sleep(sleep_time)
        def finalize():
            s.ahk.mouse_move(577, 220)
        im0, im1 = s.get_diff_image(action, initialize, finalize)
        brects = get_bounding_rects(im0, im1)
        dump_image('im0')
        dump_image('im1')
        xored = cv.bitwise_xor(im0, im1)
        dump_image('xored')
        labels = im1.copy()
        cols = get_palette(11)
        r = xored
        r = cv.cvtColor(r, cv.COLOR_RGB2GRAY)
        r = ((r > 0) * 255).astype(np.uint8)
        dump_image('r', '_r1')
        ds = 3
        element = cv.getStructuringElement(cv.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
        r = cv.dilate(r, element)
        r = ((r == 0) * 255).astype(np.uint8)
        r = cv.dilate(r, element)
        r = ((r == 0) * 255).astype(np.uint8)
        dump_image('r')
        contour, _ = cv.findContours(r, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        res = []
        for i, con in enumerate(contour):
            area = cv.contourArea(con)
            brect = cv.boundingRect(con)
            rect = Rect(*brect)
            if area > WIDGET_MINIMUM_AREA:
                res.append(rect)
        brects = res

        for i, brect in enumerate(brects):
            cv.rectangle(labels, brect.xywh(), cols[i%11])
            # win = Rect(0, 0, *s.window_rect.wh())
            #lbls = label_brect(brect, win)
            # desc = str(i) + '-' + ','.join(map(str, lbls))
            # cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=3)
            # cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        dump_image('labels')




def test_label_brect():
    from common import Rect, label_brect
    win = Rect(0, 0, 1920, 1080)
    r1 = Rect(702, 1024, 732, 56)
    r2 = Rect(0, 984, 188, 96)
    r3 = Rect(1664, 0, 256, 362)
    assert label_brect(r1, win) == set([UiLocation.HCENTER, UiLocation.BOTTOM])
    assert label_brect(r2, win) == set([UiLocation.LEFT, UiLocation.BOTTOM])
    assert label_brect(r3, win) == set([UiLocation.RIGHT, UiLocation.TOP])
    


def test_uilocation_conv():
    from common import UiLocation
    l = {UiLocation(3), UiLocation(5)}
    s = ','.join(sorted(map(str, l)))
    assert s == 'hcenter,top'

def test_parse_main_ui_caption():
    ui_img = cv.imread('tmp/test_get_central_widget_cwidget.png')
    hsv = cv.cvtColor(ui_img, cv.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    v = cv.GaussianBlur(v,(3,3),0)
    ret3,th3 = cv.threshold(v, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    ds = 3
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    r = cv.dilate(th3, element)
    #r = cv.erode(r, element)
    dump_image('v')
    dump_image('th3')
    dump_image('r')
    #cv.threshold(ui_img, )
    # strip 

def test_get_char_coords():
    pass


def test_ensure_next_frame():
    for _ in range(20):
        with Snail() as s:
            s.ensure_next_frame()
            assert s.wait_next_frame() is not None

def test_timer():
    from common import timer, timer_sec, timer_ms, timer_unit
    near_zero = 3/10 ** 3
    sleep_time_sec = 1.234
    with timer() as elapsed:
        time.sleep(sleep_time_sec)
        t = elapsed()
        logging.info(f'timer() elapsed: {t} seconds')
        assert abs(t - sleep_time_sec) < near_zero
    with timer(timer_unit.SECOND) as elapsed:
        time.sleep(sleep_time_sec)
        t = elapsed()
        logging.info(f'timer(SECOND) elapsed: {t} seconds')
        assert abs(t - sleep_time_sec) < near_zero
    with timer_sec() as elapsed:
        time.sleep(sleep_time_sec)
        t = elapsed()
        logging.info(f'timer_sec() elapsed: {t} seconds')
        assert abs(t - sleep_time_sec) < near_zero
    sleep_time_ms = int(sleep_time_sec * 1000)
    with timer(timer_unit.MILLISECOND) as elapsed:
        time.sleep(sleep_time_sec)
        t_ms = elapsed()
        logging.info(f'timer(MILLISECOND) elapsed: {t_ms} ms')
        assert abs(t_ms - sleep_time_ms) < 3
    with timer_ms() as elapsed:
        time.sleep(sleep_time_sec)
        t_ms = elapsed()
        logging.info(f'timer_ms() elapsed: {t_ms} ms')
        assert abs(t_ms - sleep_time_ms) < 3

def test_benchmark_tracking_ops():
    from mapar.snail import WidgetType
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    from common import timer_ms
    with Snail() as s:
        s.ensure_next_frame()
        load_cached_params = False
        if load_cached_params:
            non_ui_rect = Rect(x0=0, y0=0, w=1664, h=1063)
            char_offs = (961, 580)
            grid_size = 1024
        else:
            r, f0, f1 = s.get_widget_brects()
            time.sleep(0.3)
            non_ui_img = s.wait_next_frame()
            char_offs = s.get_char_coords(non_ui_img)
            logging.info(f'char entity offset: {char_offs}')
            non_ui_rect = s.non_ui_rect(r)
            logging.info(f'non ui rect: {non_ui_rect}')
            non_ui_img = crop_image(non_ui_img, non_ui_rect)
            im11 = non_ui_img
            dump_image('im11')
            pgn = s.find_grid_nodes(non_ui_img)
            assert len(pgn) > 1
            r = Rect.from_xyxy(*pgn[0], *pgn[-1])
            grid_size = max(r.w, r.h)
            logging.info(f'grid size: {grid_size}')
        current_quad = Rect(0, 0, 0, 0)
        dir = MoveDirection.UP
        char_x = 0
        char_y = 0
        assert non_ui_rect.w < grid_size * 2 and non_ui_rect.h < grid_size * 2
        seq_press = get_ahk_sequence(dir, KeyState.PRESS)
        seq_release = get_ahk_sequence(dir, KeyState.RELEASE)

        # full screen find
        im = s.wait_next_frame()
        non_ui_img = crop_image(im, non_ui_rect)
        dump_image('im')

        pgn = s.find_grid_nodes(non_ui_img)
        roi_wh = np.array((256, 256))
        get_roi = lambda x: Rect(*(x - roi_wh // 2), *roi_wh)
        roi_rect = get_roi(pgn[0])
        roi_img = crop_image(non_ui_img, roi_rect)
        
        # benchmark operations
        n = 100

        with timer_ms() as elapsed:
            for _ in range(n):
                s.find_grid_nodes(non_ui_img)
            logging.info(f'find_grid_nodes bench time (non_ui_img): {elapsed() / n} ms')

        with timer_ms() as elapsed:
            for _ in range(n):
                s.find_grid_nodes(roi_img)
            logging.info(f'find_grid_nodes bench time (roi_img): {elapsed() / n} ms')

        with timer_ms() as elapsed:
            for _ in range(n):
                non_ui_img = crop_image(non_ui_img, non_ui_rect)
                roi_img = crop_image(non_ui_img, roi_rect)
            logging.info(f'roi_crop_ms bench time: {elapsed() / n} ms')

from overlay_client import overlay_client

def test_overlay_client_exit_expect_exception():
    with pytest.raises(RuntimeError) as einfo:
        with overlay_client() as send_img:
            time.sleep(20)
            raise RuntimeError('qweqwe')


def test_overlay_client_video_feed():
    with overlay_client() as overlay_show_img, Snail() as snl:
        t0 = time.time()
        while True:
            img = snl.wait_next_frame(snl.non_ui_rect)
            overlay_show_img(img)
            if time.time() - t0 > 20:
                break
            time.sleep(0.010)

def test_explore_two_by_two_area():
    from mapar.snail import WidgetType, get_grid_rect
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    with Snail() as s:
        s.ensure_next_frame()
        load_cached_params = False
        if load_cached_params:
            non_ui_rect = Rect(x0=0, y0=0, w=1664, h=1063)
            char_offs = (961, 580)
            grid_size = 1024
        else:
            r, f0, f1 = s.get_widget_brects()
            time.sleep(0.3)
            non_ui_img = s.wait_next_frame()
            char_offs = np.array(s.get_char_coords(non_ui_img))
            logging.info(f'char entity offset: {char_offs}')
            non_ui_rect = s.non_ui_rect(r)
            logging.info(f'non ui rect: {non_ui_rect}')
            non_ui_img = crop_image(non_ui_img, non_ui_rect)
            im11 = non_ui_img
            dump_image('im11')
            pgn = s.find_grid_nodes(non_ui_img)
            assert len(pgn) > 1
            r = Rect.from_xyxy(*pgn[0], *pgn[-1])
            grid_size = max(r.w, r.h)
            logging.info(f'grid size: {grid_size}')
        current_quad = Rect(0, 0, 0, 0)
        dir = MoveDirection.UP
        char_x = 0
        char_y = 0
        grid_coord = np.array((0,0))
        assert non_ui_rect.w < grid_size * 2 and non_ui_rect.h < grid_size * 2
        assert non_ui_rect.w > grid_size + 100 and non_ui_rect.h > grid_size + 100
        seq_press = get_ahk_sequence(dir, KeyState.PRESS)
        seq_release = get_ahk_sequence(dir, KeyState.RELEASE)
        logging.info(f'{seq_press} {seq_release}')
        roi_wh = np.array((256, 256))
        grid_wh = np.array((grid_size, grid_size))
        grid_mid = grid_wh // 2
        get_roi = lambda x: Rect(*(x - roi_wh // 2), *roi_wh)

        # full screen find
        im = s.wait_next_frame()
        non_ui_img = crop_image(im, non_ui_rect)
        pgn = s.find_grid_nodes(non_ui_img)
        grid_node = np.array(pgn[0])
        roi_rect = get_roi(grid_node)
        roi_img = crop_image(non_ui_img, roi_rect)
        dump_image('roi_img')
        assert len(pgn) != 0

        grid_rect = get_grid_rect(grid_wh, grid_node, char_offs)
        grid_img = crop_image(non_ui_img, grid_rect)
        dump_image('grid_img', f'_grid_img_{grid_coord[0]:+03d}_{grid_coord[1]:+03d}')
        logging.info(f'grid rect: {grid_rect}')

        grid_mid_point = grid_node + grid_mid
        logging.info(f'grid_mid_point: {grid_mid_point}')
        absolute_char_pos = np.array(char_offs) - grid_mid_point
        logging.info(f'absolute_char_pos: {absolute_char_pos}')
        starting_points = non_ui_img.copy()
        cv.circle(starting_points, grid_mid_point, 3, (255, 0, 0), 2)
        cv.circle(starting_points, char_offs, 3, (255, 0, 0), 2)
        cv.circle(starting_points, grid_mid_point + absolute_char_pos, 7, (0, 0, 255), 2)
        dump_image('starting_points')
        cur_grid_node = deepcopy(grid_node)
        s.ahk.send_input(seq_press)
        t1 = time.time()
        roi_mid = roi_wh // 2
        roi_grid_node = roi_mid
        i = 0
        frms = list()
        rois = list()
        while True:
            # search only near rois, which are easily calculated
            # expect only one grid node in roi
            prev_roi_img = roi_img
            roi_img = s.wait_next_frame(roi=roi_rect)
            nodes = s.find_grid_nodes(roi_img)
            pgn = s.find_grid_nodes(non_ui_img)
            roi_img_xor = cv.bitwise_xor(roi_img, prev_roi_img)
            dump_image('roi_img_xor', f'_roi_img_xor{i:03d}')
            
            # tmp = roi_img.copy()
            # if len(nodes) > 0:
            #     cv.circle(tmp, nodes[0], 3, (0, 255, 255), 2)
            # rois.append(tmp)
            # roi_loc_img = non_ui_img.copy()
            # cv.rectangle(roi_loc_img, roi_rect.xywh(), (255,0,0), 1)
            # dump_image('roi_loc_img', f'_roi_loc_img_{i}')

            if non_ui_rect.h - grid_node[1] < 10:
                non_ui_img = s.wait_next_frame(roi=non_ui_rect)
                pgn = s.find_grid_nodes(non_ui_img)
                grid_node = np.array(pgn[0])
                grid_mid_point = grid_node + grid_mid
                roi_rect = get_roi(grid_node)
                grid_rect = get_grid_rect(grid_wh, grid_node, char_offs)
                grid_img = crop_image(non_ui_img, grid_rect)
                grid_coord[1] -= 1
                nodes = [roi_mid]
                logging.info(f'new grid rect discovered: {grid_rect} {grid_coord}')
                dump_image('grid_img', f'_grid_img_{grid_coord[0]:+03d}_{grid_coord[1]:+03d}')

            # assert len(nodes) == 1
            prev_roi_grid_node = roi_grid_node
            roi_grid_node = nodes[0]
            move_diff = np.array(roi_grid_node) - roi_mid
            logging.info(f'move diff: {move_diff}')
            grid_node += move_diff
            grid_mid_point += move_diff
            absolute_char_pos -= move_diff
            #roi_rect = roi_rect.moved(*move_diff)
            roi_rect += move_diff

            correction = np.array((0, 10))

            roi_rect_img = s.wait_next_frame()

            cv.rectangle(roi_rect_img, roi_rect.moved(*correction).xywh(), (255,0,0), 1)
            cv.circle(roi_rect_img, grid_node + correction, 3, (0, 255, 255), 2)
            cv.circle(roi_rect_img, char_offs, 3, (0, 255, 0), 2)
            cv.circle(roi_rect_img, grid_mid_point + correction, 3, (255, 0, 255), 2)
            frms.append(roi_rect_img)

            if time.time() - t1 > 5:
                logging.info(f'time is up')
                break
            i += 1

        s.ahk.send_input(seq_release)
        for i, f in enumerate(frms):
            dump_image('f', f'_frm{i:03d}')


def test_explore_two_by_two_area_v2():
    from mapar.snail import WidgetType, get_grid_rect, get_grid
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    with Snail() as s:
        s.ensure_next_frame()
        r, f0, f1 = s.get_widget_brects()
        time.sleep(0.3)
        non_ui_rect = s.non_ui_rect(r)
        logging.info(f'non ui rect: {non_ui_rect}')

        i = 0
        fb = []
        t0 = time.time()
        dd = set()
        while True:
            t = time.time()
            fr = s.wait_next_frame()
            non_ui_img = crop_image(fr, non_ui_rect)
            vl, hl, mvl, mhl, ds = get_grid(non_ui_img)
            dd.update(ds)

                # non_ui_img[*n] = (255, 255, 255)
            fb.append(non_ui_img)
            while time.time() - t < 0.017:
                time.sleep(0.005)
            if time.time() - t0 > 10:
                break
        logging.info(f'{dd}')
        logging.info(f'write frames to disk')
        for i, f in enumerate(fb):
            cv.imwrite(f'tmp/non_ui_img_{i:06d}.bmp', f)
            # dump_image('f', f'_non_ui_img_{i:06d}')


def test_open_debug_options():
    from mapar.snail import WidgetType
    with overlay_client() as ovl_show_img, Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
         hotkey_handler('^1', 'open_debug_ui') as open_debug_ui, \
         hotkey_handler('^2', 'stop_anim') as stop_anim, \
         track_dist(3.0) as (tr, get_ani_map), timeout(1000) as is_not_timeout:
        im = s.wait_next_frame()
        out = im.copy()
        while is_not_timeout():
            if cmd_get() == 'exit':
                w, h = s.window_rect.wh()
                out = np.zeros((h, w, 3), dtype=np.uint8)
                putOutlinedText(out, 'exiting', (100,100))
                ovl_show_img(out)
                break
            if open_debug_ui() == 'open_debug_ui':
                s.ahk.send('{F4}')
                if s.debug_ui_rect is None:
                    time.sleep(0.02)
                    out = s.wait_next_frame()
                    br, im1, im2 = s.get_widget_brects()
                    uir = None
                    for r in br:
                        lbls = label_brect(r, s.window_rect)
                        if lbls == set((UiLocation.TOP, UiLocation.LEFT)):
                            s.debug_ui_rect = r
                            cv.rectangle(out, r.xywh(), (0, 255,0), 1)
                if s.debug_ui_rect is not None:
                    uir = s.debug_ui_rect
                    s.ahk.send('^f')
                    s.ahk.set_clipboard('show-entity-positions')
                    s.ahk.send('^v')
                    time.sleep(0.01)
                    im = s.wait_next_frame(uir)
                    dump_image('im')
                    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                    _, mask = cv.threshold(im, 128, 255, cv.THRESH_BINARY)
                    mask = dilate(mask, 2, cv.MORPH_RECT)
                    ccs = get_ccs(mask, with_br=True)
                    h, w = mask.shape
                    p = np.array((0, h))
                    srt = sorted(ccs, key=lambda x: np.linalg.norm(p - Rect(*x[1]).xy()))
                    c = np.array(Rect(*srt[0][1]).center()) + uir.xy()
                    s.ahk.click(*c)
                    s.ahk.send('{Esc}')
                ovl_show_img(out)
            time.sleep(0.01)

def mouse_drag(s: Snail, rect: Rect, mouse_spd: int = 0, sleep_time: float = 0.01):
    xy, wh = np.array(rect.xy()), np.array(rect.wh())
    s.ahk.mouse_move(*xy, speed=mouse_spd)
    s.ahk.click(button='L', direction='D')
    time.sleep(sleep_time)
    s.ahk.mouse_move(*(xy+wh), speed=mouse_spd)
    s.ahk.click(button='L', direction='U')

''' select non-ghosts for decustruction, this way we disable animation
    so on consecutive frames it can be cancelled by xoring two images
'''
def mark_area_deconstruct_non_ghosts(s: Snail, rect: Rect, mouse_spd=0, sleep_duration: float = 0.05):
    # select deconstruct non-ghosts tool
    with tool_selector(s, tool='9', sleep_duration=sleep_duration):
        mouse_drag(s, rect, mouse_spd, sleep_time=sleep_duration)

from contextlib import contextmanager

@contextmanager
def tool_selector(s: Snail, tool: str | List[str], tool_reset: str | List[str] = 'q', sleep_duration: float = 0.05):
    # TODO: add translation tool to specific key presses
    logging.info('tool select')
    if type(tool) == list:
        for t in tool:
            logging.info(f'sending keypress {t}')
            s.ahk.send(t)
            time.sleep(sleep_duration)
    elif type(tool) == str:
        logging.info(f'sending keypress {tool}')
        s.ahk.send(tool)
        time.sleep(sleep_duration)
    def reset_tool():
       pass
    yield reset_tool
    logging.info('tool reset')
    if type(tool_reset) == list:
        for t in tool_reset:
            logging.info(f'sending keypress {t}')
            s.ahk.send(t)
            time.sleep(sleep_duration)
    elif type(tool_reset) == str:
        logging.info(f'sending keypress {tool_reset}')
        s.ahk.send(tool_reset)
        time.sleep(sleep_duration)

@contextmanager
def diff_frame(s: Snail, roi: Rect = None):
    diff = [s.wait_next_frame(roi)]
    logging.info('first screen')
    def accessor():
        return diff
    yield accessor
    logging.info('second screen')
    diff.append(s.wait_next_frame(roi))

@contextmanager
def mouse_drag_release(s: Snail, rect: Rect, mouse_spd: int = 0, sleep_time: float = 0.01):
    def nop():
        pass
    xy, wh = np.array(rect.xy()), np.array(rect.wh())
    s.ahk.mouse_move(*xy, speed=mouse_spd)
    s.ahk.click(button='L', direction='D')
    time.sleep(sleep_time)
    s.ahk.mouse_move(*(xy+wh), speed=mouse_spd)
    yield nop
    s.ahk.click(button='L', direction='U')

def unmark_area_deconstruct_non_ghosts(s: Snail, rect: Rect, mouse_spd: int = 0, sleep_duration: float = 0.05):
    # TODO: add tool translation, 9 is select non-ghost buildings
    with tool_selector(s, tool=['9', '{Shift Down}'], tool_reset=['{Shift Up}', 'q'], sleep_duration=sleep_duration):
        mouse_drag(s, rect, mouse_spd=mouse_spd, sleep_time=sleep_duration)

def get_ghosts_locations_diff_image1(s: Snail, rect: Rect):
    s.ahk.send('0')
    xy = np.array(rect.xy())
    wh = np.array(rect.wh())
    s.ahk.mouse_move(*(xy + (2, 2)), speed=0)
    s.ahk.click(button='L', direction='D')
    s.ahk.mouse_move(*(xy+wh), speed=0)
    time.sleep(0.01)
    im1 = s.wait_next_frame(rect)
    s.ahk.send('{Shift Down}')
    time.sleep(0.01)
    im2 = s.wait_next_frame(rect)
    s.ahk.click(button='L', direction='U')
    s.ahk.send('{Shift Up}')
    s.ahk.send('q')
    return npext(im1), npext(im2)

def get_ghosts_locations_diff_image(s: Snail, rect: Rect, mouse_spd: int = 0, sleep_duration: float = 0.01):
    
    # TODO: add tool hotkey translation
    # select ghost deconstruct tool
    with tool_selector(s, '0', sleep_duration=sleep_duration), \
        mouse_drag_release(s, rect, mouse_spd=mouse_spd, sleep_time=sleep_duration):
        with diff_frame(s, rect) as diff:
            s.ahk.send('{Shift Down}')
            time.sleep(sleep_duration)
        im1, im2 = diff()
    s.ahk.send('{Shift Up}')
    time.sleep(sleep_duration)
    return npext(im1), npext(im2)

def get_grid_img(s: Snail, rect: Rect, sleep_duration: float = 0.05):
    # shift + space is pause toggle
    s.ahk.send('+ ')
    time.sleep(sleep_duration)
    gr = s.wait_next_frame(rect)
    s.ahk.send('+ ')
    time.sleep(sleep_duration)
    return npext(gr)


def get_plan_image(s: Snail, rect: Rect, mouse_spd: int = 0, sleep_time: float = 0.05):
    # deconstruct tool
    with tool_selector(s, tool='!d', sleep_duration=sleep_time):
        mouse_drag(s, rect, mouse_spd=mouse_spd, sleep_time=sleep_time)
    with diff_frame(s, rect) as diff:
        s.ahk.send('^z')
        time.sleep(sleep_time)
    time.sleep(sleep_time)
    bg, comp = diff()
    return bg, comp


def get_widget_brects(self, default_delay=0.2) -> Tuple[List[Rect], np.ndarray, np.ndarray]:
    it_count = 2
    sleep_time = default_delay
    def initialize():
        time.sleep(sleep_time)
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


def zoom_diff(s: Snail, r: Rect = None, sleep_time: float = 0.05, it_count = 2):
    s.ahk.mouse_move(1, 1)
    with diff_frame(s, r) as diff:
        time.sleep(sleep_time)
        s.ahk.send_input(f'{{WheelUp {it_count}}}')
        time.sleep(sleep_time)
    im1, im2 = diff()
    s.ahk.send_input(f'{{WheelDown {it_count}}}')
    return im1, im2

def test_zoom_diff():
    with Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s:
        time.sleep(3.0)
        im1, im2 = zoom_diff(s)
        imxor = cv.bitwise_xor(im1, im2)
        dump_images('im1, im2, imxor')


def test_get_tooltip1():
    clear_images()
    with overlay_client() as ovl_show_img, Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
         hotkey_handler('^1', 'mark_ghosts', ahk=s.ahk) as mark_ghosts, \
         hotkey_handler('^2', 'deploy', ahk=s.ahk) as deploy, \
         hotkey_handler('^3', 'snapshot', ahk=s.ahk) as snapshot, \
         hotkey_handler('^4', 'grid_snapshot', ahk=s.ahk) as grid_snapshot, \
         hotkey_handler('^5', 'select_bp', ahk=s.ahk) as select_bp, \
         hotkey_handler('^6', 'send_spider', ahk=s.ahk) as send_spider, \
        track_dist(3.0) as (tr, get_ani_map), timeout(1000) as is_not_timeout:
        
        # s.ahk.start_hotkeys()
        im = s.wait_next_frame()
        # out = im.copy()
        # out = np.zeros_like()
        grid_width = 32
        char_reach = 10
        debug = True
        
        if s.char_offset is not None:
            rect = Rect.from_ptdm(np.array(s.char_offset) - np.array((grid_width, grid_width)) * char_reach, \
                        np.array((grid_width, grid_width)) * char_reach * 2)
        else:
            rect = s.non_ui_rect

        # out = np.zeros(shape=(*rect.wh(), 4), dtype=np.uint8)
        # out[:,:,0] = 128
        # out[:,:,3] = 128
        logging.info(f'{rect}')

        while is_not_timeout():
            if cmd_get() == 'exit':
                logging.info('exit triggered')
                break

            if send_spider() == 'send_spider':
                time.sleep(0.5)
                im = s.wait_next_frame()
                # dump_image('im')
                blobs, stats, msk = translate_calculate_restore(im)
                # midpoint = np.array(s.window_rect.wh()) // 2
                # s.ahk.click(*midpoint, button='R')
                dump_images('im, msk')
                if len(blobs) > 0:
                    s.ahk.send('{Shift Down}')
                    for p in blobs[1]:
                        x, y = p[0]
                        s.ahk.click(x, y, button='R')
                    s.ahk.send('{Shift Up}')
                pass
                

            if grid_snapshot() == 'grid_snapshot':
                nogr = s.wait_next_frame(rect)
                with tool_selector(s, tool='+ ', tool_reset='+ ', sleep_duration=0.01), diff_frame(s, rect) as diff:
                    pass
                gr, _ = diff()
                dump_image('gr')

            if select_bp() == 'select_bp':
                mouse_speed = 0
                sleep_duration = 0.02
                time.sleep(0.5)
                gr = get_grid_img(s, rect, sleep_duration)
                im2, im1 = get_ghosts_locations_diff_image(s, rect, mouse_speed, sleep_duration)
                im2 = im2 | bgr2rgb()
                im1 = im1 | bgr2rgb()
                mrks = get_marks(im1, im2)
                vl, hl, mvl, mhl, cell_width = get_grid(gr.array)
                grd = grid(vl, hl)
                ents, ents1 = get_entity_coords_from_marks(mrks.array, vl, hl, cell_width)
                map_c2e = get_cell_to_entity_map(ents1)

                out = mrks | gray2rgb()

                for i, (loc, sz) in enumerate(map_c2e.items()):
                    p = grd[*loc]
                    putOutlinedText(out.array, f'{sz}', p + (5,16), sz=0.35)

                logging.info(map_c2e)
                dump_images('gr, im1, im2, mrks, out')

                first_machine = list(map_c2e.items())[0][0]
                fm_offset = np.array(rect.xy()) + grd[*first_machine] + [cell_width, cell_width]
                logging.info(f' {first_machine} {np.array(rect.xy())} {grd[*first_machine]} {fm_offset}')

                with diff_frame(s) as diff:
                   s.ahk.mouse_move(*fm_offset)

                ttim1, ttim2 = diff()
                dump_images('ttim1, ttim2')
                

            if snapshot() =='snapshot':
                mouse_speed = 0
                sleep_duration = 0.02
                time.sleep(0.5)
                gr = get_grid_img(s, rect, sleep_duration)
                bg, comp = get_plan_image(s, rect, mouse_speed, sleep_duration)
                # time.sleep(sleep_duration * 2)
                fg, fg1 = get_foreground(bg, comp)
                # mark_area_deconstruct_non_ghosts(s, rect, mouse_speed, sleep_duration)
                # time.sleep(sleep_duration * 2)
                im2, im1 = get_ghosts_locations_diff_image(s, rect, mouse_speed, sleep_duration)
                # time.sleep(sleep_duration * 2)
                im2 = im2 | bgr2rgb()
                im1 = im1 | bgr2rgb()
                mrks = get_marks(im1, im2)
                # unmark_area_deconstruct_non_ghosts(s, rect, mouse_speed, sleep_duration)
                # time.sleep(sleep_duration * 2)

                dump_images('gr, bg, comp, fg, fg1, im1, im2, mrks')

                vl, hl, mvl, mhl, cell_width = get_grid(gr.array)
                grd = grid(vl, hl)
                ents, ents1 = get_entity_coords_from_marks(mrks.array, vl, hl, cell_width)
                map_c2e = get_cell_to_entity_map(ents1)
                bmap = get_belt_map(vl, hl, cell_width, map_c2e, fg.array)
                gray_fg = fg | to_gray() | gray2rgb()
                dd = get_similarity_test_cache(grd, gray_fg.array, bmap, map_c2e)
                cc = get_classes(dd)
                logging.info(f'entity classes: {cc}')
                is_belt = is_belt_pred(bmap)
                is_entity = is_entity_pred(map_c2e)
                map_idx2cls = dict()
                map_cls2idx = defaultdict(list)
                for grid_cell in iter_grid_cells(grd, fg.array, [is_entity, is_partial_cell, is_nz_mask_low, is_belt]):
                    ij, p = tuple(grid_cell.idx), grid_cell.loc
                    cls = get_image_class(dd, ij, cc)
                    map_idx2cls[ij] = cls
                    map_cls2idx[cls].append(ij)
                logging.info(map_idx2cls)
                logging.info(map_cls2idx)
                for i, (loc, cls_id) in enumerate(map_idx2cls.items()):
                    p = grd[*loc]
                    # logging.info(f'loc:{loc} clsid:{cls_id} p:{p}')
                    putOutlinedText(gray_fg.array, f'{cls_id}', p + (5,16), sz=0.35)

                dump_image('gray_fg', postfix='with_classes')
                def get_entity_center(p, sz, cell_size: np.ndarray = np.array([32,32])):
                    return p + (sz * cell_size) // 2
                for i, (cls, ijlist) in enumerate(map_cls2idx.items()):
                    # if cls not in [0, 1, 8]:
                    #     continue
                    # init tool
                    ij = ijlist[0]
                    xy = np.array(rect.xy())
                    sz = map_c2e[ij]
                    p = grd[*ij]
                    s.ahk.mouse_move(*(xy + get_entity_center(p, sz)), speed=mouse_speed)
                    time.sleep(sleep_duration*3)
                    s.ahk.send('q')
                    time.sleep(sleep_duration*3)
                    for ij in ijlist:
                        xy = np.array(rect.xy())
                        wh = np.array(rect.wh())
                        sz = map_c2e[ij]
                        p = grd[*ij]
                        s.ahk.mouse_move(*(xy + get_entity_center(p, sz)), speed=mouse_speed)
                        s.ahk.click()
                        time.sleep(sleep_duration*4)
                    s.ahk.send('q')

                strmap = '\n'.join([''.join(row.tolist()) for row in bmap])
                logging.info(strmap)
                graph = build_graph_from_map(strmap)
                paths = find_all_paths(graph)
                cpaths = collapse_paths(strmap, paths)
                logging.info(cpaths)
                
                gr = grd
                # n0 = gr[*(0,0)]
                # s.ahk.mouse_move(*(np.array([n0[0], n0[1]]) + rect.xy()), speed=mouse_speed*3)
                for path in cpaths:
                    # there's no need to inverse order of x, y coordinates
                    pt = path[0]
                    pt = [pt[1], pt[0]]
                    xy = gr[*pt]
                    s.ahk.mouse_move(*(np.array(xy) + rect.xy() + [16,16]), speed=mouse_speed)
                    time.sleep(sleep_duration*2)
                    s.ahk.send('q')
                    time.sleep(sleep_duration*2)
                    s.ahk.click(button='L', direction='D')
                    time.sleep(sleep_duration)
                    if len(path) <= 1:
                        s.ahk.click(button='L', direction='U')
                        continue
                    pt = path[1]
                    pt = [pt[1], pt[0]]
                    xy = gr[*pt]
                    s.ahk.mouse_move(*(np.array(xy) + rect.xy() + [16,16]), speed=mouse_speed)
                    time.sleep(sleep_duration)
                    for pt in path[2:]:
                        pt = [pt[1], pt[0]]
                        xy = gr[*pt]
                        s.ahk.mouse_move(*(np.array(xy) + rect.xy() + [16,16]), speed=mouse_speed)
                        s.ahk.send('r')
                        time.sleep(sleep_duration)
                    s.ahk.click(button='L', direction='U')
                    s.ahk.send('q')
                
            time.sleep(0.010)
        # s.ahk.stop_hotkeys()


def test_full_screen():
    with overlay_client() as ovl_show_img, Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s, \
         exit_hotkey(ahk=s.ahk) as cmd_get, \
         timeout(1000) as is_not_timeout:
        im = s.wait_next_frame()
        out = im.copy()
        while is_not_timeout():
            im = s.wait_next_frame()
            if cmd_get() == 'exit':
                break
            time.sleep(0.010)


def test_hotkeys():
    with overlay_client() as ovl_show_img, Snail(window_mode=SnailWindowMode.FULL_SCREEN) as s, exit_hotkey(ahk=s.ahk) as cmd_get, \
         hotkey_handler('^1', 'mark_ghosts', ahk=s.ahk) as mark_ghosts, \
         hotkey_handler('^2', 'deploy', ahk=s.ahk) as deploy, \
         hotkey_handler('^3', 'snapshot', ahk=s.ahk) as snapshot, \
         hotkey_handler('^4', 'grid_snapshot', ahk=s.ahk) as grid_snapshot, \
        timeout(1000) as is_not_timeout:
        s.ahk.start_hotkeys()
        while is_not_timeout():
            if cmd_get() == 'exit':
                break
            time.sleep(0.05)
        s.ahk.stop_hotkeys()


def test_hotkeys():
    ahk = autohotkey.AHK()
    ahk.start_hotkeys()
    with overlay_client() as ovl_show_img, exit_hotkey(ahk=ahk) as cmd_get, timeout(1000000) as is_not_timeout:
        while is_not_timeout():
            if cmd_get() == 'exit':
                break
    ahk.stop_hotkeys()

def test_whisper():

    import whisper
    model = whisper.load_model(".en")
    for i in range(3):
        with timer_ms() as elapsed:
            result = model.transcribe("g:/rec.ogg", language='en', task="transcribe")
            logging.info(result["text"])
            t = elapsed()
            logging.info(f'transcribe elapsed {t} ms')
