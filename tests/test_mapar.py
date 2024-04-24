from packaging import version
import logging
from mapar import MapParser, Snail
import ahk as autohotkey
import d3dshot_stub as d3dshot
import time
import datetime
import numpy as np
import cv2
import cv2 as cv
import sys, inspect
from common import DataObject, get_palette
from copy import deepcopy
import pytest
from pytest import fail
#from scipy.ndimage import center_of_mass
from common import *
import zmq

FACTORIO_WINDOW_NAME = 'Factorio'
AHK_BINARY_PATH = 'D:/portable/ahk/AutoHotkeyU64.exe'
D3DSHOT_1_0_0 = version.parse('1.0.0')

def millis_now():
    return int(time.time() * 1000)


def dump_image(img_var_name: str, postfix: str = None):
    assert type(img_var_name) == str
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    caller_locals = caller_frame.f_locals
    assert img_var_name in caller_locals
    img = caller_locals[img_var_name]

    # if swap_br is None:
    #     if img.shape[2] == 4:
    #         swap_br = False
    #     if img.shape[2] == 3:
    #         swap_br = True
    # if type(swap_br) is bool and swap_br:
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if not postfix:
        postfix = f'_{img_var_name}'
    cv.imwrite(f'tmp/{caller_name}{postfix}.bmp', img)

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
    from mapar.snail import WidgetType, get_grid_rect, get_grid, chk
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    with Snail() as s:
        s.ensure_next_frame()
        r, f0, f1 = s.get_widget_brects()
        time.sleep(0.3)
        non_ui_rect = s.non_ui_rect(r)
        logging.info(f'non ui rect: {non_ui_rect}')
        non_ui_img = crop_image(f0, non_ui_rect)
        nodes = get_grid(non_ui_img)
        vl, hl, mvl, mhl, ds = get_grid(non_ui_img)
        w, h = non_ui_rect.wh()
        assert w / max(ds) <= len(vl) + 1
        assert w / max(ds) >= len(vl) - 1
        assert h / max(ds) <= len(hl) + 1
        assert h / max(ds) >= len(hl) - 1
        assert len(ds) < 3
        logging.info(f'{nodes}')
        logging.info(f'{ds}')
        dump_image('non_ui_img')


from common import grid
from mapar.snail import get_grid
import random

def match_grid_nodes(non_ui_img, non_ui_img1, base_node=None):
    vl0, hl0, mvl0, mhl0, ds0 = get_grid(non_ui_img)
    vl1, hl1, mvl1, mhl1, ds1 = get_grid(non_ui_img1)
    m0 = grid(vl0, hl0) 
    m1 = grid(vl1, hl1)
    # selection of base node
    '''
    10
    0..9
    10 - 4 6
    7 8 9
    '''
    if base_node is None or (base_node[0] < 3 or base_node[1] < 3 or base_node[0] > len(vl0) - 4 or base_node[1] > len(hl0) - 4):
        ij = 3 * np.array((len(vl0), len(hl0))) // 4
    else:
        ij = base_node
    st = ij
    d = np.array([[0, 0], [-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
    r = []
    cont = True

    iw = len(vl0)
    ih = len(hl0)
    a = range(iw // 4, 3 * iw // 4)
    b = range(ih // 4, 3 * ih // 4)

    ind = list(map(np.array, itertools.product(a, b)))
    random.shuffle(ind)
    candidates = ind[:5] + [ij]
    # logging.info(f'candidates: {candidates}')
    while len(candidates):
        ij = candidates.pop()
        p00, p01 = m0[*ij], m0[*(ij + (1, 1))]
        g1 = crop_image(non_ui_img, Rect.from_xyxy(*p00, *p01))
        for _d in d:
            ij1 = np.array(ij) + _d
            p10 = m1[*ij1]
            p11 = m1[*(ij1 + (1,1))]
            g2 = crop_image(non_ui_img1, Rect.from_xyxy(*p10, *p11))
            if g1.shape != g2.shape:
                h1, w1, *_ = g1.shape
                h2, w2, *_ = g2.shape
                g1_ = g1[0:min(h1, h2), 0:min(w1, w2)]
                g2_ = g2[0:min(h1, h2), 0:min(w1, w2)]
            else:
                g1_ = g1
                g2_ = g2
            diff = cv.bitwise_xor(g1_, g2_)
            mask = (diff != (0,0,0)).astype(np.uint8) * 255
            # res = np.concatenate((g1_, g2_, diff, mask), axis=1)
            # dump_image('res', f'_res_{_d[0]:+03}_{_d[1]:+03}')
            h, w, *_ = mask.shape
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            c = cv2.countNonZero(mask)
            # logging.info(f'non zero: {c} {c / (w*h)} {_d}')
            if c / (w * h) <= 0.7:
                return ij, ij1, m0, m1, (vl0, hl0, mvl0, mhl0), (vl1, hl1, mvl1, mhl1)
    logging.info(f'started with {st} and ran out of cadidates')
    return None, None, None, None, None, None


def test_extract_grid_and_match():
    from mapar.snail import WidgetType, get_grid_rect, get_grid, chk
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    with Snail() as s:
        non_ui_img = cv.imread('tmp/rec6/non_ui_img_000080.bmp')
        non_ui_img1 = cv.imread('tmp/rec6/non_ui_img_000081.bmp')
        with timer_ms() as elapsed:
            ij0, ij1, m0, m1, g1, g2 = match_grid_nodes(non_ui_img, non_ui_img1, np.array((11, 5)))
            if ij0 is None and ij1 is None:
                fail('no match')
            p00, p01 = m0[*ij0], m0[*(ij0 + (1,1))]
            p10, p11 = m1[*ij1], m1[*(ij1 + (1,1))]
            logging.info(f'{p00},{p01}')
            cv.rectangle(non_ui_img, p00, p01, (0,0,255), 1)
            cv.rectangle(non_ui_img1, p10, p11, (0,0,255), 1)
        dump_image('non_ui_img')
        dump_image('non_ui_img1')

def test_extract_grid_and_match_continuous():
    from mapar.snail import WidgetType, get_grid_rect, get_grid
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence, cart_prod
    with Snail() as s:
        s.ensure_next_frame()
        r, f0, f1 = s.get_widget_brects()
        time.sleep(0.3)
        non_ui_rect = s.non_ui_rect(r)
        logging.info(f'non ui rect: {non_ui_rect}')
        fb = []
        t0 = time.time()
        dd = set()
        fr = s.wait_next_frame()
        non_ui_img_prev = crop_image(fr, non_ui_rect)
        pij = None
        char_coords = np.array((0, 0))
        fr = s.wait_next_frame()
        non_ui_img = crop_image(fr, non_ui_rect)
        char_offs = np.array(s.get_char_coords(non_ui_img))
        i = 0
        fb1 = []
        gnindex = np.array((0, 0))
        cfidx = [0, 0, 0, 0]
        while True:
            t = time.time()
            fr = s.wait_next_frame()

            non_ui_img = crop_image(fr, non_ui_rect)
            out_copy = non_ui_img.copy()
            with timer_ms() as elapsed:
                ij0, ij1, m0, m1, grid0, grid1 = match_grid_nodes(non_ui_img_prev, non_ui_img, pij)
                # logging.info(f'elapsed: {elapsed()}')
            pij = ij1
            if ij1 is None or ij0 is None:
                logging.info(f'no match {i} {i-1}')
                pass
            else:
                p00 = m0[*ij0]
                p10, p11 = m1[*ij1], m1[*(ij1 + (1,1))]
                cv.rectangle(out_copy, p10, p11, (0,0,255), 1)
                
                vl0, hl0, mvl0, mhl0 = grid0
                vl1, hl1, mvl1, mhl1 = grid1

                char_coords_prev = char_coords



                cmv1 = next(filter(lambda x: abs(x[1] - char_offs[0]) < 15, enumerate(mvl1)), None)
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

                char_coords += (p10 - p00)

                cv.circle(out_copy, char_offs + char_coords, 3, (0,0,255), 1)
                # logging.info(f'{char_coords}')
            fb.append(out_copy)
            fb1.append(non_ui_img)
            while time.time() - t < 0.05:
                time.sleep(0.005)
            if time.time() - t0 > 10:
                break
            non_ui_img_prev = non_ui_img
            i += 1
        logging.info(f'gnindex: {gnindex}')
        logging.info(f'write frames to disk')
        for i, f in enumerate(fb):
            cv.imwrite(f'tmp/rec6/out_copy_{i:06d}.bmp', f)
        for i, f in enumerate(fb1):
            cv.imwrite(f'tmp/rec6/non_ui_img_{i:06d}.bmp', f)
            # dump_image('f', f'_non_ui_img_{i:06d}')



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

def test_overlay_add_markers():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5124")
    logging.info('connected')

    import random, json
    def generate_random_json(i):
        marker_type = "rectangle"
        geometry = [random.randint(0, 1000) for _ in range(4)]
        color = [*[random.randint(0, 255) for _ in range(3)], 255]
        data = {"name": f"rect{i}", "action": "add"}
        json_data = {
            "marker_type": marker_type,
            "geometry": geometry,
            "color": color,
            "data": data
        }
        return json.dumps(json_data)
    
    for j in range(1):
        for i in range(100):
            socket.send_json(generate_random_json(i))
            r = socket.recv_string()
            assert r == 'Received'
    socket.close()
    context.term()

def test_overlay_remove_markers():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5124")

    import random, json
    def generate_random_json(i):
        marker_type = "rectangle"
        geometry = [random.randint(0, 1000) for _ in range(4)]
        color = [*[random.randint(0, 255) for _ in range(3)], 255]
        data = {"name": f"rect{i}", "action": "remove"}
        json_data = {
            "marker_type": marker_type,
            "geometry": geometry,
            "color": color,
            "data": data
        }
        return json.dumps(json_data)
    
    for i in range(100):
        socket.send_json(generate_random_json(i))
        r = socket.recv_string()
        assert r == 'Received'
    socket.close()
    context.term()

def test_overlay_video_feed():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5124")

    import random, json
    def generate_random_json(r):
        marker_type = "image"
        geometry = list(r.xywh())
        color = [0, 0, 0, 255]
        data = {"name": "img", "action": "add_image"}
        json_data = {
            "marker_type": marker_type,
            "geometry": geometry,
            "color": color,
            "data": data
        }
        return json.dumps(json_data)
    
    from mapar.snail import WidgetType, get_grid_rect, get_grid
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence, cart_prod
    with Snail() as s:
        s.ensure_next_frame()
        r, f0, f1 = s.get_widget_brects()
        time.sleep(0.3)
        non_ui_rect = s.non_ui_rect(r)
        logging.info(f'non ui rect: {non_ui_rect}')

        t0 = time.time()
        while True:
            t = time.time()
            fr = s.wait_next_frame()
            non_ui_img = crop_image(fr, non_ui_rect)
            h, w, *_ = non_ui_img.shape
            socket.send_json(generate_random_json(Rect(0, 0, w, h)))
            r = socket.recv_string()
            assert r == 'Received'
            tmp = cv.cvtColor(non_ui_img, cv.COLOR_BGR2RGB)
            socket.send(tmp.data)
            r = socket.recv_string()
            assert r == 'Received'
            if time.time() - t0 > 35:
                break
    socket.close()
    context.term()

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
