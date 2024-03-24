import logging
from mapar import MapParser, Snail
import ahk as autohotkey
from d3dshot import D3DShot, CaptureOutputs
import time
import datetime
import numpy as np
import cv2
import cv2 as cv
import sys, inspect
from common import DataObject, get_palette
from scipy.ndimage import center_of_mass

FACTORIO_WINDOW_NAME = 'Factorio'

def millis_now():
    return int(time.time() * 1000)


def dump_image(img_var_name: str, postfix: str = None):
    assert type(img_var_name) == str
    caller_frame = inspect.currentframe().f_back
    caller_name = caller_frame.f_code.co_name
    caller_locals = caller_frame.f_locals
    assert img_var_name in caller_locals
    img = cv.cvtColor(caller_locals[img_var_name], cv.COLOR_BGR2RGB)
    if not postfix:
        postfix = f'_{img_var_name}'
    cv.imwrite(f'tmp/{caller_name}{postfix}.png', img)


def test_get_client_rect():
    ahk = autohotkey.AHK()
    r = MapParser.get_factorio_client_rect(ahk, FACTORIO_WINDOW_NAME)
    def non_null_rect(r):
        r = DataObject(r)
        return r.width > 0 and r.height > 0
    assert non_null_rect(r)


def test_mapparser_get_window_screen():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot()
    pilimg = d3d.screenshot(region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    im = np.array(pilimg)
    #im = MapParser.get_window_snapshot(window_id)
    import cv2
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite('tst.png', im)
    assert im.shape

def test_mapparser_capture_wait_next_frame():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot(capture_output=CaptureOutputs.NUMPY)
    d3d.capture(target_fps=30, region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    t0 = millis_now()
    window.activate()
    try:
        r = d3d.wait_next_frame()
        cv.imwrite('tmp/test_mapparser_capture_wait_next_frame.png', r)
    except RuntimeError as e:
        pass
    finally:
        d3d.stop()
    assert r.shape


def test_mapparser_getrate_screenshot():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot()
    t0 = millis_now()
    pilimg = d3d.screenshot(region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    r = np.array(pilimg)
    n = 50
    for i in range(n):
        pilimg = d3d.screenshot(region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    dt = millis_now() - t0
    logging.info(f'd3d.screenshot fps is: {1000 / (dt / n)}')
    assert r.shape


def test_mapparser_getrate_capture():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot(capture_output=CaptureOutputs.NUMPY)
    d3d.capture(target_fps=30, region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    t0 = millis_now()
    window.activate()
    try:
        pilimg = d3d.get_latest_frame()
        r = pilimg
        n = 80
        dt1 = float(0)
        for i in range(n):
            #window.activate()
            im = d3d.get_latest_frame()
            #time.sleep(0.01)
            t1 = millis_now()
            #r = np.bitwise_and(r, im)
            r = cv.bitwise_and(r, im)
            dt1 += millis_now() - t1
        dt = millis_now() - t0
        logging.info(f'cumulative time per frame: {dt / n}')
        logging.info(f'cumulative time per frame (-processing): {(dt - dt1) / n}')
        logging.info(f'cumulative time per frame (processing only): {dt1 / n}')
        logging.info(f'd3d.get_latest_frame fps is: {1000 / (dt / n)}')
        r = cv.cvtColor(r, cv2.COLOR_BGR2RGB)
        cv.imwrite('tmp/test_mapparser_getrate_capture.png', r)
    except RuntimeError as e:
        pass
    finally:
        d3d.stop()
    assert r.shape

def test_mapparser_getrate_capture_wait_next_frame():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot(capture_output=CaptureOutputs.NUMPY)
    d3d.capture(target_fps=30, region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    t0 = millis_now()
    window.activate()
    try:
        r = d3d.wait_next_frame()
        n = 80
        dt1 = float(0)
        for i in range(n):
            #window.activate()
            im = d3d.wait_next_frame()
            #time.sleep(0.01)
            t1 = millis_now()
            #r = np.bitwise_and(r, im)
            r = cv.bitwise_and(r, im)
            dt1 += millis_now() - t1
        dt = millis_now() - t0
        logging.info(f'cumulative time per frame: {dt / n}')
        logging.info(f'cumulative time per frame (-processing): {(dt - dt1) / n}')
        logging.info(f'cumulative time per frame (processing only): {dt1 / n}')
        logging.info(f'd3d.wait_next_frame fps is: {1000 / (dt / n)}')
        r = cv.cvtColor(r, cv2.COLOR_BGR2RGB)
        cv.imwrite('tmp/test_mapparser_getrate_capture_wait_next_frame.png', r)
    except RuntimeError as e:
        pass
    finally:
        d3d.stop()
    assert r.shape


def _opt_flow():
    window_name = FACTORIO_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = MapParser.get_factorio_client_rect(ahk, window_name)
    robj = DataObject(r)
    d3d = D3DShot()
    d3d.capture(target_fps=30, region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    t0 = millis_now()
    #pilimg = d3d.screenshot(region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
    window.activate()
    pilimg = d3d.get_latest_frame()
    r = np.array(pilimg)
    # n = 100
    # for i in range(n):
    #     #window.activate()
    #     pilimg = d3d.get_latest_frame()
    #     #time.sleep(0.01)
    #     im = np.array(pilimg)
    #     r = cv.bitwise_and(r, im)
    # dt = millis_now() - t0
    # logging.info(f'fps is: {1000 / (dt / n)}')
    # r = cv.cvtColor(r, cv2.COLOR_BGR2RGB)
    # cv.imwrite('tst.png', r)
    # d3d.stop()
    # assert r.shape

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=17, blockSize=17)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame and find corners in it
    old_frame = np.array(d3d.get_latest_frame())
    old_frame = old_frame[300:-300, 300:-300]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    for j in range(100):
        frame = np.array(d3d.get_latest_frame())
        frame = frame[300:-300, 300:-300]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
        img = cv2.add(frame, mask)

        cv.imwrite(f'tmp/frame{j}.png', img)

        # Updating Previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    d3d.stop()


def test_snail():
    with Snail() as s:
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
        time.sleep(2)
        s.ahk.mouse_move(20,20)
        t00 = time.time_ns()
        f0, t0 = s.d3d.wait_next_frame(t00)
        s.ahk.send_input('{WheelUp 3}', blocking=True)
        time.sleep(0.1)
        t11 = time.time_ns()
        f1, t1 = s.d3d.wait_next_frame(t11)
        s.ahk.send_input('{WheelDown 3}', blocking=True)
        r = np.bitwise_xor(f0, f1)
        cv.imwrite(f'tmp/wait_next_frame_r.png', r)
        cv.imwrite(f'tmp/wait_next_frame_0.png', f0)
        cv.imwrite(f'tmp/wait_next_frame_1.png', f1)
        logging.info(f'{t0}, {t1}, {t1-t0}')
        logging.info(f'{t00}, {t11}, {t11-t00}')
        assert True

def test_get_widgets():
    from mapar.snail import get_bounding_rects
    from common import Rect, label_brect
    with Snail() as s:
        s.window.activate()
        s.ahk.send_input('{F9}', blocking=True)
        time.sleep(2)
        s.ahk.mouse_move(1, 1)
        t00 = time.time_ns()
        f0, t0 = s.d3d.wait_next_frame(t00)
        # f0 = cv.imread('tmp/wait_next_frame_0.png')
        s.ahk.send_input('{WheelUp 3}', blocking=True)
        time.sleep(0.1)
        t11 = time.time_ns()
        f1, t1 = s.d3d.wait_next_frame(t11)
        # f1 = cv.imread('tmp/wait_next_frame_1.png')
        s.ahk.send_input('{WheelDown 3}', blocking=True)
        c_composite = f0.copy()
        cols = get_palette(11)
        rects = get_bounding_rects(f0, f1)
        for i, brect in enumerate(rects):
            cv.rectangle(c_composite, brect.xywh(), cols[i])
            win = Rect(0, 0, *s.window_rect.wh())
            lbls = label_brect(brect, win)
            desc = str(i) + '-' + ','.join(map(str, lbls))
            cv.putText(c_composite, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=3)
            cv.putText(c_composite, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
        c_composite = cv.cvtColor(c_composite, cv.COLOR_BGR2RGB)
        cv.imwrite('tmp/test_get_central_widget.png', c_composite)
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
            cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=3)
            cv.putText(labels, desc, np.array(brect.xy()) + np.array((10,20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
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
        s.ahk.send_input('{F9}', blocking=True)
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

def test_get_tooltip():
    WIDGET_MINIMUM_AREA = 40 * 40
    from mapar.snail import get_bounding_rects
    from common import get_palette, Rect
    with Snail() as s:
        s.ahk.send_input('{F9}', blocking=True)
        time.sleep(1)
        sleep_time = 0.1
        def initialize():
            s.ahk.mouse_move(400, 330)
            time.sleep(sleep_time)
        def action():
            s.ahk.mouse_move(450, 360)
            time.sleep(sleep_time)
        def finalize():
            s.ahk.mouse_move(400, 330)
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
    l = label_brect(r1, win)
    s = ','.join(map(str, l))
    assert s == 'hcenter,bottom'
    l = label_brect(r2, win)
    s = ','.join(map(str, l))
    assert s == 'left,bottom'
    l = label_brect(r3, win)
    s = ','.join(map(str, l))
    assert s == 'right,top'
    


def test_uilocation_conv():
    from common import UiLocation
    l = {UiLocation(3), UiLocation(5)}
    s = ','.join(map(str, l))
    assert s == 'top,hcenter'

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


def test_find_grid_nodes():
    from mapar.snail import WidgetType
    from common import crop_image, Rect
    with Snail() as s:
        s.ahk.send_input('{F9}', blocking=True)
        time.sleep(1)
        r, f0, f1 = s.get_widget_brects()
        non_ui_rect = s.non_ui_rect(r)

        non_ui_img = crop_image(f0, non_ui_rect)
        c = s.find_grid_nodes(non_ui_img)
        # for p in c:
            # cv.circle(non_ui_img, p, 7, (255, 0, 0), 3)
        quadrants_written = []
        if len(c) == 4:
            x0 = c[0][0]
            y0 = c[0][1]
            x1 = c[-1][0]
            y1 = c[-1][1]
            r = Rect.from_xyxy(x0, y0, x1, y1)
            grid_img = crop_image(non_ui_img, r)
            dump_image('grid_img')
            cv.rectangle(non_ui_img, r.xywh(), (255, 0, 0), 3)
        dump_image('non_ui_img')
        return

def test_get_char_coords():
    pass


def test_ensure_next_frame():
    for _ in range(20):
        with Snail() as s:
            s.ensure_next_frame()
            assert s.wait_next_frame() is not None

def test_explore_two_by_two_area():
    from mapar.snail import WidgetType
    from common import crop_image, Rect, MoveDirectionSimple, MoveDirectionComposite, MoveDirection, KeyState, wrap, get_ahk_sequence
    with Snail() as s:
        s.ensure_next_frame()
        load_cached_params = False
        if load_cached_params:
            non_ui_rect = Rect(x0=0, y0=0, w=1664, h=1011)
            char_offs = (960, 554)
            grid_size = 1024
        else:
            r, f0, f1 = s.get_widget_brects()
            time.sleep(0.3)
            im = s.wait_next_frame()
            char_offs = s.get_char_coords(im)
            logging.info(f'char entity offset: {char_offs}')
            non_ui_rect = s.non_ui_rect(r)
            logging.info(f'non ui rect: {non_ui_rect}')
            im = crop_image(im, non_ui_rect)
            im11 = im
            dump_image('im11')
            pgn = s.find_grid_nodes(im)
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
        im = crop_image(im, non_ui_rect)
        dump_image('im')
        # benchmark operations
        t0 = time.time()
        n = 30
        for _ in range(n):
            pgn = s.find_grid_nodes(im)
        find_grid_nodes_ms = 1000 * (time.time() - t0) / n
        logging.info(f'find_grid_nodes bench time (full image): {find_grid_nodes_ms} ms')
        wh = np.array((256, 256))
        wh_half = wh // 2

        win = lambda x: Rect(*(x - wh // 2), *wh)

        im1 = crop_image(im, win(pgn[0]))
        grid_node = pgn[0]
        grid_wh = (grid_size, grid_size)
        roi = win(grid_node)
        dump_image('im1')
        t0 = time.time()
        n = 30
        for _ in range(n):
            pgn = s.find_grid_nodes(im1)
        find_grid_nodes_small_ms = 1000 * (time.time() - t0) / n
        logging.info(f'find_grid_nodes bench time (small image): {find_grid_nodes_small_ms} ms')

        t0 = time.time()
        n = 30
        for _ in range(n):
            im = s.wait_next_frame()
            im = crop_image(im, non_ui_rect)
            roi_img = crop_image(im, roi)
        roi_crop_ms = 1000 * (time.time() - t0) / n
        logging.info(f'roi_crop_ms bench time: {roi_crop_ms} ms')


        assert len(pgn) != 0

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
        
        grid_rect = m(*grid_node, *grid_wh)
        grid_img, p, q = crop_image(im, grid_rect, debug=True)
        dump_image('grid_img')
        logging.info(f'grid rect: {grid_rect}')
        logging.info(f'actual rect: {p} {q}')
        logging.info(f'grid nodes found: {pgn}')
        return

        s.ahk.send_input(seq_press)
        t1 = time.time()
        
        while True:
            im = s.wait_next_frame()
            im = crop_image(im, non_ui_rect)

            # search only near rois, which are easily calculated
            roi_img = crop_image(im, roi)

            cgn = s.find_grid_nodes(roi_img)
            if len(pgn) != 0:
                if len(pgn) == 4 and len(cgn) == 4:
                    raise RuntimeError('oy')
                elif len(pgn) == 4 and len(cgn) == 2:
                    raise RuntimeError('oy')
                elif len(pgn) == 2 and len(cgn) == 4:
                    raise RuntimeError('oy')
                elif len(pgn) == 2 and len(cgn) == 2:
                    raise RuntimeError('oy')
                elif len(pgn) == 2 and len(cgn) == 1:
                    raise RuntimeError('oy')
                elif len(pgn) == 1 and len(cgn) == 2:
                    if dir == MoveDirection.UP:
                        assert cgn[0][1] < char_offs[1] < cgn[1][1]
                        assert cgn[0][0] == cgn[1][0]
                        if char_offs[0] <= cgn[0][0]:
                            r = Rect(cgn[0][0] - grid_size, cgn[0][1], grid_size, grid_size)
                        else:
                            r = Rect(*cgn[0], grid_size, grid_size)
                        quad = crop_image(im, r)
                        logging.info(f'1 to 2 {r} {quad.shape}')
                        # dump_image('quad')
                        break
                    elif dir == MoveDirection.DOWN:
                        raise RuntimeError('oy')
                elif len(pgn) == 1 and len(cgn) == 1:
                    raise RuntimeError('oy')
                else:
                    raise RuntimeError('unreachable')
            if time.time() - t1 > 30:
                break
        s.ahk.send_input(seq_release)
