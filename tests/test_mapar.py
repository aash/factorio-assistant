

from mapar import MapParser
import ahk as autohotkey
from d3dshot import D3DShot
import time
import datetime
import numpy as np
import cv2
import cv2 as cv

class DataObject:
    def __init__(self, data_dict):
        self.__dict__ = data_dict

def millis_now():
    return int(time.time() * 1000)

def test_get_client_rect():
    ahk = autohotkey.AHK()
    r = MapParser.get_factorio_client_rect(ahk, 'Factorio')
    assert r['x'] > 0 and r['y'] > 0 and r['width'] > 0 and r['height'] > 0

import logging

def test_example():
    logging.info("This is an informational message")
    assert True
    
def test_mapparser_attach():
    window_name = 'Factorio'
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


def test_mapparser_getrate():
    window_name = 'Factorio'
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
    n = 300
    for i in range(n):
        pilimg = d3d.screenshot(region=(robj.x, robj.y, robj.x + robj.width, robj.y + robj.height))
        #im = np.array(pilimg)
        #r = cv.bitwise_or(r, im)
    dt = millis_now() - t0
    logging.info(f'fps is: {1000 / (dt / n)}')
    r = cv.cvtColor(r, cv2.COLOR_BGR2RGB)
    cv.imwrite('tst.png', r)
    assert r.shape

def test_mapparser_getrate1():
    window_name = 'Factorio'
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
    n = 100
    for i in range(n):
        #window.activate()
        pilimg = d3d.get_latest_frame()
        #time.sleep(0.01)
        im = np.array(pilimg)
        r = cv.bitwise_and(r, im)
    dt = millis_now() - t0
    logging.info(f'fps is: {1000 / (dt / n)}')
    r = cv.cvtColor(r, cv2.COLOR_BGR2RGB)
    cv.imwrite('tmp/tst.png', r)
    d3d.stop()
    assert r.shape


def test_opt_flow():
    window_name = 'Factorio'
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