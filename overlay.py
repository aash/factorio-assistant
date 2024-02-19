feat_yolo = True

import sys, os
import zmq
import win32gui
import ctypes
import win32api
import sys
from PyQt6 import QtGui, QtCore
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QApplication, QRubberBand, QLabel, QVBoxLayout, QWidget, QLineEdit, QGridLayout
from PyQt6.QtCore import QTimer, QPoint, QRect, QSize
from PyQt6.QtGui import QImage, QPainter, QColor, QVector2D, QPixmap
import win32gui
import ctypes
import ahk
#from ahk.daemon import AHKDaemon
import traceback
from enum import Enum
from datetime import *
from time import time, sleep
import numpy as np
# from minimal import BobberDetector
from PyQt6.QtCore import pyqtSlot, QRunnable, QThreadPool, pyqtSignal, QObject, QMutexLocker, QMutex, pyqtProperty, pyqtSignal
from simple_pid import PID
from re import match
# from lib.multi_depth_model_woauxi import RelDepthModel
# from lib.net_tools import load_ckpt
from collections import OrderedDict

# if feat_yolo:
#     from ultralytics import YOLO

SPES_API_SERVICE_ADDRESS = "tcp://127.0.0.1:3001"
HOPS_API_SERVICE_ADDRESS = "tcp://127.0.0.1:3002"

win_flt_str = 'Factorio'

from re import split
#from ocr import *

import cv2 as cv
from scipy.ndimage import center_of_mass
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport
import pybase64

import codecs
import yaml

with codecs.open('recipes.yml', 'r', encoding='utf-8') as f:
    factorio_items = yaml.safe_load(f)

class AudioCommands:
    QUIT = 'выйти'
    AM_SELECT_RECIPE = 'сделать'


def millis_now():
    return int(time() * 1000)

class WindowState(Enum):
    NORMAL = 'normal'
    SELECT_ASSEMBLY_MACHINE = 'select-assembly-machine'
    SELECT_AM_RECIPE = 'select-am-recipe'
    CALIBRATE1 = 'calibrate1'
    CALIBRATE2 = 'calibrate2'


def get_bounding_rects(in1, in2):
    '''
    Get bounding rect of biggest contour by analyzing two consequtive images of UI.
    UI elements are stationary so they should be present on both images.
    '''

    in1 = cv.cvtColor(in1, cv.COLOR_BGR2RGB)
    in2 = cv.cvtColor(in2, cv.COLOR_BGR2RGB)
    hsv1 = cv.cvtColor(in1, cv.COLOR_RGB2HSV)
    hsv2 = cv.cvtColor(in2, cv.COLOR_RGB2HSV)
    out = (((hsv1 == hsv2)) * 255).astype(np.uint8)
    out = cv.inRange(out, (0,0,250), (255, 255, 255))
    ds = 5
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    out = cv.erode(out, element)
    out = cv.dilate(out, element)
    contour, _ = cv.findContours(out, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    h, w = np.shape(in1)[:2]
    r = []
    for con in contour:
        r.append(cv.boundingRect(con))
    return r

def get_bounding_rect(in1, in2):
    '''
    Get bounding rect of biggest contour by analyzing two consequtive images of UI.
    UI elements are stationary so they should be present on both images.
    '''

    in1 = cv.cvtColor(in1, cv.COLOR_BGR2RGB)
    in2 = cv.cvtColor(in2, cv.COLOR_BGR2RGB)
    hsv1 = cv.cvtColor(in1, cv.COLOR_RGB2HSV)
    hsv2 = cv.cvtColor(in2, cv.COLOR_RGB2HSV)
    out = (((hsv1 == hsv2)) * 255).astype(np.uint8)
    out = cv.inRange(out, (0,0,250), (255, 255, 255))
    ds = 5
    element = cv.getStructuringElement(cv.MORPH_RECT, (2 * ds + 1, 2 * ds + 1), (ds, ds))
    out = cv.erode(out, element)
    out = cv.dilate(out, element)
    contour, _ = cv.findContours(out, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    biggest_area = -1
    biggest = None
    c_composite = np.zeros_like(in1)
    cols = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (128,128,128), (255,255,255)]
    h, w = np.shape(in1)[:2]
    center = np.array([h, w]) // 2
    d = np.array([h, w])
    dist_to_center = np.sqrt(np.inner(d, d))
    closest_index = -1
    closest_contour = None
    for i, con in enumerate(contour):
        c = np.zeros_like(out)
        cv.drawContours(c, contour, i, (255))
        p = center_of_mass(c)
        p = tuple(map(int, p))
        d = np.sqrt(np.inner(p-center, p-center))
        if d < dist_to_center:
            dist_to_center = d
            closest_index = i
            closest_contour = c
        x, y = p
        cv.drawContours(c_composite, contour, i, cols[i])
        cv.circle(c_composite, (y,x), 3, cols[i])
        area = cv.contourArea(con)
        if biggest_area < area:
            biggest_area = area
            biggest = con
    r = cv.boundingRect(closest_contour)
    return r, closest_contour, c_composite, out


def pi_clip(angle):
    if angle > 0:
        if angle > np.pi:
            return angle - 2*np.pi
    else:
        if angle < -np.pi:
            return angle + 2*np.pi
    return angle


class MainWindowSignals(QObject):
    stateChanged = pyqtSignal(WindowState) 
    onAudioCommand = pyqtSignal(str)
    

class MainWindow(QMainWindow):
    def __init__(self, rect = (0, 0 , 220, 32), wow_hwnd = 0, ahk = ahk.AHK()):
        QMainWindow.__init__(self)
        self.setWindowFlags( #QtCore.Qt.WindowType.Hint |
            #QtCore.Qt.WindowStaysOnTopHint |
            #QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.X11BypassWindowManagerHint
        )

        self.ahk = ahk
        self.track_position = False
        self.track_direction = False
        self.track_depth_map = False
        self.track_optical_flow = False
        self.worker_busy = False
        self.track_optical_flow_mod = True
        self.setWindowOpacity(1.0)
        self.w = QWidget()
        l = QGridLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self.w.setLayout(l)
        self.setCentralWidget(self.w)
        #self.centralWidget().setStyleSheet('border: 1px solid red; padding: 0px; margin: 0px;')
        #self.setStyleSheet('border: 1px solid red; padding: 0px; margin: 0px;')

        self.hwnd = wow_hwnd
        r = rect
        self.client_rect = rect

        self.view_width = r[2] - r[0]
        self.view_height = r[3] - r[1]
        #self.setGeometry(r[0], r[1], r[2], r[3])
        #self.setStyleSheet('border: 1px solid red; padding: 0px; margin: 0px;')
        self.lbl = QLabel()
        self.lbl.setFixedSize(QSize(self.view_width, self.view_height))
        # self.lbl.setStyleSheet('border: 0px; padding: 0px; margin: 0px;')
        self.lbl.setStyleSheet('border: 1px solid red ; padding: 0px; margin: 0px;')
        self.timer = QTimer()

        self.timer.setInterval(1000//10)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.timer_timeout_cnt = 0
        self.hist = []
        self.bhist = []
        self.frame_hist = []
        self.hist_max = 10
        self.selection = QRect()
        #self.stack = QVBoxLayout()
        #self.stack.setDirection(QVBoxLayout.Direction.Down)
        # self.stack.addWidget(self.windowStatus)
        # self.stack.addWidget(self.txtStatus)
        # self.l = QVBoxLayout()
        # self.l.addChildLayout(self.stack)
        #self.setLayout(self.stack)

        self.txtStatus = QLabel()
        self.txtStatus.setFixedWidth(400)
        self.txtStatus.setFixedHeight(25)
        self.txtStatus.setStyleSheet("color: #ffffff; background-color: black;")
        #self.txtStatus.setText(self.state.name)
        #self.sp = QSpacerItem(1,1)
        self.windowStatus = QLabel()
        self.windowStatus.setFixedWidth(400)
        self.windowStatus.setFixedHeight(25)
        self.windowStatus.setStyleSheet("color: #ffffff; background-color: black;")
        self.setLayoutDirection(QtCore.Qt.LayoutDirection.LayoutDirectionAuto)
        self.fish_loop_toggle = False
        self.cvtb = {True: 'Y', False: 'N'}
        self.fish_loop_toggle_fmt = lambda: f'L:{self.cvtb[self.fish_loop_toggle]}'
        self.windowStatus.setText(self.fish_loop_toggle_fmt())


        self.depthMapLbl = QLabel()
        self.depthMapLbl.setFixedSize(QSize(self.view_width, self.view_height))
        self.centralWidget().layout().addWidget(self.lbl, 0, 0)
        #self.centralWidget().layout().addWidget(self.depthMapLbl, 0, 1)
        self.centralWidget().layout().addWidget(self.txtStatus, 1, 0)
        self.centralWidget().layout().addWidget(self.windowStatus, 2, 0)

        

        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle)
        self.centralWidget().layout().addWidget(self.rubberBand, 0, 0)
        self.origin = QPoint()
        self.target_window = self.ahk.find_window(title=win_flt_str)
        self.countdown = millis_now()
        self.coord = None
        self.pcoord = None
        self.rect = None
        self.threadpool = QThreadPool()
        self.mut = QMutex()
        self.tt = None
        self.msk = None
        # if feat_yolo:
        #     self.yolo = YOLO('best.pt')
        self.visible_assembling_machines = []


        self.signals = MainWindowSignals()
        self.signals.stateChanged.connect(self.updateStateText)
        self.state = WindowState.NORMAL
        self.last = 0
        self.dims = (0, 0)
        self.screen = QtWidgets.QApplication.primaryScreen()
        assert self.target_window != None
        self.AM_SELECT_RECIPE = "Select a recipe for assembling"
        self.AM_AM1 = "Assembling machine 1"
        self.recipe_button_coord = None
        self.am_select_recipe_brect = None
        self.am_brect = None
        self.change_recipe_button_coord = None
        self.closed = False
        self.socket = None
        def _route_audio_command():
            ctx = zmq.Context()
            rpc_client = RPCClient(
                JSONRPCProtocol(),
                ZmqClientTransport.create(ctx, SPES_API_SERVICE_ADDRESS)
            )
            spes_api_endpoint = rpc_client.get_proxy()
            # while True:
            #     if self.closed:
            #         break
            #     msgs = spes_api_endpoint.get_messages()
            #     for m in msgs:
            #         pass
            #         #self.signals.onAudioCommand.emit(m)
            #     #sleep(0.1)

        # self.command_processor = Worker(_route_audio_command)
        # self.command_processor.signals.result.connect(lambda x: None)
        # self.command_processor.signals.error.connect(lambda x: None)
        # self.command_processor.signals.finished.connect(lambda : None)
        # self.signals.onAudioCommand.connect(self.processAudioCommand)
        #self.threadpool.start(self.command_processor)


        self.cmd = None
        self.calibrated = False
        if True:
            self.calibrated = True
            self.recipe_button_coord = (32, 122)
            self.change_recipe_button_coord = (839, 252)
            self.am_brect = (507, 236, 884, 456)
            self.am_select_recipe_brect = (725, 214, 448, 500)
        self.brect = None
        if 'IAM_TOKEN' in os.environ:
            self.iam_token = os.environ['IAM_TOKEN']
        else:
            raise RuntimeError('IAM_TOKEN environment variable is not defined')
        
        ctx = zmq.Context()
        rpc_client = RPCClient(
            JSONRPCProtocol(),
            ZmqClientTransport.create(ctx, HOPS_API_SERVICE_ADDRESS)
        )
        self.hops_api_endpoint = rpc_client.get_proxy()
        self.key_map = {}
        self.state_handlers = {}
        self.trigger_state(QtCore.Qt.Key.Key_6, WindowState.CALIBRATE2)
        self.define_state_handler(WindowState.CALIBRATE2, body=self._calibrate_mouseover_tooltip,
                                  final=self.worker_complete,result=self.nop, error=self.default_error)

    def define_state_handler(self, state:WindowState, body, result, error, final):
        self.state_handlers[state] = (body, result, error, final)

    def processAudioCommand(self, msg: str):
        r = self.hops_api_endpoint.translate_string(msg)
        print(r)

        if not r:
            return
        if 'command' in r:
            verb = r['command']
        if 'num' in r:
            num = r['num']
        if 'what' in r:
            ru_key = r['what']

        match verb:
            case 'quit':
                self.socket.close()
                self.closed = True
                self.threadpool.clear()
                self.close()
                QtWidgets.QApplication.instance().exit(0)
            case 'make':
                if ru_key and num:
                    self.state = WindowState.SELECT_AM_RECIPE
                    self.cmd = (int(num), ru_key)
    
    def updateStateText(self, _new_state: WindowState):
        self.txtStatus.setText(_new_state.name)
    
    @pyqtProperty(float)
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self.signals.stateChanged.emit(val)
        self._state = val
    

    def trigger_state(self, key, state):
        self.key_map[key] = state

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        if a0.key() == QtCore.Qt.Key.Key_Escape:
            # for i, img in enumerate(self.hist):ex
            #     img.save(f'img{str(i).zfill(2)}.png')
            self.closed = True
            self.threadpool.clear()
            self.close()
            QtWidgets.QApplication.instance().exit(0)
        if a0.key() == QtCore.Qt.Key.Key_0 and self.state not in [WindowState.SELECT_ASSEMBLY_MACHINE]:
            self.state = WindowState.SELECT_ASSEMBLY_MACHINE
            pass
        if a0.key() == QtCore.Qt.Key.Key_1 and self.state not in [WindowState.CALIBRATE1]:
            self.state = WindowState.CALIBRATE1
            pass
        if a0.key() == QtCore.Qt.Key.Key_2 and self.state not in [WindowState.SELECT_AM_RECIPE]:
            self.state = WindowState.SELECT_AM_RECIPE
            pass
        for k, s in self.key_map.items():
            if a0.key() == k and self.state not in [s]:
                self.state = s
        return super().keyPressEvent(a0)

    def mousePressEvent(self, event):
        pass
    
    def mouseMoveEvent(self, event):
        pass

  
    def mouseReleaseEvent(self, event):
        pass

    def worker_complete(self):
        with QMutexLocker(self.mut):
            self.worker_busy = False
        #print('worker finished')

    def worker_handle_result(self, res):
        pass

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        pass

    def nop(self):
        self.state = WindowState.NORMAL
        pass

    def default_error(self, info):
        self.state = WindowState.NORMAL
        print(info)

    # TODO: to be deleted, found better approach
    def find_recipe_button111(self):
        screen = QtWidgets.QApplication.primaryScreen()
        step = 17
        uix0, uiy0, uiw, uih = self.brect
        cx, cy, *_ = self.client_rect
        ui_qimg = screen.grabWindow(self.hwnd).copy(uix0, uiy0, uiw // 6, uih // 2).toImage()
        imp = convertQImageToMat(ui_qimg)
        cgrid = []
        for my in range(cy + uiy0 + step, cy + uiy0 + uih // 2, step):
            for mx in range(cx + uix0 + step, cx + uix0 + uiw // 6, step):
                cgrid.append((mx, my))
        for mx, my in cgrid:
            self.ahk.mouse_move(mx, my)
            ui_qimg = screen.grabWindow(self.hwnd).copy(uix0, uiy0, uiw // 6, uih // 2).toImage()
            im = convertQImageToMat(ui_qimg)
            p1 = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
            p2 = cv.cvtColor(imp, cv.COLOR_RGB2GRAY)
            diff = (np.not_equal(p1, p2) * 255).astype(np.uint8)
            contour, _ = cv.findContours(diff, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for con in contour:
                c = np.zeros_like(diff)
                cv.drawContours(c, [con], 0, (255))
                r = cv.boundingRect(c)
                cv.circle(c, (mx - cx - uix0, my - cy - uiy0), 1, (255))
                x1, y1, x2, y2 = r[0], r[1], r[0] + r[2], r[1] + r[3]
                if x1 <= mx - cx - uix0 <= x2 and y1 <= my - cy - uiy0 <= y2:
                    return (mx, my)
            imp = im

    def _calibrate_mouseover_tooltip(self):
        
        rects = self.get_ui_brects()
        for i, r in enumerate(rects):
            mapim = self.get_screenshot(r)
            cv.imwrite(f'ui{i}.png', mapim)
        #self.visible_assembling_machines[0]
        print('calibrate 2')
        


    def update(self):
        # self.get_screenshot()
        r = self.client_rect
        screenshot = self.screen.grabWindow(0, r[0] , r[1], r[2], r[3])
        s = screenshot.copy()
        s = s.scaled(self.view_width, self.view_height)
        # self.lbl.setPixmap(s)
        #print('update', self.view_width, self.view_height)
        self.timer_timeout_cnt += 1
        self.dims = (screenshot.width(), screenshot.height())


        #qimg = screenshot.copy(QRect(400, 225, 800, 450))
        qimg = screenshot.copy()
        scale_factor = 3/8

        im = convertQImageToMat(screenshot.toImage())
        im = np.ascontiguousarray(im[...,:3], dtype=np.uint8)

        self.visible_assembling_machines = []
        t = millis_now()
        #     results = self.yolo.predict(im, stream=True, verbose=False, conf=0.5)

        #im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        resize_factor = 0.3
        im_small = cv.resize(im, (0,0), fx=resize_factor, fy=resize_factor)
        _, im_enc = cv.imencode('.png', im_small)
        im_str = pybase64.b64encode_as_string(im_enc.tobytes())
        results = self.hops_api_endpoint.parse_screenshot(im_str)
        for r in results:
            r = tuple(map(int, (np.array(r) / resize_factor).tolist()))
            # print(r)
            am_center = (r[0] + r[2]//2, r[1] + r[3]//2)
            self.visible_assembling_machines.append(am_center)
            cv.rectangle(im, tuple(r), (0,0,255), 1)
            cv.putText(im, str(tuple(np.array(am_center) - np.array(self.dims) * 0.5)), (r[0]+ 3, r[1] + 3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

        # else:
        #     results = []
        # for result in results:
        #     for b in result.boxes:
        #         rect = list(map(int, b.xywh.tolist()[0]))
        #         am_center = tuple([rect[0], rect[1]])
        #         self.visible_assembling_machines.append(am_center)
        #         rect[0] -= rect[2]//2
        #         rect[1] -= rect[3]//2
        #         #print(rect)
        #         if len(rect) == 4:
        #             cv.rectangle(im, tuple(rect), (0,0,255), 1)
        #             cv.putText(im, str(tuple(np.array(am_center) - np.array(self.dims) * 0.5)), (rect[0]+ 3, rect[1] + 3), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
        #print(millis_now() - t)

        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        if self.brect:
            cv.rectangle(im, self.brect, (255, 0, 0), 3)

        # height, width, c = np.shape(im)
        # bytesPerLine = 3 * width
        # qImg = QImage(im.data, width, height, bytesPerLine, QImage.Format.Format_RGB32)
        # self.lbl.setPixmap(QPixmap(qImg).scaled(self.view_width, self.view_height))

        if not self.worker_busy and (self.state in self.state_handlers):
            w, r, e, f = self.state_handlers[self.state]
            def _wrapper():
                with QMutexLocker(self.mut):
                    self.worker_busy = True
                    print('worker flag set')
                w()
            worker = Worker(_wrapper)
            worker.signals.result.connect(r)
            worker.signals.error.connect(e)
            worker.signals.finished.connect(f)
            self.threadpool.start(worker)

        if self.state == WindowState.SELECT_ASSEMBLY_MACHINE:
            def _click_nearest_am():
                with QMutexLocker(self.mut):
                    self.worker_busy = True
                x, y = self.client_rect[0], self.client_rect[1]
                am_vec = enumerate(map(lambda x: x - np.array(self.dims) * 0.5, np.array(self.visible_assembling_machines)))
                #print(list(am_vec))
                closest_am_index = min([(i, np.inner(v,v)) for i, v in am_vec], key=lambda x: x[1])[0]
                offs_x, offs_y = self.visible_assembling_machines[closest_am_index]
                print(x, y, offs_x, offs_y)
                if len(self.visible_assembling_machines) > (self.last + 1):
                    self.last += 1
                else:
                    self.last = 0
                self.ahk.click(x=-self.pos().x() + x + offs_x, y=-self.pos().y() + y + offs_y)
                self.ahk.send("^fcir")
                #ahk.mouse_move(x, y)
                #ahk.click()
                print('done')
            if not self.worker_busy:
                worker = Worker(_click_nearest_am)
                worker.signals.result.connect(self.nop)
                worker.signals.finished.connect(self.worker_complete)
                #print('worker started')
                self.threadpool.start(worker)
            pass
        
        if self.state == WindowState.CALIBRATE1:
            def _find_recipe_button():
                st = millis_now()
                with QMutexLocker(self.mut):
                    self.worker_busy = True
                cx, cy, *_ = self.client_rect
                r = self.get_main_ui_brect()
                ui_dialog_img = self.get_screenshot(r)
                s, _ = ocr(self.iam_token, ui_dialog_img)
                dialog_caption = s.split('\n')[0]
                if dialog_caption == self.AM_SELECT_RECIPE:
                    self.am_select_recipe_brect = r
                    uix0, uiy0, *_ = r
                    self.ahk.mouse_move(cx + uix0, cy + uiy0)
                    self.ahk.send('^fwood')
                    ui_dialog_img = self.get_screenshot(r)
                    wc_img = cv.imread('wooden_chest.png')
                    wc_img = cv.cvtColor(wc_img, cv.COLOR_BGR2RGB)
                    p = template_match(ui_dialog_img, wc_img)
                    if p:
                        self.recipe_button_coord = p
                        self.ahk.click(cx + uix0 + p[0], cy + uiy0 + p[1])
                        r = self.get_main_ui_brect()
                        ui_dialog_img = self.get_screenshot(r)
                        s, _ = ocr(self.iam_token, ui_dialog_img)
                        dialog_caption = s.split('\n')[0]
                        assert dialog_caption == self.AM_AM1
                        self.am_brect = r
                        uix0, uiy0, *_ = r
                        cog_img = cv.imread('cog.png')
                        cog_img = cv.cvtColor(cog_img, cv.COLOR_BGR2RGB)
                        p = template_match(ui_dialog_img, cog_img)
                        if p:
                            self.change_recipe_button_coord = p
                            self.ahk.click(cx + uix0 + p[0], cy + uiy0 + p[1])
                        else:
                            print('could not find change recipe button')
                    else:
                        print('could not find select recipe button')
                elif dialog_caption == self.AM_AM1:
                    pass
                print(f'calibration done {millis_now() - st}')
                print(self.recipe_button_coord)
                print(self.change_recipe_button_coord)
                print(self.am_brect)
                print(self.am_select_recipe_brect)
                self.calibrated = True
            if not self.worker_busy:
                worker = Worker(_find_recipe_button)
                worker.signals.result.connect(self.nop)
                worker.signals.error.connect(self.default_error)
                worker.signals.finished.connect(self.worker_complete)
                self.threadpool.start(worker)
            pass


        if self.state == WindowState.SELECT_AM_RECIPE:
            def _find_recipe_button():
                if not self.calibrated:
                    print('warning: not calibrated')
                    return
                with QMutexLocker(self.mut):
                    self.worker_busy = True
                st = millis_now()

                cx, cy, *_ = self.client_rect
                am_vec = enumerate(map(lambda x: x - np.array(self.dims) * 0.5, np.array(self.visible_assembling_machines)))
                #print(list(am_vec))
                closest_am_index = min([(i, np.inner(v,v)) for i, v in am_vec], key=lambda x: x[1])[0]
                offs_x, offs_y = self.visible_assembling_machines[closest_am_index]
                # offs_x, offs_y = 961, 407
                # print(cx, cy, offs_x, offs_y)
                # if len(self.visible_assembling_machines) > (self.last + 1):
                #     self.last += 1
                # else:
                #     self.last = 0
                self.target_window.activate()
                self.ahk.mouse_move(cx + offs_x, cy + offs_y, speed=1)
                self.ahk.click()

                num, ru_key = self.cmd
                recipe = next(filter(lambda x: x[1]['ru-key'] == ru_key, factorio_items.items()), None)
                print(num, recipe)
                selected_item, item = recipe
                cx, cy, *_ = self.client_rect
                r = self.get_main_ui_brect()
                self.brect = r

                uix0, uiy0, *_ = r
                ui_dialog_img = self.get_screenshot(r)
                print(r)
                dtocr = 0
                # tocr = millis_now()
                # s, _ = ocr(self.iam_token, ui_dialog_img)
                # dtocr += millis_now() - tocr
                # dialog_caption = s.split('\n')[0]
                # print(dialog_caption)
                #if dialog_caption == self.AM_AM1:
                if r[2] > 700:
                    p = self.change_recipe_button_coord
                    self.ahk.click(cx + uix0 + p[0], cy + uiy0 + p[1])
                    r = self.get_main_ui_brect()
                    uix0, uiy0, *_ = r
                    assert r[2] < 700
                    # ui_dialog_img = self.get_screenshot(r)
                    # tocr = millis_now()
                    # s, _ = ocr(self.iam_token, ui_dialog_img)
                    # dtocr += millis_now() - tocr
                    # dialog_caption = s.split('\n')[0]
                    # assert dialog_caption == self.AM_SELECT_RECIPE


                self.ahk.send('^f' + item['query'])
                p = self.recipe_button_coord
                # self.ahk.mouse_move(cx + uix0 + p[0], cy + uiy0 + p[1])
                self.ahk.click(cx + uix0 + p[0], cy + uiy0 + p[1])
                self.ahk.send('e')
                    
                # def init():
                #     self.ahk.mouse_move(cx + uix0, cy + uiy0, speed=1)
                # def act():
                #     self.ahk.mouse_move(cx + uix0 + p[0], cy + uiy0 + p[1], speed=1)

                # im1, im2 = self.get_diff_image(init, act, r)
                # tool_tip = parse_tooltip(im1, im2)
                # tocr = millis_now()
                # s, _ = ocr(self.iam_token, tool_tip)
                # dtocr += millis_now() - tocr
                # tooltip_caption = s.split('\n')[0]
                
                # if tooltip_caption == item['tooltip-caption'] + ' (Recipe)':
                #     print(f'seleted tooltip: "{tooltip_caption}"')
                #     self.ahk.click()
                #     self.ahk.send('e')
                # else:
                #     print('tooltip validation failed')

                self.ahk.mouse_move(cx + offs_x, cy + offs_y)
                for c, i in item['ingredients']:
                    if factorio_items[i]['key-map']:
                        toolbar, key = factorio_items[i]['key-map']
                        print(i, factorio_items[i]['key-map'])
                        self.ahk.send('+{' + str(toolbar) + '}', key_delay=50)
                        self.ahk.send(key, key_delay=50)
                        self.ahk.send('{z down}{z up}' * (num * c), key_delay=30, key_press_duration=20)
                self.ahk.send('q', key_delay=50)

                print(f'select recipe done {millis_now() - st}, (ocr took: {dtocr})')
                self.brect = None

                
            if not self.worker_busy:
                worker = Worker(_find_recipe_button)
                worker.signals.result.connect(self.nop)
                worker.signals.error.connect(self.default_error)
                worker.signals.finished.connect(self.worker_complete)
                self.threadpool.start(worker)
            pass
        
    def get_screenshot(self, roi=None):
        img = self.screen.grabWindow(self.hwnd)
        if roi:
            img = img.copy(*roi)
        return convertQImageToMat(img.toImage())

    def get_diff_image(self, init, act, roi=None):
        init()
        im1 = self.get_screenshot(roi)
        act()
        im2 = self.get_screenshot(roi)
        return im1, im2
    
    def get_main_ui_brect(self):
        screen = QtWidgets.QApplication.primaryScreen()
        it_count = 4
        sleep_time = 0.05
        cx, cy = self.client_rect[0], self.client_rect[1]
        def init():
            self.target_window.activate()
            self.ahk.mouse_move(cx, cy)
        def action():
            # for i in range(it_count):
            self.ahk.send(f'{{WheelUp {it_count}}}')
            sleep(sleep_time)
        def action_cleanup():
            # for i in range(it_count):
            self.ahk.send(f'{{WheelDown {it_count}}}')
        im, im1 = self.get_diff_image(init, action)
        action_cleanup()
        r, *_ = get_bounding_rect(im, im1)
        return r

    def get_ui_brects(self):
        it_count = 4
        sleep_time = 0.05
        cx, cy = self.client_rect[0], self.client_rect[1]
        def init():
            self.target_window.activate()
            self.ahk.mouse_move(cx, cy)
        def action():
            self.ahk.send(f'{{WheelUp {it_count}}}')
            sleep(sleep_time)
        def action_cleanup():
            self.ahk.send(f'{{WheelDown {it_count}}}')
        im, im1 = self.get_diff_image(init, action)
        action_cleanup()
        cv.imwrite('ui1.png', im)
        cv.imwrite('ui2.png', im1)
        r, *_ = get_bounding_rects(im, im1)
        return r
    
    # def state_handler(self):
    #     def wrapper(f):
    #         pass
    #     return wrapper

    # @state_handler(WindowState.CALIBRATE1)
    # def some_handler(self):
    #     pass


def template_match(img, tpl):
    img_rgb = img
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    cog = cv.cvtColor(tpl, cv.COLOR_RGB2GRAY)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray,None)
    kp2, des2 = sift.detectAndCompute(cog,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # TODO: rework this madness using filtering by matches
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            matchesMask[i]=[1,0]
    l = list(map(lambda x: x[0], filter(lambda x: x[1] == [1, 0], enumerate(matchesMask))))
    pts = [kp1[i].pt for i in l]
    if len(l) > 0:
        pt = tuple(map(int, np.mean(pts, axis=0)))
        return pt
    else:
        return None

def parse_tooltip(im1, im2):
    rgb1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    rgb2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    diff = ((rgb1 != rgb2) * 255).astype(np.uint8)
    contour, _ = cv.findContours(diff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    biggest_area = -1
    biggest = None
    for con in contour:
        area = cv.contourArea(con)
        if biggest_area < area:
            biggest_area = area
            biggest = con
    if biggest_area > 0:
        c = np.zeros_like(diff)
        cv.drawContours(c, [biggest], 0, (255))
        r = cv.boundingRect(c)

    x1, y1, x2, y2 = r[0]+4, r[1]+4, r[0] + r[2]-8, r[1] + r[3]

    tool_tip = rgb2[y1:y2, x1:x2]
    return tool_tip

def ocr(iam_token, img):
    import requests
    import base64
    import json
    url = 'https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText'
    if np.shape(img[0][0]) != ():
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret, img_str = cv.imencode('.png', img)
    img_str = base64.b64encode(img_str.tobytes()).decode('ascii')
    curData = {
        "mimeType": "image/png",
        "languageCodes": ["en"],
        "model": "page",
        "content": img_str
    }
    hdr = {
        "authorization": f"Bearer {iam_token}",
        'x-folder-id': f'b1gplk6gop2viqk2q4b8'
    }
    requests.packages.urllib3.disable_warnings()
    ocr_result = requests.post(url, data=json.dumps(curData), headers=hdr, verify=False)
    if ocr_result.status_code == 200:
        r = json.loads(ocr_result.text)
        txt = r['result']['textAnnotation']['fullText']
        return txt, r
    else:
        raise RuntimeError(ocr_result)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(
                *self.args, **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()




# def get_hwnd(flt):

#     def enum_cb(hwnd, results):
#         winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

#     toplist, winlist = [], []
#     win32gui.EnumWindows(enum_cb, toplist)

#     for hwnd, title in winlist:
#         if flt.lower() in title.lower():
#             return hwnd


# def get_dpi():
#     PROCESS_PER_MONITOR_DPI_AWARE = 2
#     MDT_EFFECTIVE_DPI = 0
#     shcore = ctypes.windll.shcore
#     monitors = win32api.EnumDisplayMonitors()
#     hresult = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
#     assert hresult == 0
#     dpiX = ctypes.c_uint()
#     dpiY = ctypes.c_uint()
#     dpi = {}
#     for i, monitor in enumerate(monitors):
#         shcore.GetDpiForMonitor(
#             monitor[0].handle,
#             MDT_EFFECTIVE_DPI,
#             ctypes.byref(dpiX),
#             ctypes.byref(dpiY)
#         )
#         dpi[monitor[0].handle] = (dpiX.value, dpiY.value)
#     return dpi    


# def dpi_to_scale_ratio(dpi):
#     STANDARD_DPI = 96
#     if len(dpi) != 2 or dpi[0] != dpi[1]:
#         raise RuntimeError(f'non conformant DPI:{dpi[0]}x{dpi[1]}')
#     return dpi[0] / STANDARD_DPI


if __name__ == '__main__':
    ahk = ahk.AHK()
    w = ahk.find_window(title='Factorio')
    hwnd = int(w.id, 16)
    w.move(x=0, y=0)
    c = win32gui.ClientToScreen(hwnd, (0,0))
    res1 = {'width': 1920, 'height': 1080}
    res2 = {'width': 1440, 'height': 810}
    w.move(x=0, y=0, **res2)
    w.activate()

    # PROCESS_PER_MONITOR_DPI_AWARE = 2
    # MDT_EFFECTIVE_DPI = 0
    # shcore = ctypes.windll.shcore
    # hresult = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
    # r = win32gui.GetWindowRect(hwnd)
    rc = win32gui.GetClientRect(hwnd)
    print(rc, c)
    r = (c[0], c[1], rc[2], rc[3])
    app = QApplication(sys.argv)

    # r = (c[0], c[1], rc[2], rc[3])
    # print(r)
    # w = ahk.find_window()
    mywindow = MainWindow(r, hwnd)
    # print(mywindow.screen.devicePixelRatio())

    #p = w.get_position()
    #print(p)
    mywindow.move(r[0] + r[2], r[0])
    #mywindow.setGeometry()
    mywindow.show()
    app.exec()

