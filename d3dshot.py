import time
import subprocess
import struct
import numpy as np
import zmq
from common import Rect
import threading
import queue
import logging
from os import linesep
from psutil import pid_exists
import cv2
import cv2 as cv
import os, sys

__version__ = '1.0.0'

class CaptureOutputs:
    NUMPY = 1

class D3DShot:
    EXECUTABLE_PATH = "G:/projects/nvidia_dxgi_test/proj/build/Release/dxgi_test_app.exe"
    FULL_SCREEN = (0, 0, 1980*2, 1080*2)
    FRAME_HEADER_MESSAGE_FORMAT = '<cccciiHQI'
    #SCREEN_MONITOR = 'tcp:///screen_monitor'
    SCREEN_MONITOR = 'tcp://127.0.0.1:5555'

    # capture_output is here for compatibility with original D3DShot implementation
    def __init__(self, capture_output = CaptureOutputs.NUMPY, executable = EXECUTABLE_PATH, fps = 30, roi: Rect = Rect(*FULL_SCREEN)):
        self.executable = executable
        self.monitor_proc = None
        self.fps = fps
        self.roi = roi
        self._fh_len = struct.calcsize(self.FRAME_HEADER_MESSAGE_FORMAT)
        self.exit_command = 'exit'
        self.echo_command = 'echo'
        self.exit_timeout = 10.0
        self.first_frame_timeout = 2.0
        self.retries = 0



    def __enter__(self):
        self.capture(target_fps=self.fps, region=self.roi.xyxy())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
    
    def wait_next_frame(self, t = 0) -> np.ndarray:
        for i in range(10):
            if i > 0:
                self.retries += 1
            try:
                message = self.socket.recv(zmq.DONTWAIT)
                c0, c1, c2, c3, w, h, elsz, t, n = struct.unpack(self.FRAME_HEADER_MESSAGE_FORMAT, message[:self._fh_len])
                s = c0 + c1 + c2 + c3
                if s != b'frm9' or len(message) - self._fh_len != w * h * elsz:
                    raise RuntimeError('message format error')
                image = np.frombuffer(message[self._fh_len:], dtype=np.uint8).reshape((h, w, elsz))
                del message
                return image, t
            except zmq.error.Again as e:
                time.sleep(0.010)
        raise RuntimeError('timeout on recv')

    def capture(self, target_fps: int = None, region: tuple = None):
        if region is not None:
            self.roi = Rect.from_xyxy(*region)
        if target_fps is not None:
            self.fps = target_fps
        rect_str = ','.join(map(str, self.roi.xywh()))
        logging.info(f'start dxgi screen monitor:')
        logging.info(f'{self.executable} {self.fps} {rect_str}')
        self.monitor_proc = subprocess.Popen([self.executable, str(self.fps), rect_str], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.monitor_thread = threading.Thread(target=self._screen_monitor_loop)
        self.frame_lock = threading.Lock()
        self.frame = None, 0
        self.evt = threading.Event()
        self.last_frame_no = 0
        self.monitor_thread.start()

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5556")
        socket.send_string('echo')
        reply = socket.recv_string()
        assert reply == 'echo'
        socket.close()
        context.term()


    def _screen_monitor_loop(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVTIMEO, 200)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.SCREEN_MONITOR)
        while not self.evt.is_set():
            try:
                f, t = self.wait_next_frame()
                #cv.imwrite(f'frame{t:06d}.bmp', f)
                logging.info(f'acquired frame, {t}')
                with self.frame_lock:
                    self.frame = f, t
            except zmq.error.ZMQError as e:
                logging.info(f'zmq error: {e}')
            except RuntimeError as e:
                logging.info(f'runtime error: {e}')
        logging.info('screen monitor loop end')
        self.socket.close()
        self.context.destroy()


    def get_frame(self):
        with self.frame_lock:
            f, t = self.frame
        return f, t

    def wait_first_frame(self):
        t0 = time.time()
        while True:
            if time.time() - t0 > self.first_frame_timeout:
                raise RuntimeError('first frame wait timeout')
            f, t = self.get_frame()
            if f is not None:
                break
            time.sleep(0.05)

    def get_next_frame(self):
        img, t = self.get_frame()
        t0 = time.time()
        while self.last_frame_no >= t:
            time.sleep(0.002)
            img, t = self.get_frame()
            logging.info(f'wait next frame {t}')
            if time.time() - t0 > 1.5:
                raise TimeoutError('wait frame timeout')
        self.last_frame_no = t
        return img, t


    def stop(self):
        self.evt.set()
        exs = None
        try:
            self.monitor_thread.join()
        except Exception as e:
            exs = e

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5556")
        socket.send_string(self.exit_command)
        reply = socket.recv_string()
        assert reply == self.exit_command
        socket.close()
        context.term()
        timeout = 5.0
        #self.monitor_proc.wait(timeout)
        s, e = self.monitor_proc.communicate()
        logging.info(f'exitcode: {self.monitor_proc.returncode}')
        assert self.monitor_proc.returncode == 0
        assert not pid_exists(self.monitor_proc.pid)
        logging.info('monitor process outout:')
        logging.info(s)
        
        if exs:
            raise exs

