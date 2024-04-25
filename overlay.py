import sys
import time
import threading
from dataclasses import dataclass
import ahk

from PyQt5.QtGui import QCloseEvent, QPainter, QColor, QPen, QBrush, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QGraphicsLayout, QBoxLayout, QSizePolicy, QLabel
from PyQt5.QtCore import QRect, Qt, pyqtSignal, QThread, QEvent
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QBrush, QPen, QFontMetrics, QPainterPath, QPainter

import numpy as np
from collections import defaultdict
import logging
import math, time

def millis_now():
    return int(time.time()*1000)

@dataclass
class Marker:
    marker_type: str
    geometry: tuple
    color: QColor
    data: dict
import json
import zmq

close_event = threading.Event()

def json_to_marker(json_string):
    data = defaultdict(lambda: None)
    data.update(json.loads(json_string).items())
    # print(data)
    if data['action'] == 'test':
        return Marker(data={"action": "test"}, marker_type='', geometry=(), color=QColor(0, 0, 0, 0))
    return Marker(
        marker_type=data['marker_type'],
        geometry=tuple(data['geometry']),
        color=QColor(*data['color']),
        data=data['data']
    )


class TransparentWindow(QMainWindow):
    # new_marker_signal = pyqtSignal(Marker)

    def __init__(self):
        super().__init__()

        # self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
        #                     Qt.WindowType.WindowTransparentForInput |
        #                     Qt.WindowType.WindowStaysOnTopHint
        #                     #| Qt.WindowType.Tool
        #                     )
        # self.setAttribute(Qt.WA_TranslucentBackground)


        self.w = QWidget(self)
        #self.w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.w.setFixedSize(1920, 1200)
        l = QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.w.setLayout(l)
        self.setCentralWidget(self.w)

        self.w.setStyleSheet("border: 2px dashed green")

        self.setWindowOpacity(0.75)

        self.t0 = millis_now()

        # self.label = OutlinedLabel(self)
        # self.label.setStyleSheet("")
        # self.label.setAlignment(Qt.AlignRight)
        # self.label.setStyleSheet("font-family: 'JetBrainsMono Nerd Font Mono', 'Consolas'; color: white; font-size: 20px; ")
        # self.label.move(0, 0)
        # self.label.setTextMask("00:00.000")
        # self.label.setText("00:00.000")
        # self.label.setOutlineThickness(10)
        # self.label.setGeometry(QRect(0, 0, 300, 100))

        # self.update_timer_thread = threading.Thread(target=self.update_timer)
        # self.update_timer_thread.start()
        # self.hotkey_thread = threading.Thread(target=self.start_hotkey_listener)
        # self.hotkey_thread.start()

        # self.markers = {
        #     # 'rect1': Marker("rectangle", (10, 10, 100, 100), QColor(255, 0, 255, 255), {"name": "rect1"}),
        # }
        # self.new_marker_signal.connect(self.add_marker)
        #threading.Timer(3, self.close).start()



    def start_hotkey_listener(self):
        # def close_handler():
        #     self.close()
        # keyboard.add_hotkey('ctrl + g', close_handler)
        # keyboard.wait()
        # while not self.close_event.is_set():
        #     time.sleep(0.01)
        # keyboard.remove_all_hotkeys()
        # print('end hotkey thread')
        ...

    def add_marker(self, marker):
        self.markers[marker.data['name']] = marker

    def closeEvent(self, event: QEvent | None) -> None:
        global close_event
        close_event.set()
        self.markers.clear()

        event.accept()

    def update_timer(self):
        millis = millis_now() - self.t0
        seconds = millis // 1000
        minutes = seconds // 60
        self.label.setText("{:02d}:{:02d}.{:03d}".format(minutes, seconds % 60, millis % 1000))
        self.update()

    def mousePressEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        event.ignore()

    def paintEvent(self, event):
        painter = QPainter(self)
        for k, marker in self.markers.items():
            if marker.marker_type == 'rectangle':
                #print(marker.geometry, '{:08h}'.format(marker.color.rgba()))
                painter.setPen(QPen(marker.color, 1))
                # painter.setBrush(QBrush(QColor(255,0,0,128)))
                painter.drawRect(QRect(*marker.geometry))
            elif marker.marker_type == 'ellipse':
                ...
                # painter.setPen(QPen(marker.color, 1))
                # painter.drawEllipse(marker.geometry)
            elif marker.marker_type == 'line':
                ...
                # painter.setPen(QPen(marker.color, 1))
                # painter.drawLine(marker.geometry)
            elif marker.marker_type == 'image':
                x, y, w, h = marker.geometry
                painter.drawImage(x, y, self.img)

class OutlinedLabel(QLabel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = 10
        self.mode = False
        self.setBrush(Qt.white)
        self.setPen(Qt.black)
        self.setStyleSheet('border: none;')
        self._text_mask = ''

    def scaledOutlineMode(self):
        return self.mode

    def setScaledOutlineMode(self, state):
        self.mode = state

    def outlineThickness(self):
        return self.w * self.font().pointSize() if self.mode else self.w

    def setOutlineThickness(self, value):
        self.w = value

    def setBrush(self, brush):
        if not isinstance(brush, QBrush):
            brush = QBrush(brush)
        self.brush = brush

    def setPen(self, pen):
        if not isinstance(pen, QPen):
            pen = QPen(pen)
        pen.setJoinStyle(Qt.RoundJoin)
        self.pen = pen

    def sizeHint(self):
        w = math.ceil(self.outlineThickness() * 2)
        return super().sizeHint() + QSize(w, w)
    
    def minimumSizeHint(self):
        w = math.ceil(self.outlineThickness() * 2)
        return super().minimumSizeHint() + QSize(w, w)

    def setTextMask(self, text):
        self._text_mask = text

    def text_mask(self):
        return self._text_mask
    
    def paintEvent(self, event):
        w = int(self.outlineThickness())
        rect = self.rect()
        metrics = QFontMetrics(self.font())
        tr = metrics.boundingRect(self.text_mask()).adjusted(0, 0, w, w)
        if self.indent() == -1:
            if self.frameWidth():
                indent = (metrics.boundingRect('x').width() + w * 2) / 2
            else:
                indent = w
        else:
            indent = self.indent()

        if self.alignment() & Qt.AlignLeft:
            x = rect.left() + indent - min(metrics.leftBearing(self.text()[0]), 0)
        elif self.alignment() & Qt.AlignRight:
            x = rect.x() + rect.width() - indent - tr.width()
        else:
            x = (rect.width() - tr.width()) / 2
            
        if self.alignment() & Qt.AlignTop:
            y = rect.top() + indent + metrics.ascent()
        elif self.alignment() & Qt.AlignBottom:
            y = rect.y() + rect.height() - indent - metrics.descent()
        else:
            y = (rect.height() + metrics.ascent() - metrics.descent()) / 2

        path = QPainterPath()
        path.addText(x, y, self.font(), self.text())
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)

        self.pen.setWidthF(4)
        qp.strokePath(path, self.pen)
        if 1 < self.brush.style() < 15:
            qp.fillPath(path, self.palette().window())
        qp.fillPath(path, self.brush)


def accept_command(win, app):
    '''
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5124")
    # socket.setsockopt(zmq.RCVTIMEO, 200)
    # socket.setsockopt(zmq.SNDTIMEO, 200)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    while not close_event.is_set():
        try:
            socks = dict(poller.poll(300))
            if socket in socks and socks[socket] == zmq.POLLIN:
                message = socket.recv_json()
                if message:
                    marker = json_to_marker(message)
                    socket.send_string("Received")
                    if marker.data['action'] == 'add':
                        win.new_marker_signal.emit(marker)
                    elif marker.data['action'] == 'remove':
                        if marker.data['name'] in win.markers:
                            del win.markers[marker.data['name']]
                    elif marker.data['action'] == 'add_image':
                        data = socket.recv()
                        socket.send_string('Received')
                        x, y, w, h = marker.geometry
                        win._img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                        win.img = QImage(win._img.data, w, h, w*3, QImage.Format.Format_RGB888)
                        win.new_marker_signal.emit(marker)
                    elif marker.data['action'] == 'test':
                        print('test command')
                    elif marker.data['action'] == 'exit':
                        close_event.set()
                        print('exit command')
        except zmq.error.ContextTerminated as e:
            print(f'ContextTerminated: {e}')
            break
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                print(f'retry')
                continue
            print(f'ZMQError: {e}')
            break
    print('end accept command thread')
    if not close_event.is_set():
        close_event.set()
    poller.unregister(socket)
    socket.close()
    '''
    global close_event
    while not close_event.is_set():
        time.sleep(0.01)
    # time.sleep(500)
    # win.close()
    print('exit command thread')

if __name__ == '__main__':
    #json_to_marker('{"marker_type": "rectangle", "geometry": [10, 10, 100, 100], "color": [255, 0, 0, 255], "data": {"name": "rect1"}}')
    app = QApplication(sys.argv)
    window = TransparentWindow()
    s = app.primaryScreen().size()
    window.setGeometry(0, 0, s.width(), s.height())
    cmd_thread = threading.Thread(target=accept_command, args=(window, app))
    cmd_thread.start()
    window.show()
    c = app.exec()
    cmd_thread.join()
    print('end of main')
    # sys.exit(c)
