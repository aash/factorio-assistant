import sys
import time
import threading
from dataclasses import dataclass
import ahk

from PyQt5.QtGui import QCloseEvent, QPainter, QColor, QPen, QBrush, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QGraphicsLayout, QBoxLayout, QSizePolicy, QLabel
from PyQt5.QtCore import QRect, Qt, pyqtSignal
import numpy as np

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

def json_to_marker(json_string):
    data = json.loads(json_string)
    return Marker(
        marker_type=data['marker_type'],
        geometry=tuple(data['geometry']),
        color=QColor(*data['color']),
        data=data['data']
    )

close_event = threading.Event()


class TransparentWindow(QMainWindow):
    new_marker_signal = pyqtSignal(Marker)
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint |
                            Qt.WindowType.WindowTransparentForInput |
                            Qt.WindowType.WindowStaysOnTopHint
                            #| Qt.WindowType.Tool
                            )
        self.setAttribute(Qt.WA_TranslucentBackground)


        self.w = QWidget()
        #self.w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.w.setFixedSize(1920, 1200)
        l = QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        self.w.setLayout(l)
        self.setCentralWidget(self.w)

        self.w.setStyleSheet("border: 2px dashed green")

        self.setWindowOpacity(0.75)

        self.t0 = millis_now()

        self.label = OutlinedLabel(self)
        self.label.setStyleSheet("")
        self.label.setAlignment(Qt.AlignRight)
        self.label.setStyleSheet("font-family: 'Consolas'; color: white; font-size: 20px; ")
        self.label.move(0, 0)
        self.label.setTextMask("00:00.000")
        self.label.setText("00:00.000")
        self.label.setOutlineThickness(10)
        self.label.setGeometry(QRect(0, 0, 300, 100))

        self.update_timer_thread = threading.Thread(target=self.update_timer)
        self.update_timer_thread.start()
        # self.hotkey_thread = threading.Thread(target=self.start_hotkey_listener)
        # self.hotkey_thread.start()
        self.command_proc_thread = threading.Thread(target=self.accept_command)
        self.command_proc_thread.start()
        self.update_signal.connect(self.update)

        self.markers = {
            # 'rect1': Marker("rectangle", (10, 10, 100, 100), QColor(255, 0, 255, 255), {"name": "rect1"}),
        }
        self.new_marker_signal.connect(self.add_marker)
        #threading.Timer(3, self.close).start()

    def accept_command(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5124")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        while not close_event.is_set():
            try:
                socks = dict(self.poller.poll(1000))
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    message = self.socket.recv_json()
                    if message:
                        self.socket.send_string("Received")
                        marker = json_to_marker(message)
                        if marker.data['action'] == 'add':
                            self.new_marker_signal.emit(marker)
                        elif marker.data['action'] == 'remove':
                            if marker.data['name'] in self.markers:
                                del self.markers[marker.data['name']]
                        elif marker.data['action'] == 'add_image':
                            data = self.socket.recv()
                            self.socket.send_string('Received')
                            x, y, w, h = marker.geometry
                            self._img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                            self.img = QImage(self._img.data, w, h, w*3, QImage.Format.Format_RGB888)
                            self.new_marker_signal.emit(marker)
                            print(f'add image {len(data)} geometry: {marker.geometry}')

            except zmq.error.ContextTerminated as e:
                print(f'ContextTerminated: {e}')
                break
            except zmq.error.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    continue
                print(f'ZMQError: {e}')
                break
            except zmq.error.Again:
                pass
        print('end accept command thread')
        self.socket.close()
        self.context.term()
 

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

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        close_event.set()
        print('close event')

        threading.Timer(0.5, self.join_threads).start()

        return super().closeEvent(a0)

    def join_threads(self):
        self.update_timer_thread.join()
        # self.hotkey_thread.join()
        # self.command_proc_thread.join()
        # keyboard.unhook_all()
        # keyboard.unhook_all_hotkeys()
        QApplication.quit()

    def update_timer(self):
        while not close_event.is_set():
            time.sleep(0.01)
            millis = millis_now() - self.t0
            seconds = millis // 1000
            minutes = seconds // 60
            # if not close_event.is_set():
            self.label.setText("{:02d}:{:02d}.{:03d}".format(minutes, seconds % 60, millis % 1000))
            self.update_signal.emit()
            # print(f'update timer thread {millis}')
        print('timer thread ended')

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

import math
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QBrush, QPen, QFontMetrics, QPainterPath, QPainter
import time

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


if __name__ == '__main__':
    #json_to_marker('{"marker_type": "rectangle", "geometry": [10, 10, 100, 100], "color": [255, 0, 0, 255], "data": {"name": "rect1"}}')
    app = QApplication(sys.argv)
    window = TransparentWindow()
    # a = ahk.AHK()
    s = app.primaryScreen().size()
    window.setGeometry(0, 0, s.width(), s.height())
    window.show()
    # a.add_hotkey('^g', callback=window.close)
    # a.start_hotkeys()
    app.exec()
    print('end of main')
    sys.exit(0)
