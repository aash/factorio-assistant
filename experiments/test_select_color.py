import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QStyle
from PySide6.QtGui import QImage, QPixmap, QCursor
from PySide6.QtCore import Qt, QPoint, QSize
from PySide6 import QtCore
from PySide6 import QtGui


opn = cv2.imread('experiments/tmp/map.png')
height, width, _ = opn.shape
opn = cv2.resize(opn, None, fx=0.25, fy=0.25)
opn = cv2.cvtColor(opn, cv2.COLOR_BGR2RGB)


class ImageWindow(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Connected Components Highlight")
        
        # Store the original image
        self.original_image = image
        
        # Create QLabel to show the image
        self.label = QLabel(self)
        self.cl_lbl = QLabel(self)
        self.cl_lbl.setFixedSize(150,50)
        self.cl_lbl.setStyleSheet("background-color: rgb(0,128,0);")
        self.update_image(image)
        
        layout = QVBoxLayout()
        layout.addWidget(self.cl_lbl)
        layout.addWidget(self.label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.plblid = 0
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        self.label.installEventFilter(self)
        self.label.setStyleSheet('border: 1px solid red; ')
        # self.label.setAlignment()
        print('init finished')
        self.root_widget = self
        
    def update_image(self, image):
        """Update the QLabel with a QPixmap created from the given image."""
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_image))
        h, w, _ = self.original_image.shape
        # self.label.resize(QSize(w, h))
        self.label.setFixedSize(QSize(w+2,h+2))
        print(self.label.pixmap().size(), self.original_image.shape)
        
    #def mouseMoveEvent(self, event):
        # print('mouseMoveEvent: pos {}'.format(event.pos()))

    def eventFilter(self, src, event):
        if event.type() != QtCore.QEvent.Type.MouseMove:
            return False
        # print(event.type(), type(event), hasattr(event, "x"))
        # return True
        # if hasattr(event, 'buttons') and event.buttons() != QtCore.Qt.NoButton:
            # return
        # if hasattr(event, 'x') and (event.x() < 0 or event.x() >= self.width() or event.y() < 0 or event.y() >= self.height()):
            # return
        # if not hasattr(event, 'x'):
            # return
        # Get the position of the cursor
        # p = QPoint(event.localPos().x(), event.localPos().y())
        # self.cl_lbl.text = f'{p}'
        # local_pos = self.label.mapFromGlobal(event.localPos())
        # local_pos.x()
        # self.update()
        # position = self.label.mapTo(self, self.label.pos())
        position = self.label.pos()
        # top_left_lbl = self.label.mapToGlobal(self.label.pos())
        tl = np.array(position.toTuple())
        # print(tl)
        local_pos = np.array(event.position().toPoint().toTuple())
        # self.cl_lbl.setText(f'{local_pos.x()},{local_pos.y()}\n{tl}')
        # self.cl_lbl.setText(f'{local_pos}\n{tl}')
        # QCursor.pos()
        # local_pos = local_pos - 
        # local_pos = event.localPos().toPoint()
        # if hasattr(event, 'buttons') and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
        x, y = local_pos
        h, w, _ = opn.shape
        if x < w and y < h:
            r, g, b = opn[local_pos[1], local_pos[0]]

            self.cl_lbl.setStyleSheet(f'background-color: rgb({r},{g},{b});')
            self.cl_lbl.setText(f'{r},{g},{b}')
            
        else:
            self.cl_lbl.setStyleSheet(f'background-image: linear-gradient(to right, red, blue);')
        #print(r,g,b)
        # cursor_pos = QPoint(event.x(), event.y())
        # print('mouseMoveEvent: pos {}'.format(event.pos()))
        # print(cursor_pos)
        # Get the label of the connected component under the cursor
        return True

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    
    # Create the image window
    window = ImageWindow(opn)
    window.setMouseTracking(True)
    window.resize(width, height)
    window.show()
    # app.installEventFilter(window)
    print('qwe')
    
    # Start the event loop
    sys.exit(app.exec())