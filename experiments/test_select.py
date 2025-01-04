import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QPoint
from PySide6 import QtCore
from PySide6 import QtGui


if False:
    # Create a blank canvas
    width, height = 1600, 1600
    canvas = np.zeros((height, width), dtype=np.uint8)

    # Create random points for the blob
    num_points = 170
    x = np.random.randint(0, width, num_points)
    y = np.random.randint(0, height, num_points)
    r = np.random.randint(20, 50, num_points)

    # Set the points to 1 to create the blob
    for y_, x_, r_ in zip(y, x, r):

        cv2.circle(canvas, (y_,x_), r_, (255), -1)
    canvas[y, x] = 255

    # Apply Gaussian filter to make it smooth
    canvas = gaussian_filter(canvas, sigma=25, )
    _, canvas = cv2.threshold(canvas, 128, 255, cv2.THRESH_BINARY)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGRA)
    blob = canvas
else:
    opn = cv2.imread('tmp/opn.png')
    height, width, _ = opn.shape
    opn = cv2.cvtColor(opn, cv2.COLOR_BGR2GRAY)
    canvas_rgb = cv2.cvtColor(opn, cv2.COLOR_GRAY2BGRA)
    blob = opn

# Find connected components
num_labels, labels_im = cv2.connectedComponents(blob.astype(np.uint8))

# Find contours of the blob
contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Approximate the contours to a polygon
polygons = []
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.005 * cv2.arcLength(contour, True)  # 1% of the contour length
    approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
    polygons.append(approx_polygon)

# Optionally, draw the approximated polygons on the original image
output_image = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)  # Convert to BGR for coloring
for polygon in polygons:
    cv2.polylines(output_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=3)


class ImageWindow(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Connected Components Highlight")
        
        # Store the original image
        self.original_image = image
        
        # Create QLabel to show the image
        self.label = QLabel(self)
        self.update_image(image)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        
        # container = QWidget()
        # container.setLayout(layout)
        self.setCentralWidget(self.label)
        self.plblid = 0
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        print('init finished')
        
    def update_image(self, image):
        """Update the QLabel with a QPixmap created from the given image."""
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_ARGB32)
        self.label.setPixmap(QPixmap.fromImage(q_image))
        
    # def mouseMoveEvent(self, event):
        # print('mouseMoveEvent: pos {}'.format(event.pos()))

    def eventFilter(self, source, event):
        # if event.type() != QtCore.QEvent.MouseMove:
            # return
        # if hasattr(event, 'buttons') and event.buttons() != QtCore.Qt.NoButton:
            # return
        # if hasattr(event, 'x') and (event.x() < 0 or event.x() >= self.width() or event.y() < 0 or event.y() >= self.height()):
            # return
        # if not hasattr(event, 'x'):
            # return
        # Get the position of the cursor
        local_pos = self.label.mapFromGlobal(event.globalPosition().toPoint())
        # cursor_pos = QPoint(event.x(), event.y())
        # print('mouseMoveEvent: pos {}'.format(event.pos()))
        # print(cursor_pos)
        # Get the label of the connected component under the cursor
        label_id = labels_im[local_pos.y(), local_pos.x()]
        
        if label_id > 0:
            # if self.plblid != label_id:
                # print(label_id)
                # self.plblid = label_id
            # Create an image to display the highlighted connected component
            highlighted_image = self.original_image.copy()

            # Highlight the connected component
            highlighted_image[labels_im == label_id] = (0, 255, 0, 255)
            save_img = np.zeros_like(self.original_image)
            save_img[labels_im == label_id] = (255,255,255,255)
            cv2.imwrite(f'contour_{label_id}.png', save_img)

            # Combine with the original blob to keep the visual info
            

            # combined_image = cv2.addWeighted(self.original_image, 0.5, highlighted_image, 0.5, 0)
            self.update_image(highlighted_image)
        else:
            self.update_image(self.original_image)
        # return QtGui.QMainWindow.eventFilter(self, source, event)

if __name__ == "__main__":
    # Start the application
    app = QApplication(sys.argv)
    
    # Create the image window
    window = ImageWindow(canvas_rgb)
    window.setMouseTracking(True)
    window.resize(width, height)
    window.show()
    app.installEventFilter(window)
    print('qwe')
    
    # Start the event loop
    sys.exit(app.exec())