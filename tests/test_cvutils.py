
from PySide6.QtGui import QImage
import numpy as np
from cvutils import convertQImageToMat, convertMatToQImage
import pytest
import os
import cv2 as cv2
import cv2 as cv


TEST_RGBA_IMAGE = 'tests/sel.png'

def test_qimage_to_mat_conversion():
    assert(os.path.exists(TEST_RGBA_IMAGE))
    img = QImage(TEST_RGBA_IMAGE)
    assert(img.width(), img.height())
    im = convertQImageToMat(img, 4)
    assert(im.shape == (img.height(), img.width(), 4))
    im = convertQImageToMat(img, 3)
    assert(im.shape == (img.height(), img.width(), 3))
    for i in [0, 1, 2, 5, 6]:
        with pytest.raises(RuntimeError) as einfo:
            im = convertQImageToMat(img, i)

def test_mat_to_qimage_conversion():
    im = cv.imread(TEST_RGBA_IMAGE)
    qimg = convertMatToQImage(im)
    im2 = convertQImageToMat(qimg)
    assert(im.shape == im2.shape)
