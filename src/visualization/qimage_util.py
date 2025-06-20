
import numpy as np
import cv2

from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import QSize

def ndarray_to_gray(img: np.ndarray) -> QImage:
    
    assert(len(img.shape) == 2), "Only 2D array can be properly interpreted as image"

    w, h = img.shape
    img = img * 255
    img = np.require(img, np.uint8, 'C')
    return QImage(img.data, w, h, QImage.Format.Format_Grayscale8)


def ndarray_to_rgb(img: np.ndarray) -> QImage:
    
    assert(len(img.shape) == 3), "Only 2D array with channels can be properly interpreted as image"

    w, h, c = img.shape

    assert(c == 3), "RGB needs 3 channels"

    img = img * 255
    img = np.require(img, np.uint8, 'C')
    return QImage(img.data, w, h, c*w, QImage.Format.Format_RGB888)

def ndarray_to_rgb_native(img: np.ndarray) -> QImage:
    
    assert(len(img.shape) == 3), "Only 2D array with channels can be properly interpreted as image"
    h, w, c = img.shape
    assert(c == 3), "RGB needs 3 channels"

    img = np.require(img, np.uint8, 'C')
    return QImage(img.data, w, h, 3*w, QImage.Format.Format_RGB888)

def pixmap_to_ndarray(pm: QPixmap, w: int, h: int):
    img = pm.toImage().scaled(QSize(w, h))
    buf = img.bits()
    buf = buf.tobytes()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((w, h, 4))
    return img

def pixmap_to_gray_ndarray(pm: QPixmap, w: int, h: int):
    img = pixmap_to_ndarray(pm, w, h)
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return gray

