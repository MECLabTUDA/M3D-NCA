from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *

# source: https://stackoverflow.com/a/41068447

class QHLine(QFrame):
    def __init__(self):
        """
            Creates a horizontal line which can be used as a visual separator in QT
        """

        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class QVLine(QFrame):
    def __init__(self):
        """
            Creates a vertical line which can be used as a visual separator in QT
        """
        super(QVLine, self).__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)

#