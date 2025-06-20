import sys
import os

from os.path import join

from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
from src.utils.util_enumy import *
import numpy as np
from dotenv import load_dotenv
import torch

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation

sys.path.insert(0, '../../') # noqa
from typing import Dict, Any, Tuple

from src.utils.VisUtils import OutputSliceRetreiver
from src.utils.helper import merge_img_label_gt
from src.utils.DataFlowHandler import DataFlowHandler, ResultGetters, AgentProcArgType

load_dotenv()



class OverlayedAnimationWidget(QWidget):
    def __init__(self, init_data, init_src, config, dataflow_handler: DataFlowHandler):
        super().__init__()
        self.config = config

        self.dataflow_handler = dataflow_handler
        self.current_output = init_data
        self.current_src_image = init_src

        self.eventloop_augmentation = None


        layout = QGridLayout(self)

        self.max_z = len(self.current_output[0][0][0])
        self.max_step = len(self.current_output)

        self.z = int(self.max_z/2)
        self.step = self.max_step - 1


        self.title = "Steps and Z-Axis"
        self.setWindowTitle(self.title)

        self.dynamic_canvas = FigureCanvas(Figure())
        layout.addWidget(self.dynamic_canvas, 0, 0, 1, 5)
        self._dynamic_ax = self.dynamic_canvas.figure.add_subplot(111)
        # Set up a Line2D.

        self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
        self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
        self._timer = self.dynamic_canvas.new_timer(200)
        self.is_on = False
        self._timer.add_callback(self._update_canvas)

        self.button_zr = QPushButton(self)
        self.button_zr.setText("z+")
        def button_zr_clicked():
            self.z = (self.z + 1) % self.max_z
            self._dynamic_ax.clear()
            self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
            self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
            self.dynamic_canvas.draw()

        self.button_zl = QPushButton(self)
        self.button_zl.setText("z-")
        def button_zl_clicked():
            self.z = (self.z - 1) % self.max_z
            self._dynamic_ax.clear()
            self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
            self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
            self.dynamic_canvas.draw()

        self.button_sr = QPushButton(self)
        self.button_sr.setText("step+")
        def button_sr_clicked():
            self.step = (self.step + 1) % self.max_step
            self._dynamic_ax.clear()
            self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
            self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
            self.dynamic_canvas.draw()
            self._timer.stop()
            self.is_on = False
            self.button_start.setText("start")

        self.button_sl = QPushButton(self)
        self.button_sl.setText("step-")
        def button_sl_clicked():
            self.step = (self.step - 1) % self.max_step
            self._dynamic_ax.clear()
            self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
            self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
            self.dynamic_canvas.draw()
            self._timer.stop()
            self.is_on = False
            self.button_start.setText("start")

        self.button_start = QPushButton(self)
        self.button_start.setText("start")
        def button_start_clicked():
            if self.is_on:
                self._timer.stop()
                self.is_on = False
                self.button_start.setText("start")
            else:
                self._timer.start()
                self.is_on = True
                self.button_start.setText("stop")

        
        
        self.button_zr.clicked.connect(button_zr_clicked)
        self.button_zl.clicked.connect(button_zl_clicked)

        self.button_sr.clicked.connect(button_sr_clicked)
        self.button_sl.clicked.connect(button_sl_clicked)

        self.button_start.clicked.connect(button_start_clicked)

        layout.addWidget(self.button_zr, 1, 1, 1, 1)
        layout.addWidget(self.button_zl, 1, 0, 1, 1)
        layout.addWidget(self.button_sr, 1, 4, 1, 1)
        layout.addWidget(self.button_sl, 1, 2, 1, 1)
        layout.addWidget(self.button_start, 1, 3, 1, 1)

    # callback for animation
    def _update_canvas(self):

        self.step = (self.step + 1) % self.max_step
        self._dynamic_ax.clear()
        self._dynamic_ax.imshow(self.get_current_slice(self.step, self.z))
        self._dynamic_ax.set_title("Z-Axis:" + str(self.z) + ", Step:" + str(self.step))
        self.dynamic_canvas.draw()

    def setup_remote_output_getter(self, dfh: DataFlowHandler):
        def handle_new_result(data: Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]):        
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            if my_data[0] == FlowDataChangeType.ABSDIFF:
                aa, data = my_data
                self.current_output = data
                
            elif my_data[0] == FlowDataChangeType.NETWORK:
                my_data = my_data[1]
                if len(my_data)==1:
                    return
                _, data, fname = my_data
                self.current_output = data
                self.current_src_image, _ = self.dataflow_handler.get_image_for_filename(fname)
        dfh.add_network_handler(handle_new_result)
    def get_current_slice(self, step: int, slice_index: int):
        """
            Get slice at slice_index from current datapoint and merge labels with base image
        """
        retriever = OutputSliceRetreiver(self.current_output, self.config)
        mask_slice = retriever.get_output_slice(step, 2, slice_index, True)
        img_slice = self.current_src_image[:, :, slice_index]
        image = merge_img_label_gt(img_slice, mask_slice, np.zeros(img_slice.shape))
        return image

