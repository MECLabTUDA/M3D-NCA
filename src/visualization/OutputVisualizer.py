"""
Contains Visualizer Infrastructure for individual Output Channels.
"""
import matplotlib
matplotlib.use('QtAgg')

from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
from src.utils.util_enumy import *
from qtpy import QtWidgets
from typing import Dict, Any, Tuple
import os
from os.path import expanduser
import numpy as np
from src.visualization.AnimatedHeatMap import AnimatedHeatMap
from src.visualization.Surface3Dplot import Surface3Dplot
from qtpy.QtWidgets import QStackedWidget
from src.utils.DataFlowHandler import ResultGetters, DataFlowHandler, AgentProcArgType
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
Contains the AnimatedHeatMap along with a few buttons to control display.
"""
class OutputVisualizer(QtWidgets.QWidget):

    def __init__(self, stepsDict: Dict[int, np.ndarray], net_config: Dict[str, Any], step_axis: int,  parent=None, start_step: int = 0, axis_step: int = 0, 
                  channel: int = None):
        super(OutputVisualizer, self).__init__()
        self._render_video_button = QtWidgets.QPushButton()
        self._render_video_button.setText("render")
        self._render_frame_button = QtWidgets.QPushButton()
        self._render_frame_button.setText("render frame")
        layout = QtWidgets.QGridLayout(self)
        self._canvas = AnimatedHeatMap(stepsDict=stepsDict, net_config=net_config, step_axis=step_axis, 
                                    parent=parent, start_step=start_step, axis_step=axis_step, channnel=channel)
        

        layout.addWidget(self._canvas, 0, 0, 1, 5)
        self._up_axis_btn = QtWidgets.QPushButton()
        self._up_axis_btn.setText("up")
        self._axis_step_label = QtWidgets.QLabel()
        self._axis_step_label.setText("Step: " + str(axis_step))
        self._axis_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._down_axis_btn = QtWidgets.QPushButton()
        self._down_axis_btn.setText("down")
        
        self.render_step: int = 1        
        self.step_render_spinner: QSpinBox = QSpinBox()
        self.step_render_spinner.setRange(1, 10)
        self.step_render_spinner.setValue(1)
        
        self.render_step_label: QLabel = QLabel("Render Step: " + str(self.render_step))
        
        self.render_steps_button: QPushButton = QPushButton("Stepped Render")
        
        def _handle_do_stepped_render(oof = None):
            folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
            folderpath = str(folderpath)
            self._canvas.render_image_series(folderpath=folderpath, interval=self.render_step)
            
        def _handle_render_step_value_changed(val: int):
            self.render_step = val
            self.render_step_label.setText("Render Step: " + str(self.render_step))
        
        def _do_render():
            filename = QFileDialog.getSaveFileName(self, caption="Rendering",dir=os.getcwd(), filter="Images (*.gif)")
            self._canvas.do_render(filepath=filename[0])
        self._render_video_button.clicked.connect(_do_render)

        def _do_render_frame():
            filename = QFileDialog.getSaveFileName(self, caption="Rendering",dir=os.getcwd(), filter="Images (*.png)")
            self._canvas.render_current_img(filepath=filename[0])
        self._render_frame_button.clicked.connect(_do_render_frame)

        def down_axis():
            ret = self._canvas.set_axis_step(-1)
            self._axis_step_label.setText("Step: " + str(ret))
        
        def up_axis():
            ret = self._canvas.set_axis_step(1)
            self._axis_step_label.setText("Step: " + str(ret))
        
        self._up_axis_btn.clicked.connect(up_axis)
        self._down_axis_btn.clicked.connect(down_axis)
       
        self._playButton = QtWidgets.QPushButton()
        self._playButton.setText("Stop")
        self._canvas.start_stop_label = self._playButton
        self._leftButton = QtWidgets.QPushButton()
        self._leftButton.setText("<<")
        self._rightButton = QtWidgets.QPushButton()
        self._rightButton.setText(">>")
        def fun():
            ret = self._canvas.switch_running()
            self._playButton.setText(ret)
        def left():
            self._canvas.set_pending_step(-1)
        def right():
            self._canvas.set_pending_step(1)
        self._leftButton.clicked.connect(left)
        self._rightButton.clicked.connect(right)

        def handle_change_applyinterpolation(args=None):
            if self._apply_interpol_checkbox.isChecked():
                self._canvas.set_apply_transformation_func(True)
            else:
                self._canvas.set_apply_transformation_func(False)
                

        self._apply_interpol_checkbox = QtWidgets.QCheckBox()
        self._apply_interpol_checkbox.setChecked(True)
        self._apply_interpol_checkbox.stateChanged.connect(handle_change_applyinterpolation)
        self._apply_interpol_label = QtWidgets.QLabel()
        self._apply_interpol_label.setText("Apply:")
        self._playButton.clicked.connect(fun)

        self._color_mapping_chooser = QComboBox(self)
        self._color_mapping_chooser.addItems(self._canvas.color_map)
        def handle_color_change(index: int):
            self._canvas.set_color_map(self._canvas.color_map[index])
        self._color_mapping_chooser.setCurrentIndex(7)
        self._color_mapping_chooser.currentIndexChanged.connect(handle_color_change)
        
        
        self.render_steps_button.clicked.connect(_handle_do_stepped_render)
        self.step_render_spinner.valueChanged.connect(_handle_render_step_value_changed)
        

        layout.addWidget(self._canvas, 0, 0, 1, 5)
        layout.addWidget(self._up_axis_btn, 1, 3, 1, 2)
        layout.addWidget(self._axis_step_label, 1, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._down_axis_btn, 1, 0, 1, 2)
        layout.addWidget(self._leftButton, 2, 0, 1, 1)
        layout.addWidget(self._playButton, 2, 1, 1, 1)
        layout.addWidget(self._rightButton, 2, 2, 1, 1)
        layout.addWidget(self._render_video_button, 2, 3, 1, 1)
        layout.addWidget(self._color_mapping_chooser, 2, 4, 1, 1)
        layout.addWidget(self._apply_interpol_label, 3, 0, 1, 1)
        layout.addWidget(self._apply_interpol_checkbox, 3, 1, 1, 1)
        layout.addWidget(self._render_frame_button, 3, 2, 1, 1)
        layout.addWidget(self.render_step_label, 3, 3, 1, 1)
        layout.addWidget(self.step_render_spinner, 3, 4, 1, 1)
        layout.addWidget(self.render_steps_button, 3, 5, 1, 1)
        
        self.setLayout(layout)

    def setup_remote_output_getter(self, dfh: DataFlowHandler):
        def handle_new_result(data: Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]):        
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            if my_data[0] == FlowDataChangeType.ABSDIFF:
                data = my_data[1]
                self._canvas.set_new_data(data)
            elif my_data[0] == FlowDataChangeType.NETWORK:
                my_data = my_data[1]
                if len(my_data)==1:
                    return
                _, data, fname = my_data
                self._canvas.set_new_data(data)
        dfh.add_network_handler(handle_new_result)