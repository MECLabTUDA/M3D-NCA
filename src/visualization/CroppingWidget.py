import matplotlib
matplotlib.use('QtAgg')

from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
from src.utils.util_enumy import *
from qtpy import QtWidgets
from typing import Dict, Any, Tuple
import os

import numpy as np

from src.visualization.SegmentedPseudo1DScatterPlot import SegmentedPseudo1DScatterPlot

from src.utils.DataFlowHandler import DataFlowHandler, AgentProcArgType
from src.utils.MathHelper import MathHelper
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
Intended to set cropping range for the generated mapping. Restrict mapping to limited input value range to either eliminate outliers 
or generate a visually more interesting mapping.
"""
class CroppingWidget(QtWidgets.QWidget):
    layout: QtWidgets.QGridLayout = None
    scatterplot: SegmentedPseudo1DScatterPlot = None
    lower_crop_val: float
    higher_crop_val: float
    current_data: Dict[int, np.ndarray] = None
    val1_slider: QtWidgets.QSlider = None
    val2_slider: QtWidgets.QSlider = None
    apply_button: QtWidgets.QPushButton = None
    data_lower_bound: float
    data_higher_bound: float
    flat_data: np.ndarray = None
    current_channel: int = 0
    slider_lower_bound: int = 0
    slider_higher_bound: int = 1000
    cropping_change_group: str = "cropping_change"
    flowhandler: DataFlowHandler = None
    pdf_data: Tuple[np.ndarray, np.ndarray] = None
    def __init__(self, stepsDict: Dict[int, np.ndarray], flowhandler: DataFlowHandler,  parent=None, channel: int = 0, initial_crop: Tuple[float, float] = None):
        """Initializes the cropping Widget."""
        super(CroppingWidget, self).__init__(parent=parent)
        self.current_channel = channel        
        self.current_data = stepsDict
        flat_data = MathHelper.get_flat_channel(self.current_data, self.current_channel)
        self.flat_data = MathHelper.get_subsampled_array_of_desired_size(flat_data, 10000)
        self.data_lower_bound = self.flat_data.min()
        self.data_higher_bound = self.flat_data.max()
        if initial_crop is None:
            self.lower_crop_val = self.data_lower_bound
            self.higher_crop_val = self.data_higher_bound
        else:
            if initial_crop[0] < self.data_lower_bound:
                self.lower_crop_val = self.data_lower_bound
            else:
                self.lower_crop_val = initial_crop[0]

            if initial_crop[1] > self.data_higher_bound:
                self.higher_crop_val = self.data_higher_bound
            else:
                self.higher_crop_val = initial_crop[1]
        
        self.layout = QtWidgets.QGridLayout(self)
        self.scatterplot = SegmentedPseudo1DScatterPlot(self.flat_data, parent=self)
        self.layout.addWidget(self.scatterplot, 0, 0, 3, 5)
        self.val1_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.val1_slider.setRange(self.slider_lower_bound, self.slider_higher_bound)
        self.val1_slider.setSingleStep(1)
        self.val1_slider.setValue(self._translate_crop_value_to_slider_value(self.lower_crop_val))
        self.val1_slider.sliderReleased.connect(lambda val=None: self._handle_slider_release(1, val))
        self.apply_button = QtWidgets.QPushButton()
        self.apply_button.setText("Apply")
        self.apply_button.clicked.connect(self.apply_cropping)
        self.val2_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.val2_slider.setRange(self.slider_lower_bound, self.slider_higher_bound)
        self.val2_slider.setSingleStep(1)
        self.val2_slider.setValue(self._translate_crop_value_to_slider_value(self.higher_crop_val))
        self.val2_slider.sliderReleased.connect(lambda val=None: self._handle_slider_release(2, val))
        self.layout.addWidget(self.val1_slider, 3, 0, 1, 2)
        self.layout.addWidget(self.val2_slider, 3, 2, 1, 2)
        self.layout.addWidget(self.apply_button, 3, 4, 1, 1)
        xs, ys = MathHelper.get_pdf_for_data(self.flat_data)
        self.pdf_data = (xs, ys)
        self.scatterplot.draw_pdf(xs, ys)
        vlines = np.array([self.lower_crop_val, self.higher_crop_val])
        self.scatterplot.update_vlines(vline_x_pos=vlines)
        self.scatterplot._redrar_vlines()
        self.flowhandler = flowhandler
        self.flowhandler.add_network_handler(self._handle_new_result)
        self.flowhandler.create_group(self.cropping_change_group)



    def _translate_crop_value_to_slider_value(self, crop_value: float):
        temp = (crop_value - self.data_lower_bound)/(self.data_higher_bound - self.data_lower_bound)
        temp = int(self.slider_lower_bound + temp*(self.slider_higher_bound-self.slider_lower_bound))
        if temp < self.slider_lower_bound:
            temp = self.slider_lower_bound
        if temp > self.slider_higher_bound:
            temp = self.slider_higher_bound
        return temp
        

    def _translate_slider_value_to_cropping_value(self, slider_value: int):
        temp_val = float(slider_value - self.slider_lower_bound)/float(self.slider_higher_bound - self.slider_lower_bound)
        return self.data_lower_bound + temp_val*(self.data_higher_bound - self.data_lower_bound)
    
    """
    Ensures that the left slider always controls the left limit. +/-0.1 is to prevent an empty value range which causes errors.
    """
    def _handle_slider_release(self, moving_slider: int, arg = None):
        val1 = self.val1_slider.value()
        val2 = self.val2_slider.value()
        val1 = self._translate_slider_value_to_cropping_value(val1)
        val2 = self._translate_slider_value_to_cropping_value(val2)
        if val1 < val2:
            self.lower_crop_val = val1
            self.higher_crop_val = val2
        elif val2 < val1:
            if moving_slider == 1:
                self.higher_crop_val = val2 + 0.1
                self.lower_crop_val = val2 - 0.1
                slider1_pos = self._translate_crop_value_to_slider_value(self.lower_crop_val)
                self.val1_slider.setValue(slider1_pos)

            elif moving_slider == 2:
                self.higher_crop_val = val1 + 0.1
                self.lower_crop_val = val1 - 0.1
                slider2_pos = self._translate_crop_value_to_slider_value(self.higher_crop_val)
                self.val2_slider.setValue(slider2_pos)
                #self.val2_slider.update()
                
        else:
            self.lower_crop_val = val1 - 0.1
            self.higher_crop_val = val1 + 0.1
        vlines = np.array([self.lower_crop_val, self.higher_crop_val])
        self.scatterplot.update_vlines(vlines)
        self.scatterplot._redrar_vlines()
        
    def apply_cropping(self, arg = None):
        self.flowhandler.call_group_signal(group=self.cropping_change_group, data=(self.lower_crop_val, self.higher_crop_val), dummy_process_name="McCropper", job_id=self.flowhandler._get_uuid())


    def enable(self):
        self.val1_slider.setEnabled(True)
        self.val2_slider.setEnabled(True)

    def disable(self):
        self.val1_slider.setEnabled(False)
        self.val2_slider.setEnabled(False)

    def _set_new_data(self, stepsDict: Dict[int, np.ndarray]):
        self.current_data = stepsDict
        flat_data = MathHelper.get_flat_channel(self.current_data, self.current_channel)
        self.data_lower_bound = flat_data.min()
        self.data_higher_bound = flat_data.max()
        self.flat_data = MathHelper.get_subsampled_array_of_desired_size(flat_data, 10000)
        xs, ys = MathHelper.get_pdf_for_data(self.flat_data)
        self.pdf_data = (xs, ys)
        vlines = np.array([self.lower_crop_val, self.higher_crop_val])
        self.scatterplot.update_vlines(vline_x_pos=vlines)
        self.scatterplot.update_vlines(vline_x_pos=vlines)
        self.scatterplot.redraw_scatter_points(self.flat_data)
        if not self.pdf_data is None:
            self.scatterplot.kill_pdf(self.pdf_data[0], self.pdf_data[1])
            self.scatterplot.draw_pdf(self.pdf_data[0], self.pdf_data[1])


    def _set_channel(self, channel: int):
        self.current_channel = channel        
        flat_data = MathHelper.get_flat_channel(self.current_data, self.current_channel)
        self.flat_data = MathHelper.get_subsampled_array_of_desired_size(flat_data, 10000)
        self.data_lower_bound = flat_data.min()
        self.data_higher_bound = flat_data.max()
        xs, ys = MathHelper.get_pdf_for_data(self.flat_data)
        self.pdf_data = (xs, ys)
        
        vlines = np.array([self.lower_crop_val, self.higher_crop_val])
        self.scatterplot.update_vlines(vline_x_pos=vlines)
        self.scatterplot.redraw_scatter_points(self.flat_data)
        x_pdf, y_pdf = MathHelper.get_pdf_for_data(self.flat_data)
        self.scatterplot.draw_pdf(x_pdf, y_pdf)
        


    def get_croppings(self) -> Tuple[float, float]:
        return (self.lower_crop_val, self.higher_crop_val)


    def _handle_new_result(self, data: Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]):        
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            if my_data[0] == FlowDataChangeType.ABSDIFF:
                data = my_data[1]
            elif my_data[0] == FlowDataChangeType.NETWORK:
                my_data = my_data[1]
                if len(my_data)==1:
                    return
                _, data, fname = my_data

            self._set_new_data(data)
    


    


    
    