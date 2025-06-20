import sys
import matplotlib
matplotlib.use('QtAgg')
from qtpy import QtWidgets
from typing import Dict, Any
from matplotlib.figure import Figure
import numpy as np
import matplotlib.animation as animation
from src.utils.DataFlowHandler import ResultGetters, DataFlowHandler
import os
from src.visualization.OutputMappingChooser import OutputMappingChooser
from src.visualization.OutputVisualizer import OutputVisualizer
from src.visualization.CroppingWidget import CroppingWidget
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import animation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
Wrapping widget for:
- Outputvizualizer Heatmap with animation control. 
- Cropping Widget: Widget that allows the selection of a data range to be considered for the mapping generation
- OutputMappingChooser: Widget that contains all the settings for the output mapping.


Widgets are interdependant and require knowledge about each other.
Possible improvement: Handle inter-widget communication entirely through signals in groups in the dataflowhandler.
"""

class ComboChannelVisualization(QtWidgets.QWidget):
    m_chooser: OutputMappingChooser
    visSOI: OutputVisualizer
    cropper: CroppingWidget
    def __init__(self, stepsDict: Dict[int, np.ndarray], net_config: Dict[str, Any], 
                step_axis: int,  parent=None, start_step: int = 0, axis_step: int = 0, 
                channel: int = None, default_num_segments: int = 1, 
                min_channel: int = 0, max_channel: int = 15, 
                dataflowhandler: DataFlowHandler = None, 
                ompc_groupd: str = "", ompc_mapping_process: str = "", ompc_surface_process: str = ""):
        super(ComboChannelVisualization, self).__init__()
        mainLayout = QtWidgets.QGridLayout(self)
        visualizer = OutputVisualizer(stepsDict=stepsDict, 
                                      net_config=net_config, step_axis=step_axis, 
                                      start_step=start_step, axis_step=axis_step, channel=channel)
        self.cropper = CroppingWidget(stepsDict=stepsDict, parent=self, channel=channel, flowhandler=dataflowhandler)
        chooser = OutputMappingChooser(stepsDict=stepsDict, flowhandler=dataflowhandler, start_channel=channel, default_num_segments=default_num_segments, 
                                       min_channel=min_channel, max_channel=max_channel, corresponding_heatmap=visualizer._canvas, 
                                       mapping_process_name=ompc_mapping_process, surface_process_name=ompc_surface_process, group_name=ompc_groupd)
        chooser._add_cropping_widget(self.cropper)
        self.m_chooser = chooser
        self.visSOI = visualizer
        mainLayout.addWidget(visualizer, 0, 0, 5, 4)
        mainLayout.addWidget(chooser, 5, 0, 3, 4)
        mainLayout.addWidget(self.cropper, 8, 0, 2, 4)
        self.setLayout(mainLayout)
        
    def setup_remote_output_getter(self, df_handler: DataFlowHandler):
        self.m_chooser.setup_remote_output_getter(df_handler)
