"""
Contains Visualizer Infrastructure for a selected group Output Channels.
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
from src.visualization.Surface3Dplot import Surface3Dplot
from qtpy.QtWidgets import QStackedWidget
from src.visualization.AnimatedHeatMap_multi import AnimatedHeatMap_multi
from src.utils.DataFlowHandler import ResultGetters, DataFlowHandler, AgentProcArgType
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class OutputVisualizerMulitHeatMap(QtWidgets.QWidget):
    """
    Wrapper Widget handling the multi channel plots of AnimatedHeatMap_multi and its designated buttons
    """

    def __init__(self, stepsDict: Dict[int, np.ndarray], net_config: Dict[str, Any], step_axis: int,  parent=None, start_step: int = 0, axis_step: int = 0, 
                  channel: int|list[int] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        super(OutputVisualizerMulitHeatMap, self).__init__()
        self._render_video_button = QtWidgets.QPushButton()
        self._render_video_button.setText("render")
        layout = QtWidgets.QGridLayout(self)
        self.channel = channel
        self.map_color = "plasma"
        # Checks the overloaded parameter
        if isinstance(channel, list):
            # requires at least 2 channels
            if len(channel) < 2:
                channel.append(channel[0])
            self._canvas = AnimatedHeatMap_multi(stepsDict=stepsDict, net_config=net_config, step_axis=step_axis, 
                                            parent=parent, start_step=start_step, axis_step=axis_step, channnel=channel,
                                            map_color = self.map_color)
        else:
            self._canvas = AnimatedHeatMap_multi(stepsDict=stepsDict, net_config=net_config, step_axis=step_axis, 
                                        parent=parent, start_step=start_step, axis_step=axis_step, channnel=[channel, channel],
                                        map_color = self.map_color)
        

        layout.addWidget(self._canvas, 0, 0, 1, 5)
        # buttons to move up/down on the z-axis
        self._up_axis_btn = QtWidgets.QPushButton()
        self._up_axis_btn.setText("up")
        self._axis_step_label = QtWidgets.QLabel()
        self._axis_step_label.setText("Step: " + str(axis_step))
        self._axis_step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._down_axis_btn = QtWidgets.QPushButton()
        self._down_axis_btn.setText("down")

        def down_axis():
            ret = self._canvas.set_axis_step(-1)
            self._axis_step_label.setText("Step: " + str(ret))
        
        def up_axis():
            ret = self._canvas.set_axis_step(1)
            self._axis_step_label.setText("Step: " + str(ret))

        # saves animation as a gif
        def _do_render():
            filename = QFileDialog.getSaveFileName(self, caption="Rendering", dir=os.getcwd(), filter="Images (*.gif)")
            self._canvas.do_render(filepath=filename[0])
        self._render_video_button.clicked.connect(_do_render)
        
        self._channel_update = QtWidgets.QPushButton()
        self._channel_update.setText("update")
        
        self._up_axis_btn.clicked.connect(up_axis)
        self._down_axis_btn.clicked.connect(down_axis)
       
        self._playButton = QtWidgets.QPushButton()
        self._playButton.setText("Stop")
        self._canvas.start_stop_label = self._playButton
        self._leftButton = QtWidgets.QPushButton()
        self._leftButton.setText("<<")
        self._rightButton = QtWidgets.QPushButton()
        self._rightButton.setText(">>")
        self._resetButton = QtWidgets.QPushButton()
        self._resetButton.setText("reset")

        # on/off switch for the animation
        def fun():
            ret = self._canvas.switch_running()
            self._playButton.setText(ret)
        # moves the time axis up/down
        def left():
            self._canvas.set_pending_step(-1)
        def right():
            self._canvas.set_pending_step(1)
        
        # resets the figure, opening a popup-window asking for new channels
        # the inputs 1-5 and 1,2,3,4,5 are equivalent 
        def reset():
            # sees wether or not a new stepsDict is used
            try:
                dict = self._canvas.get_new_data()
            except:
                dict = stepsDict
            max_channel = np.shape(dict[0])[-1]
            # widget.closeEvent()  # mpl clean up
            updated_channels = []
            # popup dialog
            text, ok = QInputDialog.getText(self, 'Channel reset', 'Enter Channels\n \n\' - \' for range and \' , \' for seperation\n') 
            if ok:
                text_input = str(text)
                global number
                number = ""
                # loop for the input
                for t in range(len(text_input)):
                    # ignores spaces
                    if text_input[t] == ' ':
                        continue
                    # the inputs symbol for the range '-'
                    elif text_input[t] == '-':
                        if len(number) > 0:
                            number2 = ""
                            t += 1
                            # get the number behind '-'
                            while t < len(text_input) and text_input[t] != ',':
                                if text_input[t] == ' ':
                                    continue
                                number2 += text_input[t]
                                t += 1
                            # turn both numbers into int
                            c1 = int(number)
                            c2 = int(number2)
                            # add every number between c1 and c2 to the channel list
                            while c1 < c2:
                                if c1 < max_channel:
                                    updated_channels.append(c1)
                                    c1 += 1
                                else:
                                    break
                        number = ""
                    # append the input-position to the number string
                    elif text_input[t] != ',':
                        number += text_input[t]
                        # add the number to the channel list if it is the last character
                        if t + 1 == len(text_input):
                            c = int(number)
                            if c < max_channel:
                                updated_channels.append(c)
                            number == ""
                    # add number to the list if the character is a ','
                    else:
                        c = int(number)
                        if c < max_channel:
                            updated_channels.append(c)
                        number = ""
            # get the canvas widget
            item = layout.itemAt(9)
            try:
                #delete the canvas widget if present
                widget = item.widget()
                widget.close_event()
                widget.deleteLater()  # QT cleanup
                layout.removeWidget(widget)
            except:
                pass
            # dont update the channels if the list is empty
            if len(updated_channels) == 0:
                self._canvas = AnimatedHeatMap_multi(stepsDict=dict, net_config=net_config, step_axis=step_axis, 
                                                    parent=parent, start_step=start_step, axis_step=axis_step, channnel=self.channel,
                                                    map_color=self.map_color)
            else:
                # update channels
                if len(updated_channels) < 2:
                    updated_channels.append(updated_channels[0])
                self._canvas = AnimatedHeatMap_multi(stepsDict=dict, net_config=net_config, step_axis=step_axis, 
                                                    parent=parent, start_step=start_step, axis_step=axis_step, channnel=updated_channels,
                                                    map_color=self.map_color)
            # add the widget to the layout
            layout.addWidget(self._canvas, 0, 0, 1, 5)
            self._playButton.setText("Stop")
            self.channel = updated_channels
        self._leftButton.clicked.connect(left)
        self._rightButton.clicked.connect(right)
        self._resetButton.clicked.connect(reset)

        self._playButton.clicked.connect(fun)

        self._color_mapping_chooser = QComboBox(self)
        # list of all included heatmap colors
        color_map = ["gist_earth", "terrain", "gnuplot", "brg", "rainbow", "PiYG", "cool", "plasma", "inferno", "magma", "cividis"]
        self._color_mapping_chooser.addItems(color_map)

        def handle_color_change(index: int):
            # look for new stepsDict
            try:
                dict = self._canvas.get_new_data()
            except:
                dict = stepsDict
            self.map_color = color_map[index]

            item = layout.itemAt(9)
            try:
                #delete the canvas widget if present
                widget = item.widget()
                widget.close_event()
                widget.deleteLater()  # QT cleanup
                layout.removeWidget(widget)
            except:
                pass
            # update canvas widget
            self._canvas = AnimatedHeatMap_multi(stepsDict=dict, net_config=net_config, step_axis=step_axis, 
                                                    parent=parent, start_step=start_step, axis_step=axis_step, channnel=self.channel,
                                                    map_color=self.map_color)
            layout.addWidget(self._canvas, 0, 0, 1, 5)     
        self._color_mapping_chooser.setCurrentIndex(7)
        self._color_mapping_chooser.currentIndexChanged.connect(handle_color_change)
        

        layout.addWidget(self._up_axis_btn, 1, 3, 1, 2)
        layout.addWidget(self._axis_step_label, 1, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._down_axis_btn, 1, 0, 1, 2)
        layout.addWidget(self._leftButton, 2, 0, 1, 1)
        layout.addWidget(self._playButton, 2, 1, 1, 1)
        layout.addWidget(self._rightButton, 2, 2, 1, 1)
        layout.addWidget(self._render_video_button, 2, 3, 1, 1)
        layout.addWidget(self._color_mapping_chooser, 2, 4, 1, 1)
        layout.addWidget(self._resetButton, 2, 5, 1, 1)
        layout.addWidget(self._canvas, 0, 0, 1, 5)

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
        