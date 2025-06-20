import matplotlib
matplotlib.use('QtAgg')
from qtpy import QtWidgets
from qtpy.QtWidgets import QStackedWidget
from qtpy.QtCore import Qt
from enum import Enum
from typing import Dict, Any, Callable, List, Optional
import numpy as np
from src.utils.VisUtils import StepsDictOperations
from src.utils.MathHelper import MathHelper
from src.utils.DataFlowHandler import ResultGetters, DataFlowHandler, AgentProcArgType
import os
from src.visualization.SegmentedPseudo1DScatterPlot import SegmentedPseudo1DScatterPlot
from src.visualization.AnimatedHeatMap import AnimatedHeatMap
from src.visualization.Surface3Dplot import Surface3Dplot, Surface3DplotGuts
from src.utils.IntensityMapping import *
from src.utils.QThreadUtils import GuiComputationDispatcher
from src.visualization.ompc_process_funcs import InterpolationAction, Surface3DPlotOperation, get_plot_data2, get_plot_data
from src.utils.util_enumy import *
from src.visualization.CroppingWidget import CroppingWidget
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
Contains All options for the Interpolation as well as Both Scatterplot and Surface plot widget that vizualize the current interpolation.
Makes heavy use of Seperate Computation processes that handle Generating the actual intensity mapping as well as the 3D surface plot representation.
"""
class OutputMappingChooser(QtWidgets.QWidget):
    scatterplot: SegmentedPseudo1DScatterPlot = None
    surface_density_vis: Surface3Dplot
    widget_stack: QStackedWidget
    channel: int # channel that is currently visualized
    steps_dict: Dict[int, np.ndarray]
    segment_num: int
    segment_slider: QtWidgets.QSlider
    segment_nr_min: int = 1
    segment_nr_max: int = 20
    current_point_mappings: List[np.ndarray] = None
    interpolation_method_combo_box: QtWidgets.QComboBox
    interpolation_methods = ["linear", "tangens"]
    interpolation_style_combo_box: QtWidgets.QComboBox
    interpolation_style= ["single", "time stepped"]
    binning_mode_combo_box: QtWidgets.QComboBox 
    binning_modes = ["EQUI WIDTH", "MULTI OTSU"]
    diff_abs_choose_combo_borx: QtWidgets.QComboBox
    diff_abs_modes = ["ABSOLUTE", "DIFF"]
    generate_interpolation_button: QtWidgets.QPushButton
    apply_interpolation_button: QtWidgets.QPushButton
    transformation_function: List[Callable[[np.ndarray], np.ndarray]] | Callable[[np.ndarray], np.ndarray]
    corresponding_heatmap: AnimatedHeatMap
    decreaseSegmentButton: QtWidgets.QPushButton
    increaseSegmentButton: QtWidgets.QPushButton
    channel_chooser: QtWidgets.QComboBox
    q_value_slider: QtWidgets.QSlider
    q_value_label: QtWidgets.QLabel
    mapping_mode: MappingMode = MappingMode.SINGLE
    binning_mode: BinningMode = BinningMode.EQUI_WIDTH
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR
    q_value: float = np.pi/4
    mapping_process_name: str 
    surface_process_name: str
    group_name: str
    cropper: CroppingWidget = None
    flowhandler: DataFlowHandler

    def handle_mapping_vizualization_changes(self, data: Tuple[Any, Dict[str, Any]]):
        """
        This function handles returns from both computation processes. 
        3D surfce plot and Interpolation generation for the same change will always be returned at the same time. 
        Update of the vizualized intensity mapping is currently only done in this function for a stepped mapping. 
        For single mapping, the Scatterplot is updated directly in the button Handler. 


        """
        job_id, data_dict = data
        if self.mapping_process_name in data_dict:
            process_return = data_dict[self.mapping_process_name]
            return_val: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
            process_action: InterpolationAction
            process_action, return_val = process_return
            if process_action in [InterpolationAction.SET_MAPPING_MODE, InterpolationAction.SET_BIN_NR, InterpolationAction.CHANNEL_CHANGED, InterpolationAction.HANDLE_NEW_DATA, InterpolationAction.SET_BINNING_MODE]:
                if self.binning_mode == BinningMode.MULTI_OTSU and self.mapping_mode == MappingMode.SINGLE:
                    bins = return_val
                    if len(bins) > 2:
                        inner_bin_edges = np.zeros((len(bins) - 2, ))
                        for i in range(1, len(bins)-1):
                            inner_bin_edges[i-1] = bins[i][0]
                    
                    else:
                        inner_bin_edges = np.zeros((0, ))

                    self.scatterplot.update_vlines(inner_bin_edges)
                    self.scatterplot._redrar_vlines()
            elif process_action in [InterpolationAction.GENERATE]:
                self.transformation_function = return_val[0]
                bin_mapping = return_val[1]
                if self.mapping_mode == MappingMode.SINGLE:
                    bin_vals = np.zeros((len(bin_mapping), ))
                    for i in range(len(bin_mapping)):
                        bin_vals[i] = bin_mapping[i][0]
                    min_val = np.min(self.scatterplot.x_data)
                    max_val = np.max(self.scatterplot.x_data)
                    x_vals = np.linspace(min_val, max_val, 350)
                    y_vals = self.transformation_function(x_vals)
                    self.scatterplot.update_vlines(bin_vals)
                    self.scatterplot._redrar_vlines()
                    self.scatterplot.draw_interpolation_function(x_vals, y_vals)
                self.apply_interpolation_button.setEnabled(True)
            elif process_action in [InterpolationAction.SET_INTERPOLATION_MODE, InterpolationAction.SET_Q_VALUE]:
                a = 1
                    
        if self.surface_process_name in data_dict:
            process_return = data_dict[self.surface_process_name]
            paction, *_ = process_return
            if paction == Surface3DPlotOperation.CONTENT_UPDATE:
                self.display_interpolation_btn.setEnabled(True)
                

        
        self.enable_all_expect_apply()

    def __init__(self, stepsDict: Dict[int, np.ndarray], flowhandler: DataFlowHandler,  parent=None, start_channel: int = 0, default_num_segments: int = 1, min_channel: int = 0, max_channel: int = 15, 
                 corresponding_heatmap: AnimatedHeatMap = None, mapping_process_name: str = "intensity_mapper", surface_process_name: str = "surface_visualizer", 
                 group_name: str = "outputmapping_chooser"):
        """Initializes the widget. This widget is designed to allow the user to create various 
            kinds of output mappings for the data    

        Args:
            stepsDict (Dict[int, np.ndarray]): Contains Data for network run
            flowhandler: Flowhandler over which data updates & inter widget communication is handled
            parent (_type_, optional): _description_. Defaults to None.
            start_channel (int, optional): Sets which output Channel of the Data is visualized per default. Defaults to 0.
            min/max_channel: Sets minimum or maximum channel that is visualised
            corresponding_heatmap: Heatmap that the outputmapping is generated for
            mapping_process_name: ID of the process that handles the generation of the mapping
            surface_process_name: Name of the process that handles the html generation for the surface visualization.
            group_name: Name of the process group that handles multiprocessing for this OutputMapping Chooser
        """
        self.flowhandler = flowhandler
        super(OutputMappingChooser, self).__init__()
        self.mapping_process_name = mapping_process_name
        self.surface_process_name = surface_process_name
        self.group_name = group_name
        self.channel = start_channel
        self.min_channel = min_channel
        self.max_channel = max_channel
        self.steps_dict = stepsDict
        self.segment_num = default_num_segments
        mainLayout = QtWidgets.QGridLayout(self)
        flat_data, subsampled_flat_data = StepsDictOperations.channel_values(steps_dict=self.steps_dict, channel=self.channel, subsampling_interval=100)
        self.scatterplot = SegmentedPseudo1DScatterPlot(data=subsampled_flat_data, segments_num=self.segment_num)
        self.surface_density_vis = Surface3Dplot([get_plot_data2, get_plot_data], [None, None], gutless=True)  

        plotguts = Surface3DplotGuts([get_plot_data2, get_plot_data], [None, None])
        self.flowhandler.update_recursive_parameters(self.surface_process_name, plotguts, self.group_name)
        self.widget_stack = QStackedWidget(self)
        a = self.widget_stack.addWidget(self.scatterplot)
        b = self.widget_stack.addWidget(self.surface_density_vis)
        self.corresponding_heatmap = corresponding_heatmap
        int_mapping = IntensityMapping(points=stepsDict, output_range=(0.0, 1.0), channel=self.channel)
        self.flowhandler.update_recursive_parameters(self.mapping_process_name, int_mapping, self.group_name)
        self.flowhandler.add_handler_function(self.handle_mapping_vizualization_changes, self.group_name)
        def handle_changed_interpolation_method(index: int):
            """
            Only kills displayed interpolation function for single mapping. 
            Redrawing the surface plot is too expensive otherwise.
            """
            if index == 0:
                #linear
                self.interpolation_mode = InterpolationMode.LINEAR

            elif index == 1:
                #tangens
                self.interpolation_mode = InterpolationMode.TANGENS

            job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_INTERPOLATION_MODE, "interpolation_mode": self.interpolation_mode}}
            self.disable_all()
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
            if self.mapping_mode == MappingMode.SINGLE:        
                self.scatterplot.kill_interpolation_function()
        
        self.q_value_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.q_value_slider.setRange(0, 1000)
        self.q_value_slider.setSingleStep(1)
        self.q_value_slider.setValue(50)
        self.q_value_label = QtWidgets.QLabel(self)
        
        self.q_value_label.setText("q_value: " + str(round(self.q_value, 2)))
        def handle_set_q_value():
            """
            Only kills interpolation function in single_mapping . Otherwise computation too expensive.
            """
            q_value = float(self.q_value_slider.value())
            u = 0.001
            v = (np.pi/4)-0.000001
            def transform_range(val: float):
                return u + (val - 0)*(v-u)/1000
            
            if transform_range(q_value) <= 0 or transform_range(q_value) >= np.pi/2:
                raise Exception("Invalid Q value for tangens mapping")
            self.q_value = transform_range(q_value)
            job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_Q_VALUE, "q_value": self.q_value}}
            self.q_value_label.setText("q_value: " + str(round(self.q_value, 2)))
            self.disable_all()
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
            if self.mapping_mode == MappingMode.SINGLE:        
                self.scatterplot.kill_interpolation_function()
        self.q_value_slider.sliderReleased.connect(handle_set_q_value)
        # declare buttons and elements here
        self.interpolation_style_combo_box = QtWidgets.QComboBox(self)
        self.generate_interpolation_button = QtWidgets.QPushButton(self)
        self.apply_interpolation_button = QtWidgets.QPushButton(self)
        self.binning_mode_combo_box = QtWidgets.QComboBox(self)
        decrease_segment_button = QtWidgets.QPushButton(self)
        increase_segment_button = QtWidgets.QPushButton(self)
        self.display_interpolation_btn = QtWidgets.QPushButton(self)
        self.display_interpolation_btn.setText("Display interpolation")
        self.increaseSegmentButton = increase_segment_button
        self.decreaseSegmentButton = decrease_segment_button
        channel_choser = QtWidgets.QComboBox(self)
        self.channel_chooser = channel_choser
        channel_range = ["channel " + str(x) for x in list(range(self.min_channel, self.max_channel+1))]
        self.channel_chooser.addItems(channel_range)
        self.channel_chooser.setCurrentIndex(self.channel - self.min_channel)
        self.interpolation_method_combo_box = QtWidgets.QComboBox(self)
        self.interpolation_method_combo_box.addItems(self.interpolation_methods)
        self.interpolation_method_combo_box.setCurrentIndex(0)
        self.interpolation_method_combo_box.currentIndexChanged.connect(handle_changed_interpolation_method)

        self.diff_abs_choose_combo_borx = QtWidgets.QComboBox(self)

        def handle_binning_mode_change(index: int):
            """
            In single mapping mode, bins are redrawn in this method instead of the handler above.
            However, VLines are not redrawn for multi otsu, only deleted. Multi otus redraw is handled in the handler 
            above. This allows for the reuse of the generated binning.
            
            """
            self.apply_interpolation_button.setEnabled(False)
            self.display_interpolation_btn.setEnabled(False)
                
            if index == 0:
                """
                Equi width.
                """
                self.binning_mode = BinningMode.EQUI_WIDTH
                
                if self.mapping_mode == MappingMode.STEPPED:
                    # equi width
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BINNING_MODE, "binning_mode": BinningMode.EQUI_WIDTH}}
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                    # dont need to update 3d surface in this case.
                elif self.mapping_mode == MappingMode.SINGLE:
                    # equi width
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BINNING_MODE, "binning_mode": BinningMode.EQUI_WIDTH}}
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                    # redraw the Segment with equi width here
                    self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                    self.scatterplot._redrar_vlines()
                    self.scatterplot.kill_interpolation_function()

            elif index == 1:
                """
                Multi Otsu
                """
                self.binning_mode = BinningMode.MULTI_OTSU
                # multi otsu
                if self.mapping_mode == MappingMode.STEPPED:
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BINNING_MODE, "binning_mode": BinningMode.MULTI_OTSU}}
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)

                
                elif self.mapping_mode == MappingMode.SINGLE:
                    # Redraw the Vlines in handler in this case. here set only to zero.
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BINNING_MODE, "binning_mode": BinningMode.MULTI_OTSU}}
                    self.scatterplot.update_vlines(np.zeros((0, )))
                    self.scatterplot._redrar_vlines()
                    self.scatterplot.kill_interpolation_function()
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                    

                
        self.binning_mode_combo_box.addItems(self.binning_modes)
        self.binning_mode_combo_box.currentIndexChanged.connect(handle_binning_mode_change)
    

        def display_interpolation(arg=None):
            job_dispatch_args = {self.surface_process_name : {"surf3DplotOperation": Surface3DPlotOperation.DISPLAY_FIGURE, "params":None}}
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
        self.display_interpolation_btn.clicked.connect(display_interpolation)


        def handle_interpolation_style_change(index: int):
            """In case of multi step interpolation: Additional job is 
            dispatched to generate 3D Surface plot in seperate process."""
            self.apply_interpolation_button.setEnabled(False)
            self.display_interpolation_btn.setEnabled(False)
                
            if index == 1:
                self.mapping_mode = MappingMode.STEPPED
                # only displays the Distribution of the points, 
                # not the mapping
                params = {0: {"data": self.steps_dict, "channel": self.channel}, 1:None}

                self.scatterplot.update_vlines(np.zeros((0, )))
                self.scatterplot._redrar_vlines()
                self.scatterplot.kill_interpolation_function()
                self.scatterplot.redraw_scatter_points(np.array([-0.0001, 0.0001]))
                self.scatterplot.kill_pdf()
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_MAPPING_MODE, "mapping_mode": MappingMode.STEPPED}, 
                                     self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}}
                self.disable_all()

                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
            elif index == 0:

                self.mapping_mode = MappingMode.SINGLE
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_MAPPING_MODE, "mapping_mode": MappingMode.SINGLE}} 
                self.disable_all()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                if self.binning_mode == BinningMode.EQUI_WIDTH:
                    self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                    self.scatterplot._redrar_vlines()
                    self.scatterplot.kill_interpolation_function()
                    flat_data, subsampled_flat_data = StepsDictOperations.channel_values(steps_dict=self.steps_dict, channel=self.channel, subsampling_interval=100)
                    self.scatterplot.redraw_scatter_points(subsampled_flat_data)
                    x_pdf, y_pdf = MathHelper.get_pdf_for_data(subsampled_flat_data)
                    self.scatterplot.draw_pdf(x_pdf, y_pdf)
                    
                elif self.binning_mode == BinningMode.MULTI_OTSU:
                    # Update of the lines in this case handled in callback
                    self.scatterplot.update_vlines(np.zeros((0, )))
                    self.scatterplot._redrar_vlines()
                    self.scatterplot.kill_interpolation_function()
                    

        self.interpolation_style_combo_box.currentIndexChanged.connect(handle_interpolation_style_change)
        self.interpolation_style_combo_box.addItems(self.interpolation_style)

        def handle_generate_interpolation():
            """
            if stepped: Surfaces are redrawn in order to add surface of generated interpolation. 
            Currently the density surface is redrawn as well. This could be eliminated in the future to improve 
            performance
            """
            if self.mapping_mode == MappingMode.SINGLE:
                if self.segment_num <= 0:
                    return
                
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.GENERATE}} 
                self.disable_all()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                self.scatterplot.kill_interpolation_function()
                
                
            
            elif self.mapping_mode == MappingMode.STEPPED:
                
                params = {0:{"data": self.steps_dict, "channel": self.channel}, 1: {"data": self.steps_dict, "channel": self.channel, "interpolation_bins": self.segment_num, "interpolation_mode": self.interpolation_mode, "q_value": self.q_value}}                

                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.GENERATE}, 
                    self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}}
                
                self.disable_all()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)


        self.generate_interpolation_button.setText("Generate")
        self.generate_interpolation_button.clicked.connect(handle_generate_interpolation)
        
        def handle_apply_interpolation():
            if (not self.transformation_function is None) and (not self.corresponding_heatmap is None):
                self.corresponding_heatmap.set_transformation_function(self.transformation_function)
            
                
        self.apply_interpolation_button.setText("Apply")
        self.apply_interpolation_button.clicked.connect(handle_apply_interpolation)
        self.apply_interpolation_button.setEnabled(False)
        self.display_interpolation_btn.setEnabled(False)
        
        def handle_diff_abs_value_changed(index: int):
            """
            This responds to a button. Allows switching between actual data values and 
            difference between individual time steps. 
            Necesarry redraws of Interpolaton function, ... are not handled here, as switching the displayed 
            data using the flowhandler triggers a network data event. redraws are handled there.
            
            """
            if index == 0:
                "absolute"
                self.flowhandler.switch_current_stash()

            elif index == 1:
                "diff"
                if self.flowhandler.has_stashed():
                    self.flowhandler.switch_current_stash()
                else:
                    data = self.flowhandler.get_current_data()
                    new_data = {}
                    a = False
                    keys = list(data.keys())
                    for i in range(len(keys)):
                        if not a:
                            new_data[keys[i]] = np.copy(data[keys[i]])
                            a = True
                        else:
                            if data[keys[i]].shape != data[keys[i-1]].shape:
                                resized_old: np.ndarray = np.resize(data[keys[i-1]], data[keys[i]].shape)
                                new_data[keys[i]] = data[keys[i]] - resized_old    
                            else:
                                new_data[keys[i]] = data[keys[i]] - data[keys[i-1]]
                    self.flowhandler._add_stashed_data(new_data)
                    self.flowhandler.switch_current_stash()

        def handle_increase():
            
            if self.segment_num >= self.segment_nr_max:
                return
            self.segment_num += 1
            
            if self.mapping_mode == MappingMode.SINGLE:
                self.apply_interpolation_button.setEnabled(False)
                self.display_interpolation_btn.setEnabled(False)
                decrease_segment_button.setText(str(self.segment_num) + "-")
                increase_segment_button.setText(str(self.segment_num) + "+")
                self.scatterplot.kill_interpolation_function()
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}} 
                self.disable_all()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                if self.binning_mode == BinningMode.MULTI_OTSU:
                    self.scatterplot.update_vlines(np.zeros((0, )))
                    # VLInes redrawn in handler
                elif self.binning_mode == BinningMode.EQUI_WIDTH:
                    
                    self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                    self.scatterplot._redrar_vlines()
                    
                    
                    
            

            elif self.mapping_mode == MappingMode.STEPPED:
                self.apply_interpolation_button.setEnabled(False)
                self.display_interpolation_btn.setEnabled(False)
                decrease_segment_button.setText(str(self.segment_num) + "-")
                increase_segment_button.setText(str(self.segment_num) + "+")
                if self.surface_density_vis.displayed_surfaces == 1:
                    """No need to erase the old interpolated function"""

                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}} 
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
            


                else: 
                    params = {0:{"data": self.steps_dict, "channel": self.channel}, 1: None}
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}, 
                                         self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}} 
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                

            
        increase_segment_button.setText(str(self.segment_num) + "+")
        increase_segment_button.clicked.connect(handle_increase)
        
        
        def handle_decrease():
            """
            redraws the bin edges in case of single mapping. Always draws equi width bins, even for multi otsu binning.
            """
            if self.segment_num <= self.segment_nr_min:
                return
            self.segment_num -= 1
            if self.mapping_mode == MappingMode.SINGLE:
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}} 
                self.disable_all()
                decrease_segment_button.setText(str(self.segment_num) + "-")
                increase_segment_button.setText(str(self.segment_num) + "+")
                self.scatterplot.kill_interpolation_function()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)

                if self.binning_mode == BinningMode.EQUI_WIDTH:
                    self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                    self.scatterplot._redrar_vlines()

                elif self.binning_mode == BinningMode.MULTI_OTSU:
                    self.scatterplot.update_vlines(np.zeros((0, )))
                    self.scatterplot._redrar_vlines()
                    
                
            elif self.mapping_mode == MappingMode.STEPPED:
                self.apply_interpolation_button.setEnabled(False)
                self.display_interpolation_btn.setEnabled(False)
                decrease_segment_button.setText(str(self.segment_num) + "-")
                increase_segment_button.setText(str(self.segment_num) + "+")
                if self.surface_density_vis.displayed_surfaces == 1:
                    """No need to erase the old interpolated function"""
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}} 
                    self.disable_all()
                    
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                    
                else: 
                    params = {0:{"data": self.steps_dict, "channel": self.channel}, 1: None}
                    job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.SET_BIN_NR, "bin_number": self.segment_num}, 
                                         self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}} 
                    self.disable_all()
                    self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                
                
        decrease_segment_button.setText(str(self.segment_num) + "-")
        decrease_segment_button.clicked.connect(handle_decrease)
        
        
        def handle_channel_changed(val):
            """
            This action necessitates a reset of the interpolation function in the computation process. 
            All parameters need to be passed again. (This could possibly be improved upon). 
            use new_... parameters for this only, other parameters are disregarded.
            """
            self.channel = self.min_channel + val
            if not self.cropper is None:
                self.cropper._set_channel(self.channel)
            if self.mapping_mode == MappingMode.SINGLE:
                crop_low = None
                crop_high = None
                if not self.cropper is None:
                    crop_low, crop_high = self.cropper.get_croppings()
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.CHANNEL_CHANGED, "new_steps_dict": self.steps_dict, "new_output_range": (0.0, 1.0), "new_channel": self.channel, 
                                             "new_mapping_mode": self.mapping_mode, "new_bin_number": self.segment_num, "new_binning_mode": self.binning_mode, "new_interpolation_mode": self.interpolation_mode, 
                                             "new_q_value": self.q_value, "new_lower_crop": crop_low , "new_higher_crop": crop_high}} 
                self.disable_all()
                if not self.corresponding_heatmap is None:
                    self.corresponding_heatmap.set_channel(self.channel)
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)
                self.scatterplot.kill_interpolation_function()
                flat_data, subsampled_flat_data = StepsDictOperations.channel_values(steps_dict=self.steps_dict, channel=self.channel, subsampling_interval=100)
                self.scatterplot.redraw_scatter_points(subsampled_flat_data)
                x_pdf, y_pdf = MathHelper.get_pdf_for_data(subsampled_flat_data)
                self.scatterplot.draw_pdf(x_pdf, y_pdf)
                
                if self.binning_mode == BinningMode.EQUI_WIDTH:
                    
                    self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                    self.scatterplot._redrar_vlines()
                elif self.binning_mode == BinningMode.MULTI_OTSU:
                    self.scatterplot.update_vlines(np.zeros((0, )))
                    self.scatterplot._redrar_vlines()
                    

                
                
                
            elif self.mapping_mode == MappingMode.STEPPED:
                params = {0:{"data": self.steps_dict, "channel": self.channel}, 1: None}
                crop_low = None
                crop_high = None
                if not self.cropper is None:
                    crop_low, crop_high = self.cropper.get_croppings()
                if not self.corresponding_heatmap is None:
                    self.corresponding_heatmap.set_channel(self.channel)
                job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.CHANNEL_CHANGED, "new_steps_dict": self.steps_dict, "new_output_range": (0.0, 1.0), "new_channel": self.channel, 
                                             "new_mapping_mode": self.mapping_mode, "new_bin_number": self.segment_num, "new_binning_mode": self.binning_mode, "new_interpolation_mode": self.interpolation_mode, 
                                             "new_q_value": self.q_value, "new_lower_crop": crop_low , "new_higher_crop": crop_high}, 
                                             self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}} 
                self.disable_all()
                self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)

        self.channel_chooser.currentIndexChanged.connect(handle_channel_changed)
        self.diff_abs_choose_combo_borx.addItems(self.diff_abs_modes)
        self.diff_abs_choose_combo_borx.setCurrentIndex(0)
        self.diff_abs_choose_combo_borx.currentIndexChanged.connect(handle_diff_abs_value_changed)
        
        # layout
        mainLayout.addWidget(self.widget_stack, 0, 0, 1, 4)
        mainLayout.addWidget(self.interpolation_method_combo_box, 1, 0, 1, 1)
        mainLayout.addWidget(self.interpolation_style_combo_box, 1, 1, 1, 1)
        mainLayout.addWidget(self.increaseSegmentButton, 1, 2, 1, 1)
        mainLayout.addWidget(self.decreaseSegmentButton, 1, 3, 1, 1)
        mainLayout.addWidget(self.apply_interpolation_button, 2, 2, 1, 1)
        mainLayout.addWidget(self.generate_interpolation_button, 2, 1, 1, 1)
        mainLayout.addWidget(self.channel_chooser, 2, 0, 1, 1)
        mainLayout.addWidget(self.binning_mode_combo_box, 2, 3, 1, 1)
        mainLayout.addWidget(self.display_interpolation_btn, 3, 0, 1, 1)
        mainLayout.addWidget(self.q_value_label, 3, 1, 1, 1)
        mainLayout.addWidget(self.q_value_slider, 3, 2, 1, 1)
        mainLayout.addWidget(self.diff_abs_choose_combo_borx, 3, 3, 1, 1)

        self.setLayout(mainLayout)


    def set_segment_slider_val(self, val: int):
        
        self.segment_slider.value = val
        self.segment_slider.sliderPosition = val
        self.segment_slider.update()
        self.segment_slider.repaint()
    def setup_remote_output_getter(self, dfh: DataFlowHandler):
        """
        Sets up the handler that responds to new data. This handler also responds to the switch from absolute to diff 
        data triggered by this widget.
        """
        def handle_new_result(data: Tuple[Any, Dict[str, Tuple[AgentProcArgType]|Tuple[AgentProcArgType, Dict[int, np.ndarray], str]]]):        
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            if my_data[0] == FlowDataChangeType.ABSDIFF:
                data = my_data[1]
                self.set_new_data(data, do_reset_diff_abs=False)
                self.corresponding_heatmap.set_new_data(data)
        
            elif my_data[0] == FlowDataChangeType.NETWORK:
                my_data = my_data[1]
                if len(my_data)==1:
                    return
                _, data, fname = my_data
                self.set_new_data(data)
                self.corresponding_heatmap.set_new_data(data)
        dfh.add_network_handler(handle_new_result)
    def set_new_data(self, data: Dict[int, np.ndarray], do_reset_diff_abs: bool = True):
        """
        Can also be used externally if necessary. 
         This action necessitates a reset of the interpolation function in the computation process. 
            All parameters need to be passed again. (This could possibly be improved upon). 
            use new_... parameters for this only, other parameters are disregarded.
        """
        self.steps_dict = data
        if do_reset_diff_abs:
            self.diff_abs_choose_combo_borx.blockSignals(True)
            self.diff_abs_choose_combo_borx.setCurrentIndex(0)
            self.diff_abs_choose_combo_borx.blockSignals(False)
        if self.mapping_mode == MappingMode.SINGLE:
            self.scatterplot.kill_interpolation_function()
            flat_data, subsampled_flat_data = StepsDictOperations.channel_values(steps_dict=self.steps_dict, channel=self.channel, subsampling_interval=100)
            self.scatterplot.redraw_scatter_points(subsampled_flat_data)
            x_pdf, y_pdf = MathHelper.get_pdf_for_data(subsampled_flat_data)
            self.scatterplot.draw_pdf(x_pdf, y_pdf)
            crop_low = None
            crop_high = None
            if not self.cropper is None:
                crop_low, crop_high = self.cropper.get_croppings()
            job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.HANDLE_NEW_DATA, "new_steps_dict": self.steps_dict, "new_output_range": (0.0, 1.0), "new_channel": self.channel, 
                                             "new_mapping_mode": self.mapping_mode, "new_bin_number": self.segment_num, "new_binning_mode": self.binning_mode, "new_interpolation_mode": self.interpolation_mode, 
                                             "new_q_value": self.q_value, "new_lower_crop": crop_low , "new_higher_crop": crop_high}} 
            self.disable_all()
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name) 

            if self.binning_mode == BinningMode.EQUI_WIDTH:
                
                self.scatterplot.update_vlines(np.zeros((0, )))
                self.scatterplot._redrar_vlines()
                

            elif self.binning_mode == BinningMode.MULTI_OTSU:
                self.scatterplot.update_vlines(self.scatterplot._get_default_intervals(self.segment_num))
                self.scatterplot._redrar_vlines()
                
            
        elif self.mapping_mode == MappingMode.STEPPED:
            params = {0:{"data": self.steps_dict, "channel": self.channel}, 1: None}
            crop_low = None
            crop_high = None
            if not self.cropper is None:
                crop_low, crop_high = self.cropper.get_croppings()
            job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.HANDLE_NEW_DATA, "new_steps_dict": self.steps_dict, "new_output_range": (0.0, 1.0), "new_channel": self.channel, 
                                             "new_mapping_mode": self.mapping_mode, "new_bin_number": self.segment_num, "new_binning_mode": self.binning_mode, "new_interpolation_mode": self.interpolation_mode, 
                                             "new_q_value": self.q_value, "new_lower_crop": crop_low , "new_higher_crop": crop_high}, 
                                             self.surface_process_name: {"surf3DplotOperation": Surface3DPlotOperation.CONTENT_UPDATE, "params": params}} 
            
            self.disable_all()
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)

    def enable_all_expect_apply(self):
        self.generate_interpolation_button.setEnabled(True)
        self.interpolation_method_combo_box.setEnabled(True)
        self.interpolation_style_combo_box.setEnabled(True)
        self.decreaseSegmentButton.setEnabled(True)
        self.increaseSegmentButton.setEnabled(True)
        self.channel_chooser.setEnabled(True)
        self.q_value_slider.setEnabled(True)
        self.binning_mode_combo_box.setEnabled(True)
        self.diff_abs_choose_combo_borx.setEnabled(True)
        if not self.cropper is None:
            self.cropper.enable()

    def disable_all(self):
        self.apply_interpolation_button.setEnabled(False)
        self.q_value_slider.setEnabled(False)
        self.generate_interpolation_button.setEnabled(False)
        self.interpolation_method_combo_box.setEnabled(False)
        self.interpolation_style_combo_box.setEnabled(False)
        self.decreaseSegmentButton.setEnabled(False)
        self.increaseSegmentButton.setEnabled(False)
        self.channel_chooser.setEnabled(False)
        self.binning_mode_combo_box.setEnabled(False)
        self.diff_abs_choose_combo_borx.setEnabled(False)
        self.display_interpolation_btn.setEnabled(False)
        if not self.cropper is None:
            self.cropper.disable()




    def _add_cropping_widget(self, cropper: CroppingWidget):
        """
        Commuication between this widget and the cropping widget is handled using a dedicated group in the dataflowhandler. 
        Use this pattern during further refactoring. 
        cropping widget needs to be passed down to the corresponding heatmap, because the heatmap has the option 
        to display just the cropped data. 
        TODO Refactor: Handle this through the existing dedicated group.
        """
        self.cropper = cropper
        def handle_crop_value_change(data: Any):
            _, data = data
            keys = list(data.keys())
            my_data = data[keys[0]]
            lower_crop: float = None
            higher_crop: float = None
            lower_crop, higher_crop = my_data
            job_dispatch_args = {self.mapping_process_name : {"operation": InterpolationAction.CROPPING_CHANGE, "lower_crop": lower_crop,  "higher_crop": higher_crop}} 
            self.disable_all()
            self.corresponding_heatmap.set_cropping(lower_crop=lower_crop, higher_crop=higher_crop)
            self.flowhandler.dispatch_job(job_dispatch_args, group=self.group_name)

        self.flowhandler.add_handler_function(handle_crop_value_change, group=self.cropper.cropping_change_group)



    