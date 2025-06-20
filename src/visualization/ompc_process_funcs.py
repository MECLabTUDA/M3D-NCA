from enum import Enum
from src.utils.IntensityMapping import MappingMode, BinningMode, IntensityMapping
from typing import Optional, Dict, Tuple, Any
from src.utils.MathHelper import MathHelper
import numpy as np
from src.utils.util_enumy import *
from src.visualization.Surface3Dplot import Surface3DplotGuts
from src.utils.MathHelper import MathHelper
import copy
"""
This file contains functions that are used by the Outputmapping chooser in a seperate process to handle 
Changes to Interpolation and rendering of Density Visualization.
Handle Surface Data 3d currently only displays an approximation 
of the interpolation function. For eample tangens mapping is displayed as linear. 
"""
"""
get_plot_data2 and get_plot_data are both used inside the plotting process to generate the Data Surface and the Interpolation surface. 

"""
 #===========================================================================================
def get_plot_data2(data: Dict, channel: int, bins=100):
    l = list()
            
    for k in data.keys():
        l.append(data[k][..., channel].flatten())
    X, Y, Z, low, high = MathHelper.get_pdf_surface_plot_binned(l, bins=bins)
        
    return (X, Y, Z)
def get_plot_data(data: Dict, channel: int, interpolation_bins: int, render_bins=60, interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, q_value: float = np.pi/4):
    l = list()
    for k in data.keys():
        l.append(data[k][..., channel].flatten())
    low, high = MathHelper.get_percentile_values(l, data_perentage=0.985)
    mapping_funcs, bin_mappings, bins = MathHelper.get_individual_piecewise_linear_matrix_transformation_extended(l, interpolation_bins, np.array([0, 1]), interpolation_mode=interpolation_mode, q_value=q_value)
    X, Y, Z = MathHelper.get_surface_plot_from_mappings(mapping_funcs, low, high)
    return (X, Y, Z)
#========================================================================================
"""
Imports and execution are handled seperately. 
This is required by the multiprocessing implementation in the dataflowhandler to prevent unneccessary reloading.
"""
def _handle_imports():
    from enum import Enum
    from src.utils.IntensityMapping import IntensityMapping
    from src.utils.util_enumy import MappingMode, BinningMode
    from typing import Optional, Dict, Tuple, Any
    from src.utils.MathHelper import MathHelper
    import numpy as np
    from src.utils.util_enumy import InterpolationAction, InterpolationMode
    from src.visualization.Surface3Dplot import Surface3DplotGuts
    from src.utils.MathHelper import MathHelper
    import copy
def _handle_mapping_change(intensity_mapping: IntensityMapping, operation: InterpolationAction = InterpolationAction.GENERATE, 
                          mapping_mode: MappingMode = MappingMode.SINGLE, bin_number: int = 1, binning_mode: BinningMode = BinningMode.EQUI_WIDTH, interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, 
                          q_value: float = np.pi/4, lower_crop: float = None, higher_crop: float = None, new_steps_dict: Dict[int, np.ndarray] = None, new_output_range: Tuple[float, float] = None, new_channel: int = None, 
                          new_mapping_mode: MappingMode = MappingMode.SINGLE, new_bin_number: int = 1, new_binning_mode: BinningMode = BinningMode.EQUI_WIDTH, new_interpolation_mode: InterpolationMode = InterpolationMode.LINEAR, 
                          new_q_value: float = np.pi/4, new_lower_crop: float = None, new_higher_crop: float = None):
    # initial recursive agrument needs to be set. Otherwise this will not work
    if intensity_mapping is None:
        print("Unknown error", flush=True)
        return intensity_mapping, (operation, None)
    if operation == InterpolationAction.SET_MAPPING_MODE:
        intensity_mapping.set_mapping_mode(mapping_mode)
        return intensity_mapping, (operation, copy.deepcopy(intensity_mapping.bins))
    elif operation == InterpolationAction.GENERATE:
        mapping = intensity_mapping.generate_mapping()
        return intensity_mapping, (operation, (mapping, copy.deepcopy(intensity_mapping.bins)))
    elif operation == InterpolationAction.SET_BIN_NR:
        intensity_mapping.set_bin_number(bin_number=bin_number)
        return intensity_mapping, (operation, copy.deepcopy(intensity_mapping.bins))
    elif operation == InterpolationAction.CHANNEL_CHANGED:
        intensity_mapping = IntensityMapping(points=new_steps_dict, output_range=new_output_range, channel=new_channel, 
                                             mapping_mode=new_mapping_mode, bin_number=new_bin_number, initial_binning_mode=new_binning_mode, initial_interpolation_mode=new_interpolation_mode, 
                                             initial_q_value=new_q_value, lower_crop=new_lower_crop, higher_crop=new_higher_crop)
        return intensity_mapping, (operation, copy.deepcopy(intensity_mapping.bins))
    elif operation == InterpolationAction.HANDLE_NEW_DATA:
        intensity_mapping = IntensityMapping(points=new_steps_dict, output_range=new_output_range, channel=new_channel, 
                                             mapping_mode=new_mapping_mode, bin_number=new_bin_number, initial_binning_mode=new_binning_mode, initial_interpolation_mode=new_interpolation_mode, 
                                             initial_q_value = new_q_value, lower_crop=new_lower_crop, higher_crop=new_higher_crop)
        return intensity_mapping, (operation, copy.deepcopy(intensity_mapping.bins))
    
    elif operation == InterpolationAction.SET_BINNING_MODE:
        bins = intensity_mapping.set_binning_mode(binning_mode=binning_mode)
        return intensity_mapping, (operation, bins)
    elif operation == InterpolationAction.SET_INTERPOLATION_MODE:
        intensity_mapping.set_interpolation_mode(interpolation_mode)
        return intensity_mapping, (operation, None)
    elif operation == InterpolationAction.SET_Q_VALUE:
        intensity_mapping.set_q_value(q_value=q_value)
        return intensity_mapping, (operation, None)
    elif operation == InterpolationAction.CROPPING_CHANGE:
        intensity_mapping.set_crop(lower_crop=lower_crop, higher_crop=higher_crop)
        return intensity_mapping, (operation, None)
def _handle_3dSurfaceUpdate(plot_guts: Surface3DplotGuts, surf3DplotOperation: Surface3DPlotOperation, 
                            params: Dict[int, Any]):
    if surf3DplotOperation == Surface3DPlotOperation.CONTENT_UPDATE:
        html = plot_guts.update_contents(params)
        surfaces = plot_guts.displayed_surfaces
        return plot_guts, (Surface3DPlotOperation.CONTENT_UPDATE, html, surfaces)
    elif surf3DplotOperation == Surface3DPlotOperation.DISPLAY_FIGURE:
        plot_guts.display_figure()
        return plot_guts, (Surface3DPlotOperation.DISPLAY_FIGURE, None)
    else:
        return plot_guts, (surf3DplotOperation, None)
    

    

    






