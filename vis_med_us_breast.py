
import sys
from qtpy import QtWidgets, QtCore # No longer need QtGui specifically for this simple case

import sys
import os
import torch
from dotenv import load_dotenv

from qtpy.QtWidgets import QApplication

from src.visualization.mainwindow import MainWindow
from src.visualization.ConfigEditor import *
from src.utils.DataFlowHandler import DataFlowHandler

from src.visualization.ConfigEditor import config_options_from_dict
from src.visualization.docking_window_test import MainWindow as TestMainWindow
from src.visualization.ompc_process_funcs import _handle_imports, _handle_3dSurfaceUpdate, _handle_mapping_change
from src.utils.Nii_Gz_Dataset_experimental_vis import Nii_Gz_Dataset_vis
from src.utils.MedNCA_experimental_vis import MedNCA_exp_vis
from src.utils.MedNCAAgent_simple_experimental_vis import MedNCAAgent_simple_vis
import torch.multiprocessing as mp
import multiprocess




default_config = {
    'img_path': r"data/Dataset401_BUID/imagesTs",
    'label_path': r"data/Dataset401_BUID/labelsTs",
    'name': r'med_nca_kalkhof', #12 or 13, 54 opt, 
    'device':"cpu",
    'model_path':r'models/med_nca_buid',
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 10,
    'evaluate_interval': 201,
    'n_epoch': 200,
    'batch_duplication': 1,
    # Model
    'channel_n': 32,        # Number of CA state channels
    'inference_steps': 64,
    'cell_fire_rate': 0.5,
    'batch_size': 8,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 128,
    'train_model':1,
    # Data
    'input_size': [(64, 64), (256, 256)] ,
    'scale_factor': 4,
    'data_split': [1.0, 0, 0], 
    'keep_original_scale': False,
    'rescale': True,
    "label_select": "seg"
}



config_options = {
    "device" : ConfigOptionType.Text,
}
config_defaults = {
    "device" : default_config["device"]
}



def main():
    # create model, dataset and dataflow handler
    dataset_args = { 
        "image_path": r"data/Dataset401_BUID/imagesTs",
        "labels_path": r"data/Dataset401_BUID/labelsTs",
        "config": default_config,
        # "slices": None,
        "slice_axis": None
    }
    
    vis_model_args = {
        'steps': default_config['inference_steps'],
        'device': default_config['device'],
        'input_channels': default_config['input_channels'],
        'output_channels': default_config['output_channels'],
        'batch_duplication': default_config['batch_duplication'],
        'channel_n': default_config['channel_n'],
        'fire_rate': default_config['cell_fire_rate'],
        'hidden_size': default_config['hidden_size']
        
    }
    agent_args = {"config": [default_config]}   
    flowhandler = DataFlowHandler(dataset_args, agent_args, Nii_Gz_Dataset_vis, MedNCAAgent_simple_vis, MedNCA_exp_vis, vis_model_kwargs=vis_model_args, 
                                device_type=default_config['device'])

    with flowhandler.with_network():
        with flowhandler.with_process(import_function=_handle_imports, action_function=_handle_mapping_change, 
                                      process_id="omp_mapping", use_recursive_argument=True, initial_recursive_argumet=None, group="omp_chooser"):
            with flowhandler.with_process(import_function=_handle_imports, action_function= _handle_3dSurfaceUpdate, 
                                          process_id="surface_process", use_recursive_argument=True, initial_recursive_argumet=None, 
                                          group="omp_chooser"):
                # now create GUI components and connect to dataflow handler
                app = QApplication(sys.argv)
                default_opts = config_options_from_dict(config_options, config_defaults.values())
                w = MainWindow("NCA VIS", flowhandler, default_opts, default_config, icon_path="src/visualization/assets", ompc_group="omp_chooser", ompc_mapping_process="omp_mapping", 
                               ompc_surface_process="surface_process")
                #w = StandardWindow()
                #w = SimpleDemoWindow()
    
                ## w = DatapointLoaderWindow("yo", flowhandler)
                #w = TestMainWindow(flowhandler)
                w.show()
                app.exec_()

    flowhandler.cleanup()

if  __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    multiprocess.set_start_method("spawn", force=True)
    main()