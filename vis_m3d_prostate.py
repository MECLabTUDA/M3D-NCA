
import sys
from qtpy import QtWidgets, QtCore # No longer need QtGui specifically for this simple case

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
from src.utils.Nii_Gz_Dataset_3d_experimental_vis import Dataset_NiiGz_3D_vis
from src.utils.m3d_nca_experimental_vismodel import M3DNCA_exp_vis
from src.utils.M3DNCAAGENT_SImple_experimental_vis import M3DNCAAgent_Simple_vis
import torch.multiprocessing as mp
import multiprocess




default_config = {
    'img_path': r"data/Prostate_MEDSeg/imagesTs/",
    'label_path': r"data/Prostate_MEDSeg/labelsTs/",
    'name': r'm3d_nca_kalkhof', #12 or 13, 54 opt,
    'model_path': 'models/m3d_nca_prostate/',
    'device':"cpu",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
    'betas': (0.9, 0.99),
    # Training
    'save_interval': 50,
    'evaluate_interval': 2001,
    'n_epoch': 2000,
    'batch_duplication': 2,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [20, 40], # [20, 40]
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(80, 80, 6), (320, 320, 24)] , #(160, 160, 12),
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3],
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
        "image_path": r"data/Prostate_MEDSeg/imagesTs/",
        "labels_path": r"data/Prostate_MEDSeg/labelsTs/",
        "config": default_config,
        # "slices": None,
        "slice_axis": None 
    }
    
    vis_model_args = {
        'device': default_config['device'],
        'batch_duplication': default_config['batch_duplication'],
        'steps': default_config['inference_steps']
        
    }
    agent_args = {"config": [default_config]}   
    flowhandler = DataFlowHandler(dataset_args, agent_args, Dataset_NiiGz_3D_vis, M3DNCAAgent_Simple_vis, M3DNCA_exp_vis, vis_model_kwargs=vis_model_args, 
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
                # w = DatapointLoaderWindow("yo", flowhandler)
                # w = TestMainWindow(flowhandler)
                w.show()
                app.exec_()

    flowhandler.cleanup()

if  __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    multiprocess.set_start_method("spawn", force=True)
    main()