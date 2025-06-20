from typing import Any, Dict, List
import os
import torch
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
#from torch.utils.tensorboard import SummaryWriter
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from typing import List, Dict, Any
import numpy as np
from PIL import Image as PILImage
from src.utils.Experiment import Experiment

"""
Used internally for some purposes
"""
class Experiment_vis(Experiment):

    replace_config: List[Dict[str, Any]] = None
    def __init__(self, config: List[Dict[str, Any]], dataset, model, agent):

        if not isinstance(config, List):
            config = [config]
        self.replace_config = config
        super().__init__(config, dataset, model, agent)


    



    def add_required_to_config(self):
        r"""Fills config with basic setup if not defined otherwise
        """
        if 'Persistence' not in self.projectConfig[0]:
            self.projectConfig[0]['Persistence'] = False
        if 'batch_duplication' not in self.projectConfig[0]:
            self.projectConfig[0]['batch_duplication'] = 1
        if 'keep_original_scale' not in self.projectConfig[0]:
            self.projectConfig[0]['keep_original_scale'] = False
        if 'rescale' not in self.projectConfig[0]:
            self.projectConfig[0]['rescale'] = True
        if 'channel_n' not in self.projectConfig[0]:
            self.projectConfig[0]['channel_n'] = 16
        if 'cell_fire_rate' not in self.projectConfig[0]:
            self.projectConfig[0]['cell_fire_rate'] = 0.5
        if 'output_channels' not in self.projectConfig[0]:
            self.projectConfig[0]['output_channels'] = 1

        # Basic Configs
        if 'model_path' not in self.projectConfig[0]:
            self.projectConfig[0]['model_path'] = os.path.join(pc.STUDY_PATH, 'Experiments', self.projectConfig[0]['name'])


   
    def setup(self):
        r"""Initial experiment setup when first started
        """
        # Create dirs
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))), exist_ok=True)

        self.data_split = self.new_datasplit()
        dump_pickle_file(self.data_split, os.path.join(self.config['model_path'], 'data_split.dt'))
        dump_json_file(self.projectConfig, os.path.join(self.config['model_path'], 'config.dt'))

    def reload(self):
        r"""Reload old experiment to continue training

        """

        print(os.path.join(self.config['model_path'], 'data_split.dt'))

        self.data_split = load_pickle_file(os.path.join(self.config['model_path'], 'data_split.dt'))
        self.projectConfig = load_json_file(os.path.join(self.config['model_path'], 'config.dt'))

        # Find most recent Experiment with name and reload
        if not isinstance(self.projectConfig, List):
            self.projectConfig = [self.projectConfig]
        self.config = self.projectConfig[0]
        if self.replace_config is not None:
            self.temporarly_overwrite_config(self.replace_config)
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep))
        print(model_path)
        if os.path.exists(model_path):
            print("Reload State " + str(self.currentStep))
            self.agent.load_state(model_path, self.get_from_config("device"))

    def write_scalar(self, tag, value, step):
        r"""Write scalars to tensorboard
        """
        #self.writer.add_scalar(tag, value, step)
        
        return