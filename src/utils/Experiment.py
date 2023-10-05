import os
import torch
from src.utils.helper import dump_json_file, load_json_file, dump_pickle_file, load_pickle_file
from torch.utils.tensorboard import SummaryWriter

class Experiment():
    r"""This class handles:
            - Interactions with the experiment folder
            - Loading / Saving experiments
            - Datasets
    """
    def __init__(self, config, dataset, model, agent):
        self.projectConfig = config
        self.add_required_to_config()
        self.config = self.projectConfig[0]
        self.dataset = dataset
        self.model = model
        self.agent = agent
        self.general()
        if(os.path.isdir(os.path.join(self.config['model_path'], 'models'))):
            self.reload()
        else:
            self.setup()
        self.currentStep = self.currentStep+1
        self.set_current_config()

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

    def setup(self):
        r"""Initial experiment setup when first started
        """
        # Create dirs
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(os.path.join(self.config['model_path'], 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))), exist_ok=True)
        # Create basic configuration
        self.data_split = self.new_datasplit()
        dump_pickle_file(self.data_split, os.path.join(self.config['model_path'], 'data_split.dt'))
        dump_json_file(self.projectConfig, os.path.join(self.config['model_path'], 'config.dt'))

    def new_datasplit(self):
        return DataSplit(self.config['img_path'], self.config['label_path'], data_split = self.config['data_split'], dataset = self.dataset)

    def temporarly_overwrite_config(self, config):
        r"""This function is useful for evaluation purposes where you want to change the config, e.g. data paths or similar.
            It does not save the config and should NEVER be used during training.
        """
        print("WARNING: NEVER USE \'temporarly_overwrite_config\' FUNCTION DURING TRAINING.")
        self.projectConfig = config
        self.set_current_config()
        self.data_split = self.new_datasplit()
        self.set_size()

    def get_max_steps(self):
        r"""Get max defined training steps of experiment
        """
        return self.projectConfig[-1]['n_epoch']

    def reload(self):
        r"""Reload old experiment to continue training
            TODO: Add functionality to load any previous saved step
        """
        # TODO: Proper reload
        print(os.path.join(self.config['model_path'], 'data_split.dt'))
        self.data_split = load_pickle_file(os.path.join(self.config['model_path'], 'data_split.dt'))
        self.projectConfig = load_json_file(os.path.join(self.config['model_path'], 'config.dt'))
        self.config = self.projectConfig[0]
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep))
        print(model_path)
        if os.path.exists(model_path):
            print("Reload State " + str(self.currentStep))
            self.agent.load_state(model_path)
    
    def set_size(self):
        if isinstance(self.config['input_size'][0], tuple):
            self.dataset.set_size(self.config['input_size'][-1])
        else:
            print(self.config['input_size'])
            self.dataset.set_size(self.config['input_size'])

    def general(self):
        r"""General experiment configurations needed after setup or loading
        """
        self.currentStep = self.current_step()
        self.set_size()
        self.writer = SummaryWriter(log_dir=os.path.join(self.get_from_config('model_path'), 'tensorboard', os.path.basename(self.get_from_config('model_path'))))
        self.set_current_config()
        self.agent.set_exp(self)
        if self.currentStep == 0:
            self.write_text('config', str(self.projectConfig), 0)

        if self.get_from_config('unlock_CPU') is None or self.get_from_config('unlock_CPU') is False:
            print('In basic configuration threads are limited to 1 to limit CPU usage on shared Server. Add \'unlock_CPU:True\' to config to disable that.')
            torch.set_num_threads(1)


    def reload_model(self):
        r"""Reload model
            TODO: Move to a more logical position. Probably to the model and then call directly from the agent
        """
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep), 'model.pth')
        if os.path.exists(model_path):
            self.agent.load_model(model_path)

    def save_model(self):
        r"""TODO: Same as for reload -> move to better location
        """
        model_path = os.path.join(self.config['model_path'], 'models', 'epoch_' + str(self.currentStep+1))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))

    def current_step(self):
        r"""Find out the initial epoch by checking the saved models"""
        model_path = os.path.join(self.config['model_path'], 'models')
        if os.path.exists(model_path):
            dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(os.path.join(self.config['model_path'], 'models'), d))]
            if dirs:
                maxDir = max([int(d.split('_')[1]) for d in dirs])
                return maxDir
        return 0

    def set_model_state(self, state: str) -> None:
        r"""TODO: remove? """
        self.model_state = state
        self.dataset.setPaths(self.config['img_path'], self.data_split.get_images(state), self.config['label_path'], self.data_split.get_labels(state))
        self.dataset.setState(state)

        models = [self.model] if not isinstance(self.model, list) else self.model
        for m in models:
            if self.model_state == "train":
                m.train()
            else:
                m.eval()

        
    def get_from_config(self, tag):
        r"""Get from config
            #Args
                tag (String): Key of requested value
        """
        if tag in self.config.keys():
            return self.config[tag]
        else:
            return None

    def set_current_config(self):
        r"""Set current config. This can change during training and will always 
            overwrite previous settings, but keep everything else
        """
        self.config = {}
        for i in range(0, len(self.projectConfig)):
            for k in self.projectConfig[i].keys():
                self.config[k] = self.projectConfig[i][k]
            if self.projectConfig[i]['n_epoch'] > self.currentStep:
                return

    def increase_epoch(self):
        r"""Increase current epoch
        """
        self.currentStep = self.currentStep +1
        self.set_current_config()

    def get_current_config(self):
        r"""TODO: remove?"""
        return self.config

    def write_scalar(self, tag, value, step):
        r"""Write scalars to tensorboard
        """
        self.writer.add_scalar(tag, value, step)

    def write_img(self, tag, image, step):
        r"""Write an image to tensorboard
        """
        self.writer.add_image(tag, image, step, dataformats='HWC')

    def write_text(self, tag, text, step):
        r"""Write text to tensorboard
        """
        self.writer.add_text(tag, text, step)

    def write_histogram(self, tag, data, step):
        r"""Write data as histogram to tensorboard
        """
        self.writer.add_histogram(tag, data, step)

    def write_figure(self, tag, figure, step):
        r"""Write a figure to tensorboard images
        """
        self.writer.add_figure(tag, figure, step)


class DataSplit():
    r"""Handles the splitting of data
    """
    def __init__(self, path_image, path_label, data_split, dataset):
        self.images = self.split_files(self.getFilesInFolder(path_image, dataset), data_split)
        self.labels = self.split_files(self.getFilesInFolder(path_label, dataset), data_split)

    def get_images(self, state):
        r"""#Returns the images of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.images[state])

    def get_labels(self, state):
        r"""#Returns the labels of selected state
            #Args
                state (String): Can be 'train', 'val', 'test'
        """
        return self.get_data(self.labels[state])

    def get_data(self, data):
        r"""#Returns the data in a list rather than the stored folder strucure
            #Args
                data ({}): Dictionary ordered by {id, {slice, img_name}}
        """
        lst = data.values()
        lst_out = []
        for d in lst:
            lst_out.extend([*d.values()])
        return lst_out

    def split_files(self, files, data_split):
        r"""Split files into train, val, test according to definition
            while keeping patients slics together.
            #Args
                files ({int, {int, string}}): {id, {slice, img_name}}
                data_split ([float, float, float]): Sum of 1
        """
        dic = {'train':{}, 'val':{}, 'test':{}}
        for index, key in enumerate(files):
            if index / len(files) < data_split[0]:
                dic['train'][key] = files[key]
            elif index / len(files) < data_split[0] + data_split[1]: 
                dic['val'][key] = files[key]
            else:
                dic['test'][key] = files[key]
        print("Datasplit-> train entries: {}, val entries: {}, test entries: {}".format(len(dic['train']), len(dic['val']), len(dic['test'])))
        return dic

    def getFilesInFolder(self, path, dataset):
        r"""Get files in folder
            #Args
                path (String): Path to folder
                dataset (Dataset)
        """
        return  dataset.getFilesInPath(path) 
    