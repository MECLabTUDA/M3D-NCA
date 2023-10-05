import torch
import numpy as np
from src.utils.helper import dump_compressed_pickle_file, load_compressed_pickle_file
from src.agents.Agent import BaseAgent
import os

class Agent_NCA(BaseAgent):
    """Base agent for training NCA models
    """
    def initialize(self):
        super().initialize()
        self.input_channels = self.exp.get_from_config('input_channels')
        self.output_channels = self.exp.get_from_config('output_channels')
        self.pool = Pool()

    def loss_noOcillation(self, x, target, freeChange=True):
        #x = torch.flatten(x)
        if freeChange:
            x[x <= 1] = 0
            loss = x.sum() / torch.numel(x)
        else:
            xin_sum = torch.sum(x) + 1
            x = torch.square(target-x)
            loss = torch.sum(x) / xin_sum
        return loss

    def save_state(self, model_path):
        r"""Save state - Add Pool to state
        """
        super().save_state(model_path)
        if self.pool.__len__() != 0 and self.exp.get_from_config('save_pool'):
            dump_compressed_pickle_file(self.pool, os.path.join(model_path, 'pool.pbz2'))

    def load_state(self, model_path):
        r"""Load state - Add Pool to state
        """
        super().load_state(model_path)
        if os.path.exists(os.path.join(model_path, 'pool.pbz2')):
            self.pool = load_compressed_pickle_file(os.path.join(model_path, 'pool.pbz2'))

    def pad_target_f(self, target, padding):
        r"""Creates a padding around the tensor 
            #Args
                target (tensor)
                padding (int): padding on all 4 sides
        """
        target = np.pad(target, [(padding, padding), (padding, padding), (0, 0)])
        target = np.expand_dims(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32)).to(self.device)
        return target

    def make_seed(self, img):
        r"""Create a seed for the NCA - TODO: Currently only 0 input
            #Args
                shape ([int, int]): height, width shape
                n_channels (int): Number of channels
        """
        # 2D
        if( self.exp.dataset.slice != None):
            if len(img.shape) == 3:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
                seed[..., :img.shape[3]] = img
            else:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
                seed[..., 0:img.shape[-1]] = img 

        # 3D
        else:
            if len(img.shape) == 4:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
                seed[..., 0] = img  
            else:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], img.shape[3], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)#torch.from_numpy(np.zeros([img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')], np.float32)).to(self.device)
                seed[..., 0:img.shape[-1]] = img 

        return seed

    def repeatBatch(self, seed, target, batch_duplication):
        r"""Repeat batch -> Useful for better generalisation when doing random activated neurons
            #Args
                seed (tensor): Seed for NCA
                target (tensor): Target of Model
                batch_duplication (int): How many times it should be repeated
        """
        return torch.Tensor.repeat_interleave(seed, batch_duplication, dim=0), torch.Tensor.repeat_interleave(target, batch_duplication, dim=0)

    def getInferenceSteps(self):
        r"""Get the number of steps for inference, if its set to an array its a random value inbetween
        """
        #if len(self.exp.get_from_config('inference_steps')) > 1:
        #    steps = np.random.randint(self.exp.get_from_config('inference_steps')[0], self.exp.get_from_config('inference_steps')[1])
        #else:
        if type(self.exp.get_from_config('inference_steps')) is list:
            steps = self.exp.get_from_config('inference_steps')
        else:
            steps = self.exp.get_from_config('inference_steps')
        return steps

    def prepare_data(self, data, eval=False):
        r"""Prepare the data to be used with the model
            #Args
                data (int, tensor, tensor): identity, image, target mask
            #Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets = data
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.make_seed(inputs)
        if not eval:
            if self.exp.get_from_config('Persistence'):
                inputs = self.pool.getFromPool(inputs, id, self.device)
            inputs, targets = self.repeatBatch(inputs, targets, self.exp.get_from_config('batch_duplication'))
        return id, inputs, targets

    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets

    def initialize_epoch(self):
        r"""Create a pool for the current epoch
        """
        if self.exp.get_from_config('Persistence'):
            self.epoch_pool = Pool()

    def conclude_epoch(self):
        r"""Set epoch pool as active pool
        """
        if self.exp.get_from_config('Persistence'):
            self.pool = self.epoch_pool
            print("Pool_size: " + str(len(self.pool)))
        return

    def prepare_image_for_display(self, image):
        return image[...,0:3]

class Pool():
    r"""Keeps the previous outputs of the model stored in a pool
    """
    def __init__(self):
        self.pool = {}

    def __len__(self):
        return len(self.pool)

    def addToPool(self, outputs, ids):
        r"""Add a value to the pool
            #Args
                output (tensor): Output to store
                idx (int): idx in dataset
                exp (Experiment): All experiment related configs
                dataset (Dataset): The dataset
        """
        for i, key in enumerate(ids):
            self.pool[key] = outputs[i]

    def getFromPool(self, inputs, ids, device):   
        r"""Get value from pool
            #Args
                item (int): idx of item
                dataset (Dataset)
        """
        for i, key in enumerate(ids):
            if key in self.pool.keys():
                inputs[i] = self.pool[key].to(device)
        return inputs

    