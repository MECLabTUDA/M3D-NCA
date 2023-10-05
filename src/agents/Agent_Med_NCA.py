import torch
import numpy as np
from src.agents.Agent_Multi_NCA import Agent_Multi_NCA
import random
import torchio as tio

class Agent_Med_NCA(Agent_Multi_NCA):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """
    def initialize(self):
        super().initialize()
        self.stacked_models = self.exp.get_from_config('stacked_models')
        self.scaling_factor = self.exp.get_from_config('scaling_factor')   

    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data

        # Create down-scaled image
        down_scaled_size = (int(inputs.shape[1] / 4), int(inputs.shape[2] / 4))
        inputs_loc = self.resize4d(inputs.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device')) 
        targets_loc = self.resize4d(targets.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device'))

        # After training run inference on full image
        if full_img == True:
            with torch.no_grad():
                # Start with low res lvl and go to high res level
                for m in range(self.exp.get_from_config('train_model')+1):
                    if m == self.exp.get_from_config('train_model'):
                        outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                    else:
                        outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        # Upscale lowres features to high res
                        up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                        outputs = torch.permute(outputs, (0, 3, 1, 2))
                        outputs = up(outputs)
                        inputs_loc = inputs     
                        outputs = torch.permute(outputs, (0, 2, 3, 1))       
                        # Concat lowres features with high res image     
                        inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                        targets_loc = targets
        # During training run inference on patches
        else:
            # Start with low res lvl and go to high res level
            for m in range(self.exp.get_from_config('train_model')+1):
                if m == self.exp.get_from_config('train_model'):
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                else:
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))

                    # Upscale lowres features to high res
                    up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                    outputs = torch.permute(outputs, (0, 3, 1, 2))
                    outputs = up(outputs)
                    inputs_loc = inputs     
                    outputs = torch.permute(outputs, (0, 2, 3, 1))
                    # Concat lowres features with high res image             
                    inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                    targets_loc = targets

                    # Prepare array to store patch of 
                    size = self.exp.get_from_config('input_size')[0]
                    inputs_loc_temp = inputs_loc
                    targets_loc_temp = targets_loc
                    inputs_loc = torch.zeros((inputs_loc.shape[0], size[0], size[1], inputs_loc.shape[3])).to(self.exp.get_from_config('device'))
                    targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], targets_loc_temp.shape[3])).to(self.exp.get_from_config('device'))

                    # Choose random patch of upscaled image
                    for b in range(inputs_loc.shape[0]): 
                        pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                        pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])

                        inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]
                        targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]

        # Add pooling - not functional
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc 


    def resize4d(self, img, size=(64,64), factor=4, label=False):
        r"""Resize input image
            #Args
                img: 4d image to rescale
                size: image size
                factor: scaling factor
                label: is Label?
        """
        if label:
            transform = tio.Resize((size[0], size[1], -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], -1))
        img = transform(img)
        return img