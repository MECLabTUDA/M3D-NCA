import torch
import numpy as np
from src.agents.Agent_MedNCA_Simple import MedNCAAgent
import random
import torchio as tio
from matplotlib import pyplot as plt
import torch.optim as optim
from itertools import chain
from src.losses.LossFunctions import DiceFocalLoss_2
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torchvision.models import vgg19
    
class Agent_Med_NCA_finetuning(MedNCAAgent):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """

    def __init__(self, model: torch.nn.Module, gamma = 100):
        super().__init__(model)
        self.gamma = gamma

    def initialize(self):
        # create test  model
        super().initialize()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

        params = chain(self.model.backbone_lowres.p0.parameters(), \
                       self.model.backbone_lowres.p1.parameters(), \
                       self.model.backbone_highres.p0.parameters(), \
                       self.model.backbone_highres.p1.parameters())

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        
        if 'variance' in data:
            variance = data['variance']
        if 'pred' in data:
            pred = data['pred']

        if self.model.training:
            inputs, targets, variance, pred, inputs_loc = self.model(inputs, targets, variance, pred, return_channels=False)
            return inputs, targets, variance, pred, inputs_loc
        else:
            inputs, targets = self.model(inputs, targets, return_channels=False)
            return inputs, targets

    def batch_step(self, data: tuple, loss_f: torch.nn.Module, gradient_norm: bool = False) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """

        data = self.prepare_data(data)


        outputs, targets, variance, pred, inputs_loc = self.get_outputs(data, return_channels=False)

        difference = inputs_loc[1][0, :, :, 0:1] - inputs_loc[0][0, :, :, 0:1]
        difference = (difference - difference.min()) / (difference.max() - difference.min())
        cat_img =  torch.cat((inputs_loc[1][0, :, :, 0:1], difference, inputs_loc[0][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
        self.exp.write_img('preprocessing_main_level',
                                cat_img,
                                self.exp.currentStep)
        difference = inputs_loc[3][0, :, :, 0:1] - inputs_loc[2][0, :, :, 0:1]
        difference = (difference - difference.min()) / (difference.max() - difference.min())
        cat_img =  torch.cat((inputs_loc[3][0, :, :, 0:1], difference, inputs_loc[2][0, :, :, 0:1]), dim=1).detach().cpu().numpy()
        self.exp.write_img('preprocessing_patch_level',
                                cat_img,
                                self.exp.currentStep)

        loss = 0

        dice_f = DiceFocalLoss_2()

        # NQM loss
        nqm_loss = 0
        nqm_loss2 = 0
        l1 = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()

        loss2 = l1(torch.sigmoid(inputs_loc[4][..., self.input_channels:self.input_channels+self.output_channels]), torch.sigmoid(inputs_loc[5][..., self.input_channels:self.input_channels+self.output_channels])) + \
            l1(torch.sigmoid(inputs_loc[6][..., self.input_channels:self.input_channels+self.output_channels]), torch.sigmoid(inputs_loc[7][..., self.input_channels:self.input_channels+self.output_channels]))/16


        loss_ret = {}# 

        loss = loss_f(outputs, pred, variance) + loss2*self.gamma 
        loss_ret[0] = loss.item()

        if loss != 0:
            loss.backward()

            #learning_rates = [param_group['lr'] for param_group in self.optimizer.param_groups] 

            self.optimizer.step()
            self.scheduler.step()

        return loss_ret
    
    def labelVariance(self, images: torch.Tensor, median: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = torch.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = torch.pow(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = torch.sqrt(stdd)

        print("NQM Score: ", torch.sum(stdd) / (images.shape[1] * images.shape[2]))

        return torch.sum(stdd) / (images.shape[1] * images.shape[2])#(torch.sum(median)+1e-9)
