import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torchvision.transforms import Resize
from src.mednca.mednca.BasicNCA import BasicNCA
import numpy as np
from typing import Dict, Callable
import copy
from src.mednca.mednca.MedNCA import MedNCA
 
class BackboneNCA_vis(BasicNCA):
    r"""Implementation of the backbone NCA of Med-NCA
    """
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(BackboneNCA_vis, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.steps_dict: Dict[int, np.ndarray] = None
        
    def perceive(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = torch.cat((x,y1,y2),1)
        return y

        
    def set_state_dict(self, dict: Dict[int, np.ndarray]):
            self.steps_dict = dict

    def export_state_dict(self) -> Dict[int, np.ndarray]:
        
        ret: Dict[int, np.ndarray] = self.steps_dict
        self.steps_dict = None
        return ret

    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        self.instrumentation_function = function
        


    def forward(self, x, steps=64, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        tmp: np.ndarray = x.clone().detach().cpu().numpy().squeeze()
        tmp = tmp.transpose((1, 0, 2))
        self.steps_dict[0] = tmp
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,:self.input_channels], x2[...,self.input_channels:]), 3)
            tmp: np.ndarray = x.clone().detach().cpu().numpy().squeeze()
            tmp = tmp.transpose((1, 0, 2))
            self.steps_dict[step + 1] = tmp
        return x



class MedNCA_exp_vis(MedNCA):
    r"""Implementation of the backbone NCA of Med-NCA
    """
    def __init__(self, *args, device, channel_n: int = 32, fire_rate: float = 0.5, steps: int = 64, 
    hidden_size: int = 128, input_channels: int = 1, output_channels: int = 1, batch_duplication: int = 1, **kwargs):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(MedNCA, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.batch_duplication = batch_duplication
        self.steps_dict: Dict[int, np.ndarray]
        self.instrumentation_function: Callable[[np.ndarray, int], bool] = None
        
        self.backbone_lowres = BackboneNCA_vis(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)
        self.backbone_highres = BackboneNCA_vis(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels)

    def set_state_dict(self, dict: Dict[int, np.ndarray]):
            self.steps_dict = dict
            self.backbone_lowres.set_state_dict(copy.deepcopy(dict))
            self.backbone_highres.set_state_dict(copy.deepcopy(dict))
    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        self.instrumentation_function = function

    
    def export_state_dict(self) -> Dict[int, np.ndarray]:
        
        ret: Dict[int, np.ndarray] = self.steps_dict
        self.steps_dict = None
        return ret


    def forward_eval(self, x: torch.Tensor):
        down_scaled_size = (x.shape[1] // 4, x.shape[2] // 4)
        inputs_loc = self.resize4d(x.cpu(), size=down_scaled_size).to(self.device) 

        # Start with low res lvl and go to high res level
        for m in range(2):
            if m == 1:
                outputs = self.backbone_highres(inputs_loc, 
                                               steps=self.steps, 
                                               fire_rate=self.fire_rate)
                offset_index: int = list(self.steps_dict.keys())[-1]
                highres_steps_dict: Dict[int, np.ndarray] = self.backbone_highres.export_state_dict()
                for k in list(highres_steps_dict.keys()):
                    self.steps_dict[offset_index + 1 + k] = highres_steps_dict[k]
            else:
                outputs = self.backbone_lowres(inputs_loc, 
                                                steps=self.steps, 
                                                fire_rate=self.fire_rate)
                lowres_steps_dict: Dict[int, np.ndarray] = self.backbone_lowres.export_state_dict()
                for k in list(lowres_steps_dict.keys()):
                    self.steps_dict[k] = lowres_steps_dict[k]
                # Upscale lowres features to high res
                up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                outputs = torch.permute(outputs, (0, 3, 1, 2))
                outputs = up(outputs)   
                inputs_loc = x  
                outputs = torch.permute(outputs, (0, 2, 3, 1))       
                # Concat lowres features with high res image     
                inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
        return outputs[..., self.input_channels:self.input_channels+self.output_channels]

    