import torch
import torch.nn as nn
from src.mednca.mednca.BasicNCA3D import BasicNCA3D
from src.mednca.mednca.M3DNCA import M3DNCA
import random
import math
from typing import Dict, Callable
import numpy as np
from src.mednca.mednca.BasicNCA3D import BasicNCA3D
import copy

 
class BasicNCA3D_vis(BasicNCA3D):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
        """
        super(BasicNCA3D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm3d(hidden_size, track_running_stats=False)
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)
        self.steps_dict: Dict[int, np.ndarray] = None
        self.instrumentation_function: Callable[[np.ndarray, int], bool] = None
        
        
    def set_state_dict(self, dict: Dict[int, np.ndarray]):
            self.steps_dict = dict

    def export_state_dict(self) -> Dict[int, np.ndarray]:
       
        ret: Dict[int, np.ndarray] = self.steps_dict
        self.steps_dict = None
        return ret

    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        self.instrumentation_function = function

    
    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        if not self.steps_dict is None:
            tmp: np.ndarray = x.clone().detach().cpu().numpy().squeeze()
            
            self.steps_dict[0] = tmp
        if not self.instrumentation_function is None:
            self.instrumentation_function(step, x.clone().detach().cpu().numpy().squeeze())

        for step in range(steps):

            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
            if not self.steps_dict is None:
                
                tmp: np.ndarray = x.clone().detach().cpu().numpy().squeeze()
                for i in range(tmp.shape[-2]):
                    if i > 0:
                        trmp = tmp[:, :, i, :] - tmp[:, :, i-1, :]
                        t1 = trmp.min()
                        t2 = trmp.max()
                
                self.steps_dict[step+1] = tmp
            if not self.instrumentation_function is None:
                self.instrumentation_function(step+1, x.clone().detach().cpu().numpy().squeeze())

        return x




class M3DNCA_exp_vis(M3DNCA):
    r"""Implementation of M3D-NCA
    """

    def __init__(self, *args, device="cpu", channel_n: int = 16, fire_rate: float = 0.5, 
                 steps: int = 20, hidden_size: int = 64, input_channels: int = 1, output_channels: int = 1, 
                 scale_factor: int = 4, levels: int = 2, kernel_size: int = 7, batch_duplication: int = 1, **kwargs):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(M3DNCA, self).__init__()
        self.eval()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.scale_factor = scale_factor
        self.levels = levels
        self.batch_duplication = batch_duplication
        self.steps_dict: Dict[int, np.ndarray]
        self.instrumentation_function: Callable[[np.ndarray, int], bool] = None
        self.model = nn.ModuleList()
        for i in range(self.levels):
            kernel_size = kernel_size if i == 0 else 3
            self.model.append(BasicNCA3D_vis(channel_n=self.channel_n, fire_rate=self.fire_rate, device=self.device, 
                            hidden_size=hidden_size, input_channels=input_channels))

        
    def set_state_dict(self, dict: Dict[int, np.ndarray]):
            self.steps_dict = dict

    def export_state_dict(self) -> Dict[int, np.ndarray]:
        ret: Dict[int, np.ndarray] = self.steps_dict
        ks = list(ret.keys())
        zmax: int = 0
        for key in ks:
            if ret[key].shape[-2] > zmax:
                zmax = ret[key].shape[-2]
        for key in ks:
            datapoint: np.ndarray = ret[key]
            if datapoint.shape[-2] < zmax:
                old_z = datapoint.shape[-2]
                
                new_shape = (datapoint.shape[0], datapoint.shape[1], zmax, datapoint.shape[3])
                new_point: np.ndarray = np.zeros(new_shape)
                slices_per_slice:float = float(zmax)/float(old_z)
                actual_z_index: int = 0
                estimated_z_index: float = 0.0

                for i in range(old_z):
                    if actual_z_index >= zmax:
                        break
                    new_point[:, :, actual_z_index, :] = datapoint[:, :, i, :]
                    actual_z_index += 1
                    estimated_z_index += slices_per_slice
                    while actual_z_index + 1 <= estimated_z_index and actual_z_index < zmax:
                        new_point[:, :, actual_z_index, :] = datapoint[:, :, i, :]
                        actual_z_index += 1
                while actual_z_index < zmax:
                    new_point[:, :, actual_z_index, :] = datapoint[:, :, -1, :]
                ret[key] = new_point
        
        self.steps_dict = None
        
        return ret

    def set_instrumentation_function(self, function: Callable[[np.ndarray, int], bool]):
        self.instrumentation_function = function


        
    def forward_eval(self, x: torch.Tensor):
        inputs_loc, _ = self.downscale_image(x, x, iterations=self.levels-1)

        inputs_loc = self.make_seed(inputs_loc)

        full_res = x

        with torch.no_grad():
            # Start with low res lvl and go to high res level
            for m in range(self.levels):
                self.model[m].set_state_dict(copy.deepcopy(self.steps_dict))
                self.model[m].set_instrumentation_function(self.instrumentation_function)
            for m in range(self.levels):
                if m == self.levels-1:
                    outputs = inputs_loc
                    
                    outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)
                    sub_model_state_dict: Dict[int, np.ndarray] = self.model[m].export_state_dict()
                    if not m == 0:
                        sub_model_state_dict.pop(0)
                    if m == 0:
                        offset_index: int = 0
                    else:
                        offset_index = list(self.steps_dict.keys())[-1]
                    for k in sub_model_state_dict.keys():
                        self.steps_dict[k+offset_index] = sub_model_state_dict[k]
                    inputs_loc = outputs

                # Scale m-1 times 
                else:
                    up = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')

                    outputs = inputs_loc
                    outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)
                    sub_model_state_dict: Dict[int, np.ndarray] = self.model[m].export_state_dict()
                    if not m == 0:
                        sub_model_state_dict.pop(0)
                    if m == 0:
                        offset_index: int = 0
                    else:
                        offset_index = list(self.steps_dict.keys())[-1]
                    for k in sub_model_state_dict.keys():
                        self.steps_dict[k+offset_index] = sub_model_state_dict[k]
                    inputs_loc = outputs

                    # Upscale lowres features to next level
                    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                    outputs = up(outputs)
                    inputs_loc = x     
                    outputs = torch.permute(outputs, (0, 2, 3, 4, 1))         

                    next_res, _ = self.downscale_image(full_res, full_res, iterations=self.levels-(m+2))

                    # Concat lowres features with higher res image
                    inputs_loc = torch.concat((next_res[...,:self.input_channels], outputs[...,self.input_channels:]), 4)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels]
