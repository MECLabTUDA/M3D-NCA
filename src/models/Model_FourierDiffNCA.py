import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
 
import torch.nn.utils.weight_norm as weight_norm

class SpatiallyIndependentNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(SpatiallyIndependentNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        # Compute mean and variance across the channel dimension at each spatial location
        mean = x.mean(dim=1, keepdim=True)  
        variance = x.var(dim=1, keepdim=True, unbiased=False)  

        # Normalize each channel independently at each spatial location
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)

        return normalized_x

class FourierDiffNCA(nn.Module):
    r"""Implementation of Diffusion NCA
    """

    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, img_size=28, normalisation=0):
        r"""Init function
        """
        super(FourierDiffNCA, self).__init__() 

        # Hardcoded model parameters
        extra_channels = 4
        self.device=device
        self.input_channels = input_channels
        self.channel_n = channel_n
        kernelSize = 3
        padding = int((kernelSize-1)/2)

        # ---------------- MODEL 0: NCA FOURIER -----------------

        # Normalisation method
        if normalisation == 0: # Then Spatially Independent
            self.norm_fourier = SpatiallyIndependentNorm()
        else:
            self.norm_fourier = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)

        # Model layers
        self.p0_fourier = weight_norm(nn.Conv2d(channel_n*2+extra_channels, channel_n*2, kernel_size=kernelSize, stride=1, padding=padding))
        self.fc0_fourier = weight_norm(nn.Conv2d(channel_n*2*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0)) 
        self.fc1_fourier = weight_norm(nn.Conv2d(hidden_size, channel_n*2, kernel_size=1, stride=1, padding=0)) 

        self.embedding_map_1_fourier = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels, hidden_size//4, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(hidden_size//4, channel_n*2*2+extra_channels, kernel_size=1, stride=1, padding=0)), # Guides flow based on conditioning
        )

        self.embedding_map_2_fourier = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels, hidden_size//4, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(hidden_size//4, hidden_size, kernel_size=1, stride=1, padding=0)), # Guides flow based on conditioning
        )

        # combine pos and timestep
        self.embedding_net_fourier = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0))
        )

        self.model_0 = {"normalisation":self.norm_fourier, "conv0": self.p0_fourier, "fc0": self.fc0_fourier, "fc1": self.fc1_fourier, "embedding_net": self.embedding_net_fourier, "embedding_map_1": self.embedding_map_1_fourier, "embedding_map_2": self.embedding_map_2_fourier}#, "pt1": self.conv_pt_1, "pt2": self.conv_pt_2}#, "fc05": self.fc05_middle_real, "fc06": self.fc05_middle_real, "fc07": self.fc05_middle_real}

        kernelSize = 3
        padding = int((kernelSize-1)/2)

        self.bn = nn.BatchNorm2d(hidden_size)

        # ---------------- MODEL 1: NCA IMAGE RGB -----------------
        # Normalisation method
        if normalisation == 0: # Then Spatially Independent
            self.norm_rgb = SpatiallyIndependentNorm()
        else:
            self.norm_rgb = nn.GroupNorm(num_groups =  1, num_channels=hidden_size)
        
        # Model layers
        self.p0_rgb = weight_norm(nn.Conv2d(channel_n+extra_channels, channel_n, kernel_size=kernelSize, stride=1, padding=padding, padding_mode="reflect"))
        self.fc0_rgb = weight_norm(nn.Conv2d(channel_n*2+extra_channels, hidden_size, kernel_size=1, stride=1, padding=0)) 
        self.fc1_rgb = weight_norm(nn.Conv2d(hidden_size, channel_n, kernel_size=1, stride=1, padding=0)) 

        self.embedding_map_1_rgb = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels, hidden_size//4, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(hidden_size//4, channel_n*2+extra_channels, kernel_size=1, stride=1, padding=0)), 
        )

        self.embedding_map_2_rgb = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels, hidden_size//4, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(hidden_size//4, hidden_size, kernel_size=1, stride=1, padding=0)), 
        )

        self.embedding_network_rgb = nn.Sequential(
            weight_norm(nn.Conv2d(extra_channels*4, 256, kernel_size=1, stride=1, padding=0)),
            nn.SiLU(),
            weight_norm(nn.Conv2d(256, extra_channels, kernel_size=1, stride=1, padding=0))
        )
        self.model_1 = {"normalisation":self.norm_rgb, "conv0": self.p0_rgb, "fc0": self.fc0_rgb, "fc1": self.fc1_rgb, "embedding_net": self.embedding_network_rgb, "embedding_map_1": self.embedding_map_1_rgb, "embedding_map_2": self.embedding_map_2_rgb}
    
    def perceive_dict(self, x, conv0):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = conv0(x)
        y = torch.cat((x,y1),1)
        return y

    def channel_embedding_4d(self, inputs: torch.Tensor, max_period: int = 10000):
        """ Sinusoidal channel embeddings for 4D data.

        :param inputs: Input tensor of shape (batch_size, channels, height, width).
        :param max_period: Maximum period for the sinusoidal function.
        :return: Input tensor with channel encodings.
        """

        channels = inputs.shape[1]

        freq = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=channels//2, dtype=torch.float32) / (channels//2)
        ).to(inputs.device)

        channel_steps = torch.arange(channels, device=inputs.device).float()[None, :, None, None]


        for i in range(channels):
            if i == 0:
                out = inputs[:,i:i+1,...] * torch.cat([torch.cos(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None]), 
                                        torch.sin(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None])], dim=1)
            else:
                out_loc = inputs[:,i:i+1,...] * torch.cat([torch.cos(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None]), 
                                        torch.sin(channel_steps[:,i:i+1,:,:] * freq[None, :, None, None])], dim=1)
                out = torch.cat([out, out_loc], dim=1)

        return out 

    def update_dict(self, x, fire_rate, diff_step, model_dict, nca_step = 0):
        r"""
        stochastic update stage of NCA
        :param x_in: perception vector
        :param fire_rate:
        :param angle: rotation
        :return: residual updated vector
        """

        dx = x

        # Create positional encodings
        pos_x = torch.linspace(1, 0, dx.shape[3]).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)
        pos_y = torch.linspace(1, 0, dx.shape[2]).expand(dx.shape[0], 1, dx.shape[3], dx.shape[2]).to(self.device).transpose(2,3)
        # Diffusion time step encoding 
        diff_step = diff_step.expand_as(pos_x.transpose(0, 3)).transpose(0, 3)
        # NCA step encoding
        nca_step = torch.tensor(nca_step).expand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(self.device)

        # Sinusoidal encoding
        pos_t_enc = self.channel_embedding_4d(torch.concat((pos_x, pos_y, diff_step, nca_step), 1))
        pos_t_enc = model_dict["embedding_net"](pos_t_enc)
        dx = torch.concat((dx, pos_t_enc), 1)

        # Perceive neighbourhood
        dx = self.perceive_dict(dx, model_dict["conv0"])

        # Map embedding to dx
        dx_conditioning = model_dict["embedding_map_1"](pos_t_enc)
        dx = dx * dx_conditioning

        dx = model_dict["fc0"](dx)

        # Normalize
        dx = model_dict["normalisation"](dx)
        dx = dx.transpose(1, 3)
        dx = F.leaky_relu(dx)

        dx = dx.transpose(1, 3)

        # Map embedding to dx
        dx_conditioning = model_dict["embedding_map_2"](pos_t_enc)
        dx = dx * dx_conditioning

        dx = model_dict["fc1"](dx)
        dx = dx.transpose(1, 3)

        if fire_rate is None:
            fire_rate = self.fire_rate

        # Stochastic Cell Update
        stochastic = (torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]).to(self.device)) > fire_rate
        stochastic = stochastic.float()
        
        dx = dx * stochastic

        dx = dx.transpose(1, 3)

        x = x + dx 

        return x
    
    def forward(self, x, steps=10, fire_rate=None, t=0, **kwargs):
        r"""
        forward pass from NCA
        :param x: perception
        :param steps: number of steps, such that far pixel can communicate
        :param fire_rate:
        :param angle: rotation
        :return: updated input
        """


        # ---------------- MODEL 0 -----------------
        x = x.transpose(1, 3) 

        # Define fourier size
        pixel_X = 16
        pixel_Y = 16

        # Convert to fourier space
        x = torch.fft.fft2(x, norm="forward")
        x = torch.fft.fftshift(x, dim=(2,3))
        x_old = x.clone()
        # Select patch
        x_start, y_start = x.shape[2]//2, x.shape[3]//2 
        x = x[..., x_start:x_start+pixel_X, y_start:y_start+pixel_Y]
        steps_f = pixel_X
        x = torch.concat((x.real, x.imag), 1)
        # Run NCA 0 in fourier space
        for step in range(steps_f*2):
            x_new = self.update_dict(x, 0, diff_step=t, model_dict=self.model_0, nca_step=step/(steps_f)) 
            x[:, self.input_channels:self.channel_n, ...] = x_new[:, self.input_channels:self.channel_n, ...]
            x[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...] = x_new[:, self.input_channels+self.channel_n:self.channel_n+self.channel_n, ...]

        # Convert back to image space
        x = x.transpose(1, 3)
        x = torch.complex(torch.split(x, int(x.shape[3]/2), dim=3)[0], torch.split(x, int(x.shape[3]/2), dim=3)[1])
        x = x.transpose(1, 3)
        x_old[:, self.input_channels:, x_start:x_start+pixel_X, y_start:y_start+pixel_Y] = x[:, self.input_channels:, ...]
        x_old = torch.fft.ifftshift(x_old, dim=(2,3))
        x = torch.fft.ifft2(x_old, norm="forward").real 

        del x_old

    
        # ---------------- MODEL 1 -----------------
        # Run second NCA in imaeg space for final diffusion
        for step in range(steps):
            x[:, self.input_channels:, ...] = self.update_dict(x, fire_rate, diff_step=t, model_dict=self.model_1, nca_step=step/steps)[:, self.input_channels:, ...] #* mask[:, self.input_channels:, ...]
        x = x.transpose(1, 3)
        
        return x