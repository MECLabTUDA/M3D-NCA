import torch
import torch.nn as nn
from src.models.Model_BasicNCA import BasicNCA
 
class BackboneNCA(BasicNCA):
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
        super(BackboneNCA, self).__init__(channel_n, fire_rate, device, hidden_size)
        self.p0 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        self.p1 = nn.Conv2d(channel_n, channel_n, kernel_size=3, stride=1, padding=1, padding_mode="reflect")

    def perceive(self, x):
        r"""Perceptive function, combines 2 conv outputs with the identity of the cell
            #Args:
                x: image
        """
        y1 = self.p0(x)
        y2 = self.p1(x)
        y = torch.cat((x,y1,y2),1)
        return y
