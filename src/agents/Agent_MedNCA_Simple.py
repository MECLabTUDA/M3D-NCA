import torch
from src.agents.Agent_UNet import UNetAgent
from src.agents.Agent_MedSeg2D import Agent_MedSeg2D

class MedNCAAgent(UNetAgent):
    """Base agent for training UNet models
    """
    def initialize(self):
        super().initialize()

    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        
        inputs, targets = self.model(inputs, targets)
        return inputs, targets
