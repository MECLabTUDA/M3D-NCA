import torch
from src.agents.Agent_UNet import UNetAgent

class M3DNCAAgent_Simple(UNetAgent):
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
        
        inputs = inputs.permute(0, 2, 3, 4, 1)

        inputs, targets = self.model(inputs, targets)
        return inputs, targets 

