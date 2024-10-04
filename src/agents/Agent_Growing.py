import numpy as np
from src.agents.Agent_NCA import Agent_NCA
import torch 

class Agent_Growing(Agent_NCA):
    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data['id'], data['image'], data['label']
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., 0:4], targets
    
    @torch.no_grad()
    def test(self, *args, **kwargs):
        r"""Test the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        # Prepare dataset for testing
        dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            outputs, targets = self.get_outputs(data, full_img=True, tag="0")

        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate')).detach().cpu().numpy()

        self.exp.write_img("Sample0", outputs[..., -1], self.exp.currentStep)