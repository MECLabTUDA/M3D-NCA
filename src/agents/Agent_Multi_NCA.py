import torch
from src.agents.Agent_NCA import Agent_NCA
import os

class Agent_Multi_NCA(Agent_NCA):
    """Base functionality for multiple NCAs working in combination
    """
    def batch_step(self, data, loss_f):
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        outputs, targets = self.get_outputs(data)
        for m in range(self.exp.get_from_config('train_model')+1):
            self.optimizer[m].zero_grad()
        loss = 0
        loss_ret = {}
        for m in range(outputs.shape[-1]):
            if 1 in targets[..., m]:
                loss_loc = loss_f(outputs[..., m], targets[..., m])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            for m in range(self.exp.get_from_config('train_model')+1):
                self.optimizer[m].step() 
                self.scheduler[m].step()
        return loss_ret

    def save_state(self, model_path):
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)

        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            torch.save(m.state_dict(), os.path.join(model_path, 'model'+ str(id) +'.pth'))
            torch.save(o.state_dict(), os.path.join(model_path, 'optimizer'+ str(id) +'.pth'))
            torch.save(s.state_dict(), os.path.join(model_path, 'scheduler'+ str(id) +'.pth'))

    def load_state(self, model_path):
        r"""Load state of current model
        """
        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            m.load_state_dict(torch.load(os.path.join(model_path, 'model'+ str(id) +'.pth')))
            o.load_state_dict(torch.load(os.path.join(model_path, 'optimizer'+ str(id) +'.pth')))
            s.load_state_dict(torch.load(os.path.join(model_path, 'scheduler'+ str(id) +'.pth')))