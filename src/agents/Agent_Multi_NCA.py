import torch
from src.agents.Agent_NCA import Agent_NCA
import os

class EMA():
    def __init__(self, model, decay):
        """
        model  : the neural network model
        decay  : the decay rate for EMA
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.original_params = {}

        # Initialize shadow parameters with the model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update the shadow parameters with the current model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the shadow parameters to the model and store the original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store the original parameters
                self.original_params[name] = param.data.clone()
                # Apply the shadow parameters
                param.data = self.shadow[name]

    def restore_original(self):
        """Restore the original parameters to the model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Restore the original parameters
                param.data = self.original_params[name]

class Agent_Multi_NCA(Agent_NCA):
    """Base functionality for multiple NCAs working in combination
    """
    def batch_step(self, data: tuple, loss_f: torch.nn.Module) -> dict:
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

    def save_state(self, model_path: str) -> None:
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)

        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            torch.save(m.state_dict(), os.path.join(model_path, 'model'+ str(id) +'.pth'))
            torch.save(o.state_dict(), os.path.join(model_path, 'optimizer'+ str(id) +'.pth'))
            torch.save(s.state_dict(), os.path.join(model_path, 'scheduler'+ str(id) +'.pth'))

            if self.ema is not None:
                ema_parameters = self.ema.shadow 
                torch.save(ema_parameters, os.path.join(model_path, 'ema_state.pth'))
                

    def load_state(self, model_path: str) -> None:
        r"""Load state of current model
        """
        for id, z in enumerate(zip(self.model, self.optimizer, self.scheduler)):
            m, o, s = z
            m.load_state_dict(torch.load(os.path.join(model_path, 'model'+ str(id) +'.pth'), map_location=self.device))
            o.load_state_dict(torch.load(os.path.join(model_path, 'optimizer'+ str(id) +'.pth'), map_location=self.device))
            s.load_state_dict(torch.load(os.path.join(model_path, 'scheduler'+ str(id) +'.pth'), map_location=self.device))

            if os.path.exists(os.path.join(model_path, 'ema_state.pth')):
                self.ema = EMA(m, decay=0.99)
                self.ema.shadow = torch.load(os.path.join(model_path, 'ema_state.pth'), map_location=self.device)
                print("LOAD EMA")