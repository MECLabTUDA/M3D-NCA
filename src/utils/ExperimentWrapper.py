import torch
from src.utils.Experiment import Experiment

class ExperimentWrapper():
    def createExperiment(self, config : dict, model, agent, dataset, loss_function):
        model.to(config['device'])
        exp = Experiment(config, dataset, model, agent)
        data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
        exp.set_loss_function(loss_function)
        exp.set_data_loader(data_loader)
        return exp