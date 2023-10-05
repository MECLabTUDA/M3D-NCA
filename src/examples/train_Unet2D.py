from unet import UNet2D
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceBCELoss
from src.agents.Agent_UNet import Agent

config = [{
    'img_path': r"image_path",
    'label_path': r"label_path",
    'model_path': r'model_path',
    'device':"cuda:0",
    # Learning rate
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    # Training config
    'save_interval': 100,
    'evaluate_interval': 10,
    'n_epoch': 1000,
    'batch_size': 500,
    # Model config
    'channel_n': 16,        # Number of CA state channels
    'cell_fire_rate': 0.5,
    'output_channels': 3,
    # Data
    'input_size': (64, 64),
    'data_split': [0.7, 0, 0.3], 

}]

# Define Experiment
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca = UNet2D(in_channels=1, padding=1, out_classes=1).to(device)
agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))
loss_function = DiceBCELoss() 

# Number of parameters
print(sum(p.numel() for p in ca.parameters() if p.requires_grad))

# Train Model
agent.train(data_loader, loss_function)

# Average Dice Score on Test set
agent.getAverageDiceScore()