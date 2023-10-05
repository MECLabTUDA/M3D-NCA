import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.Model_BackboneNCA import BackboneNCA
from src.losses.LossFunctions import DiceBCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_Med_NCA import Agent_Med_NCA

config = [{
    'img_path': r"image_path",
    'label_path': r"label_path",
    'model_path': r'model_path',
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.5, 0.5),
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 1000,
    'batch_size': 48,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': 64,
    'cell_fire_rate': 0.5,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 128,
    'train_model':1,
    # Data
    'input_size': [(16, 16), (64, 64)],
    'data_split': [0.7, 0, 0.3], 
}
]

# Define Experiment
dataset = Dataset_NiiGz_3D(slice=2)
device = torch.device(config[0]['device'])
ca1 = BackboneNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
ca2 = BackboneNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], input_channels=config[0]['input_channels']).to(device)
ca = [ca1, ca2]
agent = Agent_Med_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

loss_function = DiceBCELoss() 

agent.train(data_loader, loss_function)

# Average Dice Score on Test set
#agent.getAverageDiceScore()
