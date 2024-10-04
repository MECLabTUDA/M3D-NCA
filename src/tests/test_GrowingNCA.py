import torch
from ..datasets.png_Dataset import png_Dataset
from ..losses.LossFunctions import DiceLoss
from ..utils.Experiment import Experiment
from ..agents.Agent_Growing import Agent_Growing
from ..utils.ProjectConfiguration import ProjectConfiguration
from ..models.Model_GrowingNCA import GrowingNCA
import tempfile
from ..tests.test_FourierDiff_NCA import create_testdata
import pytest

def test_GrowingNCA():
    #pytest.skip("Test not implemented yet")
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()

    config = [{
        # Basic
        'img_path': path_img,
        'label_path': path_label, 
        'name': r'test_growing', 
        'device':"cuda:0",
        'unlock_CPU': True,
        # Optimizer
        'lr': 16e-4, 
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training
        'save_interval': 1,
        'evaluate_interval': 1,
        'n_epoch': 1,
        'batch_size': 16,
        # Model
        'channel_n': 32,        # Number of CA state channels
        'batch_duplication': 1,
        'inference_steps': 10,
        'cell_fire_rate': 0.5,
        'input_channels': 3,
        'output_channels': 3,
        'hidden_size':  128,
        'schedule': 'linear',
        # Data
        'input_size': (32, 32),
        'data_split': [0.7, 0, 0.3], 
        'timesteps': 13,
        '2D': True,
        'unlock_CPU': True,
    }
    ]

    dataset = png_Dataset(buffer=True)
    device = torch.device(config[0]['device'])

    ca = GrowingNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size']).to(device)
    
    agent = Agent_Growing(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceLoss() 

    agent.train(data_loader, loss_function)
    agent.getAverageDiceScore()