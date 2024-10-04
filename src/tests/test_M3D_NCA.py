import tempfile
import torch
from ..datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from ..models.Model_BasicNCA3D import BasicNCA3D
from ..models.Model_M3DNCA import M3DNCA
from ..losses.LossFunctions import DiceFocalLoss
from ..utils.Experiment import Experiment
from ..agents.Agent_M3D_NCA import Agent_M3D_NCA
from ..agents.Agent_M3DNCA_Simple import M3DNCAAgent
from ..tests.test_Med_NCA import create_temp_noise_nifti, create_temp_ones_zeros_nifti, create_testdata
from ..utils.ProjectConfiguration import ProjectConfiguration

def test_M3DNCA():
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()


    config = [{
    'img_path': path_img,
    'label_path': path_label,
    'name': r'test3d', #12 or 13, 54 opt, 
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
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [4, 4],
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(16, 16, 13),(64, 64, 52)], # 
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': True,
    'rescale': True,
    }
    ]

    dataset = Dataset_NiiGz_3D()
    device = torch.device(config[0]['device'])
    ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca3 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca4 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=3, input_channels=config[0]['input_channels']).to(device)
    ca = [ca1, ca2, ca3, ca4]
    agent = Agent_M3D_NCA(ca)
    exp = Experiment(config, dataset, ca, agent)
    #dataset.set_experiment(exp)
    #exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceFocalLoss() 

    agent.train(data_loader, loss_function)

    agent.getAverageDiceScore(pseudo_ensemble=False)

def test_M3DNCA_simplified():
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()


    config = [{
    'img_path': path_img,
    'label_path': path_label,
    'name': r'test3d_simplified', #12 or 13, 54 opt, 
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
    'batch_duplication': 1,
    # Model
    'channel_n': 16,        # Number of CA state channels
    'inference_steps': [4, 4],
    'cell_fire_rate': 0.5,
    'batch_size': 4,
    'input_channels': 1,
    'output_channels': 1,
    'hidden_size': 64,
    'train_model':1,
    # Data
    'input_size': [(16, 16, 13),(64, 64, 52)], # 
    'scale_factor': 4,
    'data_split': [0.7, 0, 0.9], 
    'keep_original_scale': True,
    'rescale': True,
    }
    ]

    dataset = Dataset_NiiGz_3D()
    device = torch.device(config[0]['device'])
    ca1 = M3DNCA(config[0]['channel_n'], config[0]['cell_fire_rate'], device, hidden_size=config[0]['hidden_size'], kernel_size=7, input_channels=config[0]['input_channels'], levels=2, scale_factor=4, steps=20).to(device)
    ca = ca1
    agent = M3DNCAAgent(ca)
    exp = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    loss_function = DiceFocalLoss() 

    agent.train(data_loader, loss_function)

    agent.getAverageDiceScore(pseudo_ensemble=False)