import numpy as np
import nibabel as nib
import tempfile
import os
import torch
from ..datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from ..models.Model_BackboneNCA import BackboneNCA
from ..losses.LossFunctions import DiceBCELoss
from ..utils.Experiment import Experiment
from ..agents.Agent_Med_NCA import Agent_Med_NCA
from ..utils.ProjectConfiguration import ProjectConfiguration
import pytest 
from ..agents.Agent_MedNCA_Simple import MedNCAAgent
from ..models.Model_MedNCA import MedNCA
import os

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# Create a function to generate and save random noise data to a NIfTI file
def create_temp_noise_nifti(shape, filename):
    noise_data = np.random.rand(*shape)
    nifti_image = nib.Nifti1Image(noise_data, affine=np.eye(4), dtype=np.float64)
    nib.save(nifti_image, filename)

# Create a function to generate and save NIfTI files filled with ones and zeros
def create_temp_ones_zeros_nifti(shape, filename):
    data = np.full(shape, 1)
    nifti_image = nib.Nifti1Image(data, affine=np.eye(4), dtype=np.int64)
    nib.save(nifti_image, filename)

def create_testdata():
    volume_shape = (64, 64, 64)

    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()

    # Create subdirectories for noise data and ones and zeros
    noise_dir = os.path.join(temp_dir, "noise")
    ones_zeros_dir = os.path.join(temp_dir, "ones_zeros")

    os.makedirs(noise_dir)
    os.makedirs(ones_zeros_dir)

    # Create temporary NIfTI files for noise data
    temp_noise_files = []
    for i in range(5):
        filename = os.path.join(noise_dir, f"noise_{i}.nii.gz")
        create_temp_noise_nifti(volume_shape, filename)
        temp_noise_files.append(filename)

    # Create temporary NIfTI files for ones and zeros
    temp_ones_zeros_files = []
    for i in range(5):
        filename = os.path.join(ones_zeros_dir, f"noise_{i}.nii.gz")
        create_temp_noise_nifti(volume_shape, filename)
        temp_noise_files.append(filename)

    return noise_dir, ones_zeros_dir



def test_MedNCA():
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()

    config = [{
    'img_path': path_img,
    'label_path': path_label,
    'name': r'test', #12 or 13, 54 opt, 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
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
    'input_size': [(16, 16, 16), (64, 64, 64)] ,
    'scale_factor': 2,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
    }
    ]

    # Initialize experiments
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
    agent.getAverageDiceScore()

def test_MedNCA_simplified():
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()

    config = [{
    'img_path': path_img,
    'label_path': path_label,
    'name': r'test', #12 or 13, 54 opt, 
    'device':"cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 16e-4,
    'lr_gamma': 0.9999,#0.9999,
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
    'input_size': [(16, 16, 16), (64, 64, 64)] ,
    'scale_factor': 2,
    'data_split': [0.7, 0, 0.3], 
    'keep_original_scale': False,
    'rescale': True,
    }
    ]

    # Define Experiment
    dataset = Dataset_NiiGz_3D(slice=2)
    device = torch.device(config[0]['device'])
    ca = MedNCA(channel_n=17, fire_rate=0.5, steps=64, device = "cuda:0", hidden_size=128, input_channels=1, output_channels=1).to("cuda:0")
    agent = MedNCAAgent(ca)
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