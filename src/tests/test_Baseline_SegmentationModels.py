import pytest
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.utils.Experiment import Experiment
import torch
from src.agents.Agent_UNet import UNetAgent
import numpy as np
import argparse
from src.models.vit_seg_modeling import VisionTransformer as ViT_seg
from src.models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from ..utils.ProjectConfiguration import ProjectConfiguration
import tempfile
from ..tests.test_Med_NCA import create_testdata
from ..losses.LossFunctions import DiceBCELoss

def test_UNet2D():
    pytest.skip("Test not implemented yet")

def test_UNet3D():
    pytest.skip("Test not implemented yet")

def test_MinimalUNet2D():
    pytest.skip("Test not implemented yet")

def test_TransNet2D():
    #pytest.skip("Test not implemented yet")
    ProjectConfiguration.STUDY_PATH = tempfile.mkdtemp()

    path_img, path_label = create_testdata()

    config = [{
        'img_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/imagesTr/",
        'label_path': r"/home/jkalkhof_locale/Documents/Data/Prostate_MEDSeg/labelsTr/",
        'name': r'test_transunet',
        'device':"cuda:0",
        # Learning rate
        'lr': 1e-4,
        'lr_gamma': 0.9999,
        'betas': (0.9, 0.99),
        # Training config
        'save_interval': 1,
        'evaluate_interval': 1,
        'n_epoch': 1,
        'batch_size': 16,
        # Data
        'input_size': (320, 320),
        'data_split': [0.7, 0, 0.3], 

    }]

    # Define Experiment
    dataset = Dataset_NiiGz_3D(slice=2)
    device = torch.device(config[0]['device'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=320, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args(parser._get_args())

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # Load TransUNet weights
    net.load_from(weights=np.load(ProjectConfiguration.VITB16_WEIGHTS))#config_vit.pretrained_path))

    

    agent = UNetAgent(net)
    exp = Experiment(config, dataset, net, agent)
    exp.set_model_state('train')
    dataset.set_experiment(exp)
    loss_function = DiceBCELoss() 
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=exp.get_from_config('batch_size'))

    agent.train(data_loader, loss_function)
    agent.getAverageDiceScore(pseudo_ensemble=False)


