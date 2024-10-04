from src.utils.ExperimentWrapper import ExperimentWrapper
from src.utils.Experiment import merge_config
import numpy as np
from ..losses.LossFunctions import DiceBCELoss
from torch.utils.data import Dataset

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from unet import UNet2D
from src.agents.Agent_UNet import UNetAgent 


class EXP_UNet2D(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'UNet2D',
            'lr': 1e-4,
            'batch_duplication': 1,
            # Model
            'batch_size': 16,
        }
        
        config = merge_config(merge_config(study_config, config), detail_config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = UNet2D(in_channels=1, padding=1, out_classes=1)
        agent = UNetAgent(model)
        loss_function = DiceFocalLoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    
from src.models.Model_M3DNCA import M3DNCA
from src.agents.Agent_M3DNCA_Simple import M3DNCAAgent
from src.losses.LossFunctions import DiceFocalLoss

class EXP_M3DNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'M3DNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 16,        # Number of CA state channels
            'inference_steps': 20,
            'cell_fire_rate': 0.5,
            'batch_size': 4,
            'hidden_size': 64,
            'train_model':3,
            # Data
            'scale_factor': 4,
            'kernel_size': 7,
            'levels': 2,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D()
        model = M3DNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], kernel_size=config['kernel_size'], input_channels=config['input_channels'], levels=config['levels'], scale_factor=config['scale_factor'], steps=config['inference_steps'])
        agent = M3DNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
    
from src.models.Model_MedNCA import MedNCA
from src.agents.Agent_MedNCA_Simple  import MedNCAAgent

class EXP_MEDNCA(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):
        config = {
            'description': 'MEDNCA',
            'lr': 16e-4,
            'batch_duplication': 1,
            # Model
            'channel_n': 32,        # Number of CA state channels
            'inference_steps': 64,
            'cell_fire_rate': 0.5,
            'batch_size': 12,
            'hidden_size': 128,
            'train_model':1,
            'betas': (0.9, 0.99),
            # Data
            'scale_factor': 4,
            'kernel_size': 3,
            'levels': 2,
            'input_size': (320,320) ,
        }

        config = merge_config(merge_config(study_config, config), detail_config)
        print("CONFIG", config)
        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        model = MedNCA(config['channel_n'], config['cell_fire_rate'], device=config['device'], hidden_size=config['hidden_size'], input_channels=config['input_channels'], steps=config['inference_steps'])
        agent = MedNCAAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)

import argparse
from src.models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from src.models.vit_seg_modeling import VisionTransformer as ViT_seg
from ..utils.ProjectConfiguration import ProjectConfiguration
class EXP_TransUNet(ExperimentWrapper):
    def createExperiment(self, study_config : dict, detail_config : dict = {}, dataset : Dataset = None):

        config = { 'description': 'TransUNet',
                  'lr': 1e-4,
                  'batch_size': 16}
        config = merge_config(merge_config(study_config, config), detail_config)

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
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        # Load TransUNet weights
        model.load_from(weights=np.load(ProjectConfiguration.VITB16_WEIGHTS))#config_vit.pretrained_path))

        if dataset is None:
            dataset = Dataset_NiiGz_3D(slice=2)
        agent = UNetAgent(model)
        loss_function = DiceBCELoss() 

        return super().createExperiment(config, model, agent, dataset, loss_function)
 


    

