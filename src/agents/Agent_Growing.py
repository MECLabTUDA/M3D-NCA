import numpy as np
import torch
from src.utils.helper import convert_image, merge_img_label_gt
import math
from PIL import Image as PILImage

import cv2
from src.agents.Agent_NCA import Agent_NCA


class Agent_Growing(Agent_NCA):
    def get_outputs(self, data, full_img=False, tag: str = ""):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data
        outputs = self.model(inputs, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)
        return outputs[..., 0:4], targets
    
    
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
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        if len(outputs.shape) == 5:
            for m in range(outputs.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[...])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()
        else:
            for m in range(outputs.shape[-1]):
                loss_loc = loss_f(outputs[..., m], targets[..., m])
                loss = loss + loss_loc
                loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return loss_ret
    
    
    def make_seed(self, img):
        r"""Create a seed for the NCA 
            #Args
                shape ([int, int]): height, width shape
                n_channels (int): Number of channels
        """
        # 2D
        if( self.exp.dataset.slice != None):
            if len(img.shape) == 3:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., :img.shape[3]] = img
            else:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., 0:img.shape[-1]] = img 

        # 3D
        else:
            if len(img.shape) == 3:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., :img.shape[3]] = img
            else:
                seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], self.exp.get_from_config('channel_n')), dtype=torch.float32, device=self.device)
                seed[..., 0:img.shape[-1]] = img 

        return seed


    
    def test(self, loss_f, save_img = None, tag='test/img/', pseudo_ensemble=False, **kwargs):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        with torch.no_grad():
            # Prepare dataset for testing
            dataset = self.exp.dataset
            self.exp.set_model_state('test')
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            # Prepare arrays
            patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
            patient_real_Img = None
            loss_log = {}
            # loss log contains one dict for each output channel...
            for m in range(self.output_channels):
                loss_log[m] = {}
            if save_img == None:
                save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

            # For each data sample
            for i, data in enumerate(dataloader):
                data = self.prepare_data(data, eval=True)
                data_id, inputs, _ = data
                outputs, targets = self.get_outputs(data, full_img=True, tag="0")

                if isinstance(data_id, str):
                    _, id, slice = dataset.__getname__(data_id).split('_')
                else:
                    text = data_id.detach().cpu().numpy()
                    id = text


                # Run inference 10 times to create a pseudo ensemble
                if pseudo_ensemble:
                    outputs2, _ = self.get_outputs(data, full_img=True, tag="1")
                    outputs3, _ = self.get_outputs(data, full_img=True, tag="2")
                    outputs4, _ = self.get_outputs(data, full_img=True, tag="3")
                    outputs5, _ = self.get_outputs(data, full_img=True, tag="4")
                    if True: 
                        outputs6, _ = self.get_outputs(data, full_img=True, tag="5")
                        outputs7, _ = self.get_outputs(data, full_img=True, tag="6")
                        outputs8, _ = self.get_outputs(data, full_img=True, tag="7")
                        outputs9, _ = self.get_outputs(data, full_img=True, tag="8")
                        outputs10, _ = self.get_outputs(data, full_img=True, tag="9")
                        stack = torch.stack([outputs, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7, outputs8, outputs9, outputs10], dim=0)
                        
                        # Calculate median
                        outputs, _ = torch.median(stack, dim=0)
                        self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy() )

                    else:
                        outputs, _ = torch.median(torch.stack([outputs, outputs2, outputs3, outputs4, outputs5], dim=0), dim=0)
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_3d_real_Img = inputs.detach().cpu()
                patient_id = int(id[0])
                print(patient_id)

                print(patient_3d_image.shape,patient_3d_label.shape )
                for m in range(patient_3d_image.shape[-1]):
                    loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item()
                    print(",",loss_log[m][patient_id])
                    # Add image to tensorboard
                if True: 
                        
                    out_data = torch.sigmoid(outputs.detach().cpu()).numpy()
                    self.exp.write_img2(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), out_data, self.exp.currentStep)
                        
                
            for key in loss_log.keys():
                if len(loss_log[key]) > 0:
                    print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
                    a = loss_log[key]
                    print(type(a))
                    print("Standard Deviation 3d: " + str(key) + ", " + str(standard_deviation(loss_log[key])))

            self.exp.set_model_state('train')
            return loss_log
        
    
        
def standard_deviation(loss_log):
    r"""Calculate the standard deviation
        #Args
            loss_log: losses
    """
    mean = sum(loss_log.values())/len(loss_log)
    stdd = 0
    for e in loss_log.values():
        stdd = stdd + pow(e - mean, 2)
    stdd = stdd / len(loss_log)
    stdd = math.sqrt(stdd)
    return stdd
