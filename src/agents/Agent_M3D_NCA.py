import torch
import numpy as np
from src.agents.Agent_Multi_NCA import Agent_Multi_NCA
import os
import random
import math
import nibabel as nib

class Agent_M3D_NCA(Agent_Multi_NCA):
    """M3D-NCA training agent that uses 3d patches across n-levels during training to optimize VRAM.
    """
    def initialize(self):
        super().initialize()
        self.stacked_models = self.exp.get_from_config('stacked_models')
        self.scaling_factor = self.exp.get_from_config('scaling_factor')

    def get_outputs(self, data, full_img=False, tag="", **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data

        if len(targets.shape) < 5:
            targets = torch.unsqueeze(targets, 4)
        
        # Set scaling factor
        scale_fac = 2
        if self.exp.get_from_config('scale_factor') is not None:
            scale_fac = self.exp.get_from_config('scale_factor')

        # Choose Pooling
        max_pool = torch.nn.MaxPool3d(2, 2, 0)
        
        targets_loc = targets 

        # Scale Image to Initial Size
        full_res = inputs
        full_res_gt = targets
        inputs_loc = inputs

        # Scale image down square(scale_factor) -> Replace with single downscaling step
        for i in range(self.exp.get_from_config('train_model')*int(math.log2(scale_fac))):
            inputs_loc = inputs_loc.transpose(1,4)
            inputs_loc = max_pool(inputs_loc)
            inputs_loc = inputs_loc.transpose(1,4)
            targets_loc = targets_loc.transpose(1,4)
            targets_loc = max_pool(targets_loc)
            targets_loc = targets_loc.transpose(1,4)

        input_channel = self.exp.get_from_config('input_channels')
        

        # After training run inference on full image
        if full_img == True:

            # REFACTOR: Visualisation
            save4d = False
            slice_all_Channels = False
            if not slice_all_Channels:
                label_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1], inputs.shape[2], inputs.shape[3]))
                img_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1], inputs.shape[2], inputs.shape[3]))
            else:
                x_size = math.ceil(math.sqrt(inputs.shape[4]))
                label_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1]*x_size, inputs.shape[2]*x_size,1), dtype=float)
                img_mri_4d = np.empty((sum(self.getInferenceSteps()), inputs.shape[1]*x_size, inputs.shape[2]*x_size,1), dtype=float)
            step = 0
            # -------------------------
            
            with torch.no_grad():
                # Start with low res lvl and go to high res level
                for m in range(self.exp.get_from_config('train_model')+1):
                    if m == self.exp.get_from_config('train_model'):
                        if type(self.getInferenceSteps()) is list:
                            stp = self.getInferenceSteps()[m]
                        else:
                            stp = self.getInferenceSteps()
                        # REFACTOR: Visualisation
                        if save4d:
                            outputs = inputs_loc
                            for i in range(self.getInferenceSteps()[m]):
                                outputs = self.model[m](outputs, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                if not slice_all_Channels:
                                    label_mri_4d[step, ...] = outputs[0, ..., 1].detach().cpu().numpy() 
                                    img_mri_4d[step, ...] = inputs_loc[0, ..., 0].detach().cpu().numpy() 
                                else:
                                    for x in range(4):
                                        for y in range(4):
                                            label_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = outputs[0, ..., 11, x+y*4].detach().cpu().numpy() 
                                            img_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = inputs_loc[0, ..., 11, x+y*4].detach().cpu().numpy() 

                                step = step +1 
                        else:
                            # Standard inference
                            outputs = self.model[m](inputs_loc, steps=stp, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                    # Scale m-1 times 
                    else:
                        up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')

                        # REFACTOR: Visualisation
                        if save4d:
                            outputs = inputs_loc
                            for i in range(self.getInferenceSteps()[m]):
                                outputs = self.model[m](outputs, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                if not slice_all_Channels:
                                    label_mri_4d[step, ...] = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 1].detach().cpu().numpy() 
                                    img_mri_4d[step, ...] = torch.permute(up(torch.permute(inputs_loc, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 0].detach().cpu().numpy() 
                                else:
                                    for x in range(4):
                                        for y in range(4):
                                            label_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = torch.permute(up(torch.permute(outputs, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 11, x+y*4].detach().cpu().numpy() 
                                            img_mri_4d[step, x*320:(x+1)*320, y*320:(y+1)*320, 0] = torch.permute(up(torch.permute(inputs_loc, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1))[0, ..., 11, x+y*4].detach().cpu().numpy()                                     
                                step = step +1 
                        else:
                            outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        

                        # Upscale lowres features to next level
                        outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                        outputs = up(outputs)
                        inputs_loc = inputs     
                        outputs = torch.permute(outputs, (0, 2, 3, 4, 1))         
   
                        # Create higher res image for next level -> Replace with single downscaling step
                        next_res = full_res
                        for i in range(self.exp.get_from_config('train_model') - (m +1)):
                            next_res = next_res.transpose(1,4)
                            next_res = max_pool(next_res)
                            next_res = next_res.transpose(1,4)

                        # Concat lowres features with higher res image
                        inputs_loc = torch.concat((next_res[...,:input_channel], outputs[...,input_channel:]), 4)
                        targets_loc = targets
            
            # REFACTOR: Visualisation
            if save4d:
                if not slice_all_Channels:
                    nib_save = torch.sigmoid(torch.from_numpy(np.transpose(label_mri_4d, (1, 2, 3, 0)))).numpy()
                    nib_save[nib_save>0.5] = 1
                    nib_save[nib_save != 1] = 0 
                else:
                    nib_save = torch.from_numpy(np.transpose(label_mri_4d, (1, 2, 3, 0))).numpy() 
                    sign = nib_save<0
                    
                nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) 
                nib.save(nib_save, os.path.join("path", str(id)+"_"+tag+".nii.gz"))

                nib_save = torch.sigmoid(torch.from_numpy(np.transpose(img_mri_4d, (1, 2, 3, 0)))).numpy()
                nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                nib.save(nib_save, os.path.join("path", str(id)+"_img.nii.gz"))
        # During training run inference on patches
        else:
            # For number of downscaling levels
            for m in range(self.exp.get_from_config('train_model')+1): 
                # If last step -> run normal inference on final patch
                if m == self.exp.get_from_config('train_model'):
                    if type(self.getInferenceSteps()) is list:
                        stp = self.getInferenceSteps()[m]
                    else:
                        stp = self.getInferenceSteps()
                    outputs = self.model[m](inputs_loc, steps=stp, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                else:
                    # Create higher res image for next level -> Replace with single downscaling step
                    next_res = full_res
                    for i in range(self.exp.get_from_config('train_model') - (m +1)):
                        next_res = next_res.transpose(1,4)
                        next_res = max_pool(next_res)
                        next_res = next_res.transpose(1,4)
                    # Create higher res groundtruth for next level -> Replace with single downscaling step
                    next_res_gt = full_res_gt
                    for i in range(self.exp.get_from_config('train_model') - (m +1)):
                        next_res_gt = next_res_gt.transpose(1,4)
                        next_res_gt = max_pool(next_res_gt)
                        next_res_gt = next_res_gt.transpose(1,4)

                    # Run model inference on patch
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps()[m], fire_rate=self.exp.get_from_config('cell_fire_rate'))
                    
                    # Upscale lowres features to next level
                    up = torch.nn.Upsample(scale_factor=scale_fac, mode='nearest')
                    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                    outputs = up(outputs)
                    outputs = torch.permute(outputs, (0, 2, 3, 4, 1))        
                    # Concat lowres features with higher res image
                    inputs_loc = torch.concat((next_res[...,:input_channel], outputs[...,input_channel:]), 4)

                    # Array to store intermediate states
                    targets_loc = next_res_gt
                    size = self.exp.get_from_config('input_size')[0]
                    inputs_loc_temp = inputs_loc
                    targets_loc_temp = targets_loc

                    # Array to store next states
                    inputs_loc = torch.zeros((inputs_loc_temp.shape[0], size[0], size[1], size[2] , inputs_loc_temp.shape[4])).to(self.exp.get_from_config('device'))
                    targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], size[2] , targets_loc_temp.shape[4])).to(self.exp.get_from_config('device'))
                    full_res_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res.shape[4])).to(self.exp.get_from_config('device'))
                    full_res_gt_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac), full_res_gt.shape[4])).to(self.exp.get_from_config('device'))

                    # Scaling factors
                    factor = self.exp.get_from_config('train_model') - m -1
                    factor_pow = math.pow(2, factor)

                    # Choose random patch of image for each element in batch
                    for b in range(inputs_loc.shape[0]): 
                        while True:
                            pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                            pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])
                            pos_z = random.randint(0, inputs_loc_temp.shape[3] - size[2])
                            break

                        # Randomized start position for patch
                        pos_x_full = int(pos_x * factor_pow)
                        pos_y_full = int(pos_y * factor_pow)
                        pos_z_full = int(pos_z * factor_pow)
                        size_full = [int(full_res.shape[1]/scale_fac), int(full_res.shape[2]/scale_fac), int(full_res.shape[3]/scale_fac)]

                        # Set current patch of inputs and targets
                        inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        if len(targets_loc.shape) > 4:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2], :]
                        else:
                            targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

                        # Update full res image to patch of full res image
                        full_res_new[b] = full_res[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]
                        full_res_gt_new[b] = full_res_gt[b, pos_x_full:pos_x_full+size_full[0], pos_y_full:pos_y_full+size_full[1], pos_z_full:pos_z_full+size_full[2], :]

                    full_res = full_res_new
                    full_res_gt = full_res_gt_new

        # Add pooling - not functional
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc 

