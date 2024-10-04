import torch
import torch.nn as nn
from src.models.Model_BasicNCA3D import BasicNCA3D
import torchio as tio
import random
import math
import torch.nn.functional as F
import subprocess as sp

class M3DNCA(nn.Module):
    r"""Implementation of M3D-NCA
    """
    def __init__(self, channel_n, fire_rate, device, steps=64, hidden_size=128, input_channels=1, output_channels=1, scale_factor=4, levels=2, kernel_size=7):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
        """
        super(M3DNCA, self).__init__()

        self.channel_n = channel_n
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.device = device
        self.fire_rate = fire_rate
        self.steps = steps
        self.scale_factor = scale_factor
        self.levels = levels
        self.fast_inf = False
        self.margin = 20

        self.model = nn.ModuleList()
        for i in range(self.levels):
            ks = kernel_size if i == 0 else 3
            self.model.append(BasicNCA3D(channel_n=channel_n, fire_rate=fire_rate, device=device, hidden_size=hidden_size, input_channels=input_channels, kernel_size=ks))

    def make_seed(self, x):
        seed = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], self.channel_n), dtype=torch.float32, device=self.device)
        seed[..., 0:x.shape[-1]] = x 
        return seed


    def get_gpu_memory(self):
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
        except sp.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        return memory_use_values

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, batch_duplication=1):
        x = x.to(self.device)
        #x = x.transpose(1,4)
        #y = y.transpose(1,4)
        #print(x.shape, y.shape)
        y = y.to(self.device)
        if self.training:
            if batch_duplication != 1:
                x = torch.cat([x] * batch_duplication, dim=0)
                y = torch.cat([y] * batch_duplication, dim=0)

            x, y = self.forward_train(x, y)
            return x, y
            
        else:
            x = self.forward_eval(x)
            return x, y

    #@torch.no_grad()
    def downscale_image(self, inputs, targets, iterations=1):
        max_pool = torch.nn.MaxPool3d(2, 2, 0)

        inputs = inputs.transpose(1,4)
        targets = targets.transpose(1,4)
        for _ in range(iterations*int(math.log2(self.scale_factor))): # Multiples of 2 rightnow
            inputs = max_pool(inputs)
            targets = max_pool(targets)
        inputs = inputs.transpose(1,4)
        targets = targets.transpose(1,4)

        return inputs, targets

    def get_inference_steps(self, level=0):
        if isinstance(self.steps , list):
            return self.steps[level]
        return self.steps

    def compute_bbox(self, mask, margin=0):
        # Flatten the first two dimensions (Batch and Channels)
        flattened_mask = mask.view(-1, *mask.shape[-3:])  
        # Get the indices of non-zero elements in the mask
        non_zero_indices = torch.nonzero(flattened_mask)

        # Get the minimum and maximum indices for each dimension of Depth, Height, and Width
        min_idx = non_zero_indices.min(dim=0)[0][1:]
        max_idx = non_zero_indices.max(dim=0)[0][1:]

        # Add the margin
        min_idx = torch.clamp(min_idx - margin, min=0)
        max_idx = torch.clamp(max_idx + margin, max=flattened_mask.shape[-1]-1)

        return min_idx, max_idx

    def crop_to_bbox(self, tensor, min_idx, max_idx):
        # Crop the tensor using the bounding box indices
        return tensor[..., min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, min_idx[2]:max_idx[2]+1]

    def pad_to_multiple(self, tensor, multiple):
        # Calculate necessary padding for Depth, Height, and Width
        depth_pad = (multiple - tensor.shape[-3] % multiple) % multiple
        height_pad = (multiple - tensor.shape[-2] % multiple) % multiple
        width_pad = (multiple - tensor.shape[-1] % multiple) % multiple
        
        # Apply padding
        padded_tensor = F.pad(tensor, (0, width_pad, 0, height_pad, 0, depth_pad))
        
        return padded_tensor

    def forward_train(self, x: torch.Tensor, y: torch.Tensor):
        max_pool = torch.nn.MaxPool3d(2, 2, 0)

        x_shape = x.shape

        full_res, full_res_gt = x, y

        # Crop volume to bounding box
        if False:
            full_res, full_res_gt = full_res.transpose(1,4), full_res_gt.transpose(1,4)
            min_idx, max_idx = self.compute_bbox(full_res_gt, margin=32)
            full_res_gt = self.crop_to_bbox(full_res_gt, min_idx, max_idx)
            full_res = self.crop_to_bbox(full_res, min_idx, max_idx)
            full_res, full_res_gt = self.pad_to_multiple(full_res, self.scale_factor*(self.levels-1)), self.pad_to_multiple(full_res_gt, self.scale_factor*(self.levels-1))
            full_res, full_res_gt = full_res.transpose(1,4), full_res_gt.transpose(1,4)

            x, y, = full_res, full_res_gt

        #print("First HERE")
        #print(self.get_gpu_memory())
        inputs_loc, targets_loc = self.downscale_image(x, y, iterations=self.levels-1)

        inputs_loc = self.make_seed(inputs_loc)

        #print("DOWNSCALED", inputs_loc.shape, targets_loc.shape)

        up = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        # For number of downscaling levels
        for m in range(self.levels): 
            # If last step -> run normal inference on final patch
            if m == self.levels-1:
                outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)
            else:
                #print("STILL HERE")
                #print(self.get_gpu_memory())
                with torch.no_grad():
                    # Create higher res image for next level -> Replace with single downscaling step
                    #next_res = full_res
                    #for i in range(self.levels - (m+2)):
                    #    next_res = next_res.transpose(1,4)
                    #    next_res = max_pool(next_res)
                    #    next_res = next_res.transpose(1,4)
                    # Create higher res groundtruth for next level -> Replace with single downscaling step
                    #next_res_gt = full_res_gt
                    #for i in range(self.levels - (m+2)):
                    #    next_res_gt = next_res_gt.transpose(1,4)
                    #    next_res_gt = max_pool(next_res_gt)
                    #    next_res_gt = next_res_gt.transpose(1,4)
                    #print(next_res.shape, next_res_gt.shape)
                    next_res, next_res_gt = self.downscale_image(full_res, full_res_gt, iterations=self.levels-(m+2))
                    #print(next_res.shape, next_res_gt.shape)

                # Run model inference on patch
                outputs = self.model[m](inputs_loc, steps=self.get_inference_steps(m), fire_rate=self.fire_rate)

                # Upscale lowres features to next level
                outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                outputs = up(outputs)
                outputs = torch.permute(outputs, (0, 2, 3, 4, 1))        
                # Concat lowres features with higher res image
                inputs_loc = torch.concat((next_res[...,:self.input_channels], outputs[...,self.input_channels:]), 4)

                # Array to store intermediate states
                targets_loc = next_res_gt
                size = (x_shape[1]//int(math.pow(self.scale_factor, (self.levels-1))),
                        x_shape[2]//int(math.pow(self.scale_factor, (self.levels-1))),
                        x_shape[3]//int(math.pow(self.scale_factor, (self.levels-1))))
                inputs_loc_temp = inputs_loc
                targets_loc_temp = targets_loc

                # Array to store next states
                inputs_loc = torch.zeros((inputs_loc_temp.shape[0], size[0], size[1], size[2] , inputs_loc_temp.shape[4])).to(self.device)
                targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], size[2] , targets_loc_temp.shape[4])).to(self.device)
                full_res_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor), full_res.shape[4])).to(self.device)
                full_res_gt_new = torch.zeros((full_res.shape[0], int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor), full_res_gt.shape[4])).to(self.device)

                # Scaling factors
                factor = self.levels - m -2
                factor_pow = math.pow(2, factor*int(math.log2(self.scale_factor)))
                #if m == 0:
                #    factor_pow = 1
                #else:
                #    factor_pow = self.scale_factor
                #factor = self.levels - m -2
                #factor_pow = math.pow(self.scale_factor, factor)

                # Choose random patch of image for each element in batch
                for b in range(inputs_loc.shape[0]): 
                    while True:
                        #print(inputs_loc_temp.shape, full_res.shape)
                        pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                        pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])
                        pos_z = random.randint(0, inputs_loc_temp.shape[3] - size[2])
                        break

                    # Randomized start position for patch
                    pos_x_full = int(pos_x * factor_pow)
                    pos_y_full = int(pos_y * factor_pow)
                    pos_z_full = int(pos_z * factor_pow)
                    size_full = [int(full_res.shape[1]/self.scale_factor), int(full_res.shape[2]/self.scale_factor), int(full_res.shape[3]/self.scale_factor)]

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

        return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc
    
    def forward_eval(self, x: torch.Tensor):
        #max_pool = torch.nn.MaxPool3d(2, 2, 0)
        inputs_loc, _ = self.downscale_image(x, x, iterations=self.levels-1)

        inputs_loc = self.make_seed(inputs_loc)

        full_res = x

        no_optim = not self.fast_inf

        with torch.no_grad():
            # Start with low res lvl and go to high res level
            for m in range(self.levels):
                if m == self.levels-1:
                    skipped = 0
                    for s in range(self.get_inference_steps(m)):
                        t_sig = (inputs_loc[..., self.input_channels:self.input_channels+self.output_channels].transpose(1,4) > 0).to(torch.int) #torch.sigmoid(inputs_loc[..., self.input_channels:self.input_channels+self.output_channels].transpose(1,4)).to(torch.int)
                        outputs = inputs_loc
                        if t_sig.__contains__(1) == False or no_optim == True:
                            outputs = self.model[m](inputs_loc, steps=1, fire_rate=self.fire_rate)
                            inputs_loc = outputs
                        else:
                            min_idx, max_idx = self.compute_bbox(t_sig, margin=self.margin)
                            out = self.model[m](outputs[:, min_idx[0]:max_idx[0]+1, min_idx[2]:max_idx[2]+1, min_idx[1]:max_idx[1]+1, :], steps=1, fire_rate=self.fire_rate)
                            skipped += (out.shape[1]*out.shape[2]*out.shape[3])
                            #print(min_idx, max_idx, out.shape, outputs.shape)
                            outputs[:, min_idx[0]:max_idx[0]+1, min_idx[2]:max_idx[2]+1, min_idx[1]:max_idx[1]+1, :] = out
                            inputs_loc = outputs
                    #print("OPT SKIPPED: ", 1-(skipped/(outputs.shape[1]*outputs.shape[2]*outputs.shape[3]*self.get_inference_steps(m))), "\% pixels")
                # Scale m-1 times 
                else:
                    up = torch.nn.Upsample(scale_factor=self.scale_factor, mode='nearest')

                    for s in range(self.get_inference_steps(m)):
                        t_sig = (inputs_loc[..., self.input_channels:self.input_channels+self.output_channels].transpose(1,4) > 0 ).to(torch.int)#torch.sigmoid(inputs_loc[..., self.input_channels:self.input_channels+self.output_channels].transpose(1,4)).to(torch.int)
                        outputs = inputs_loc
                        if m == 0 or t_sig.__contains__(1) == False or no_optim == True:
                            outputs = self.model[m](inputs_loc, steps=1, fire_rate=self.fire_rate)
                            inputs_loc = outputs
                        else:
                            min_idx, max_idx = self.compute_bbox(t_sig, margin=self.margin)
                            out = self.model[m](outputs[:, min_idx[2]:max_idx[2]+1, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, :], steps=1, fire_rate=self.fire_rate)
                            outputs[:, min_idx[2]:max_idx[2]+1, min_idx[0]:max_idx[0]+1, min_idx[1]:max_idx[1]+1, :] = out
                            inputs_loc = outputs

                    # Upscale lowres features to next level
                    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
                    outputs = up(outputs)
                    inputs_loc = x     
                    outputs = torch.permute(outputs, (0, 2, 3, 4, 1))         

                    # Create higher res image for next level -> Replace with single downscaling step
                    #next_res = full_res
                    #for i in range(self.levels - (m +2)):
                    #    next_res = next_res.transpose(1,4)
                    #    next_res = max_pool(next_res)
                    #    next_res = next_res.transpose(1,4)

                    next_res, _ = self.downscale_image(full_res, full_res, iterations=self.levels-(m+2))

                    # Concat lowres features with higher res image
                    #print(next_res.shape, outputs.shape)
                    inputs_loc = torch.concat((next_res[...,:self.input_channels], outputs[...,self.input_channels:]), 4)

        return outputs[..., self.input_channels:self.input_channels+self.output_channels]