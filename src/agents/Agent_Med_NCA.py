import torch
import numpy as np
from src.agents.Agent_Multi_NCA import Agent_Multi_NCA
import random
import torchio as tio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import torchio as tio

class Agent_Med_NCA(Agent_Multi_NCA):
    """Med-NCA training agent that uses 2d patches across 2-levels during training to optimize VRAM.
    """
    def initialize(self):
        super().initialize()
        self.stacked_models = self.exp.get_from_config('stacked_models')
        self.scaling_factor = self.exp.get_from_config('scaling_factor')   


    def visualizeInference(self, sample=12, erase=None, artifact=None, compute_variance=False, save_as_gif=False):
        with torch.no_grad():
            # Prepare dataset for testing
            dataset = self.exp.dataset
            self.exp.set_model_state('test')
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # For each data sample
            for i, data in enumerate(dataloader):
                if i != sample:
                    continue
                data = self.prepare_data(data, eval=True)

                data_id, inputs, _ = data

                # perturbation
                if erase is not None:
                    inputs[0, 24:32, 34:42, 0] = 0
                # Apply ghosting artifact
                if artifact is not None:
                    selected_slice = inputs[0, :, :, 0]  
                    selected_slice = selected_slice.unsqueeze(0).unsqueeze(-1)  
                    artifact_slice = artifact(selected_slice).to(self.exp.get_from_config('device'))  
                    artifact_slice = artifact_slice.squeeze(0).squeeze(-1)  
                    inputs[0, :, :, 0] = artifact_slice  

                    
                if compute_variance:
                    # Initialize a list to store inference outputs
                    inference_list = []

                    # Perform 10 forward passes
                    for _ in range(10):
                        outputs, targets, inference_temp = self.get_outputs(data, full_img=True, tag="0", returnInference=True)
                        inference_list.append(torch.sigmoid(inference_temp).detach().cpu().numpy())

                    # Convert the list to a numpy array with shape (10, H, W, C)
                    inference_array = np.stack(inference_list, axis=0)

                    # Compute the standard deviation across the 10 inference passes (along the first axis)
                    inference_std = np.std(inference_array, axis=0)

                    # Overwrite the original inference with the standard deviation
                    inference = torch.tensor(inference_std).to(self.exp.get_from_config('device'))
                else:
                    outputs, targets, inference = self.get_outputs(data, full_img=True, tag="0", returnInference=True)

                # Visualize the inference steps
                inference_np = inference.detach().cpu().numpy()

                if False: #save_input_as_image:
                    input_image_filename = "input.png"
                    # Save the input image
                    plt.figure()
                    plt.imshow(inputs[0, ..., 0].detach().cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    plt.savefig(input_image_filename, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f"Input image saved as {input_image_filename}")

                if save_as_gif:
                    gif_filename = "inference.gif"
                    fig, ax = plt.subplots()
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
                    img = ax.imshow(inference_np[0])
                    ax.axis('off')

                    def update(step):
                        img.set_data(inference_np[step])
                        return [img]

                    ani = FuncAnimation(fig, update, frames=len(inference_np), blit=True, repeat=True)
                    #ani.save(gif_filename, writer='pillow', fps=6)
                    #ani.save(gif_filename, writer='pillow', fps=6, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0, 'frameon': False})
                    ani.save(gif_filename, writer='pillow', dpi=300)
                    plt.close(fig)
                    print(f"Inference GIF saved as {gif_filename}")

                # Show input and output
                # Create a new figure for the input image and output segmentation
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                ax1.imshow(inputs[0, ..., 0].detach().cpu().numpy(), cmap='gray')
                ax1.set_title('Input Image')
                ax1.axis('off')

                # Display the output segmentation
                ax2.imshow(torch.sigmoid(outputs[0, ..., 0]).detach().cpu().numpy()) #, cmap='gray')
                ax2.set_title('Output Segmentation')
                ax2.axis('off')

                # Create the animation for the inference steps
                img = ax3.imshow(inference_np[0]) #, cmap='gray')
                ax3.set_title('Inference Steps (GIF)')
                ax3.axis('off')

                def update(step):
                    img.set_data(inference_np[step])
                    return [img]

                ani = FuncAnimation(fig, update, frames=len(inference_np), blit=True, repeat=True)

                # Instead of plt.show(), display the animation in the notebook
                display(HTML(ani.to_jshtml()))

                plt.close()

                # Set the number of rows and columns
                rows, cols = 4, 8

                # Create a figure with subplots
                fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

                # Plot each inference step in its respective subplot
                for i in range(rows):
                    for j in range(cols):
                        step_index = i * cols + j  # Calculate the correct index
                        axes[i, j].imshow(inference_np[step_index]) #, cmap='gray')
                        axes[i, j].axis('off')  # Turn off the axis for each subplot

                # Adjust spacing between subplots
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.show()

                return


    def get_outputs(self, data, full_img=False, returnInference=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets = data

        # Create down-scaled image
        down_scaled_size = (int(inputs.shape[1] / 4), int(inputs.shape[2] / 4))
        inputs_loc = self.resize4d(inputs.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device')) 
        targets_loc = self.resize4d(targets.cpu(), size=down_scaled_size).to(self.exp.get_from_config('device'))

        # for visualization
        if returnInference:
            inference = torch.zeros((self.getInferenceSteps()*2, inputs_loc.shape[1]*4, inputs_loc.shape[2]*4)).to(self.exp.get_from_config('device'))


        # After training run inference on full image
        if full_img == True:
            with torch.no_grad():
                # Start with low res lvl and go to high res level
                for m in range(self.exp.get_from_config('train_model')+1):
                    if m == self.exp.get_from_config('train_model'):
                        if returnInference is False:
                            outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        else:
                            for i in range(self.getInferenceSteps()):
                                outputs = self.model[m](inputs_loc, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                inference[i+self.getInferenceSteps()] = outputs[0, ..., 1]
                                inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                    else:
                        if returnInference is False:
                            outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                        else:
                            for i in range(self.getInferenceSteps()):
                                outputs = self.model[m](inputs_loc, steps=1, fire_rate=self.exp.get_from_config('cell_fire_rate'))
                                scaled_output = F.interpolate(outputs[0, ..., 1].unsqueeze(0).unsqueeze(0), scale_factor=4, mode='nearest')
                                inference[i] = scaled_output.squeeze(0).squeeze(0)
                                inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                        # Upscale lowres features to high res
                        up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                        outputs = torch.permute(outputs, (0, 3, 1, 2))
                        outputs = up(outputs)
                        inputs_loc = inputs     
                        outputs = torch.permute(outputs, (0, 2, 3, 1))       
                        # Concat lowres features with high res image     
                        inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                        targets_loc = targets
        # During training run inference on patches
        else:
            # Start with low res lvl and go to high res level
            for m in range(self.exp.get_from_config('train_model')+1):
                if m == self.exp.get_from_config('train_model'):
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))
                else:
                    outputs = self.model[m](inputs_loc, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))

                    # Upscale lowres features to high res
                    up = torch.nn.Upsample(scale_factor=4, mode='nearest')
                    outputs = torch.permute(outputs, (0, 3, 1, 2))
                    outputs = up(outputs)
                    inputs_loc = inputs     
                    outputs = torch.permute(outputs, (0, 2, 3, 1))
                    # Concat lowres features with high res image             
                    inputs_loc = torch.concat((inputs_loc[...,:self.input_channels], outputs[...,self.input_channels:]), 3)
                    targets_loc = targets

                    # Prepare array to store patch of 
                    size = self.exp.get_from_config('input_size')[0]
                    inputs_loc_temp = inputs_loc
                    targets_loc_temp = targets_loc
                    inputs_loc = torch.zeros((inputs_loc.shape[0], size[0], size[1], inputs_loc.shape[3])).to(self.exp.get_from_config('device'))
                    targets_loc = torch.zeros((targets_loc_temp.shape[0], size[0], size[1], targets_loc_temp.shape[3])).to(self.exp.get_from_config('device'))

                    # Choose random patch of upscaled image
                    for b in range(inputs_loc.shape[0]): 
                        pos_x = random.randint(0, inputs_loc_temp.shape[1] - size[0])
                        pos_y = random.randint(0, inputs_loc_temp.shape[2] - size[1])

                        inputs_loc[b] = inputs_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]
                        targets_loc[b] = targets_loc_temp[b, pos_x:pos_x+size[0], pos_y:pos_y+size[1], :]

        # Add pooling - not functional
        if self.exp.get_from_config('Persistence'):
            if np.random.random() < self.exp.get_from_config('pool_chance'):
                self.epoch_pool.addToPool(outputs.detach().cpu(), id)

        if returnInference == False:
            return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc 
        else:
            return outputs[..., self.input_channels:self.input_channels+self.output_channels], targets_loc, inference


    def resize4d(self, img, size=(64,64), factor=4, label=False):
        r"""Resize input image
            #Args
                img: 4d image to rescale
                size: image size
                factor: scaling factor
                label: is Label?
        """
        if label:
            transform = tio.Resize((size[0], size[1], -1), image_interpolation='NEAREST')
        else:
            transform = tio.Resize((size[0], size[1], -1))
        img = transform(img)
        return img