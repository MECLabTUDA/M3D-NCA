import torch
from src.agents.Agent import BaseAgent
from src.utils.helper import convert_image, merge_img_label_gt, merge_img_label_gt_simplified
import numpy as np
import math 
from matplotlib import pyplot as plt
import nibabel as nib
import os
from src.losses.LossFunctions import DiceLoss

class Agent_MedSeg2D(BaseAgent):
    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', pseudo_ensemble: bool = False, dataset=None, save_meanVariance=False, **kwargs):
        r"""Evaluate model on testdata by merging it into 3d volumes first
            TODO: Clean up code and write nicer. Replace fixed images for saving in tensorboard.
            #Args
                dataset (Dataset)
                loss_f (torch.nn.Module)
                steps (int): Number of steps to do for inference
        """
        loss_f = DiceLoss()
        # Prepare dataset for testing
        if dataset is None:
            dataset = self.exp.dataset
        self.exp.set_model_state('test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        # Prepare arrays
        patient_id, patient_3d_image, patient_3d_label, average_loss, patient_count = None, None, None, 0, 0
        patient_real_Img = None
        loss_log = {}
        for m in range(self.output_channels):
            loss_log[m] = {}
        if save_img == None:
            save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

        # For each data sample
        for i, data in enumerate(dataloader):
            data = self.prepare_data(data, eval=True)
            data_id, inputs, _ = data['id'], data['image'], data['label']
            if 'name' in data:
                name = data['name']
            outputs, targets = self.get_outputs(data, full_img=True, tag="0")

            if isinstance(data_id, str):
                _, id, slice = dataset.__getname__(data_id).split('_')
            else:
                text = str(data_id[0]).split('_')
                if len(text) == 3:
                    _, id, slice = text
                else:
                    id = data_id[0]
                    slice = None

                # Run inference 10 times to create a pseudo ensemble
                if pseudo_ensemble: # 5 + 5 times
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
                        
                        # save images in path
                        outputs, _ = torch.median(stack, dim=0)

            
            # --------------- 2D ---------------------
            # If next patient
            if (id != patient_id or dataset.slice == -1) and patient_id != None:
                out = str(patient_id) + ", "
                for m in range(patient_3d_label.shape[3]):
                    if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):


                        if "4993" in name: 
                            plt.imshow(np.squeeze(torch.sigmoid(patient_3d_image[0, :, :, m]).detach().cpu().numpy()))
                            plt.show()
                            plt.imshow(np.squeeze(patient_3d_label[0, :, :, m].detach().cpu().numpy()), alpha=0.5)
                            plt.show()
                            exit()
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() #,, mask = patient_3d_label[...,4].bool()

                        if save_meanVariance:
                            if True:
                                # SAVE MEAN
                                out_mean = np.swapaxes(np.squeeze(torch.sigmoid(outputs).detach().cpu().numpy()), 0, 1)
                                out_mean[out_mean > 0.5] = 1
                                out_mean[out_mean <= 0.5] = 0
                                nifti_img = nib.Nifti1Image(out_mean[:, :, np.newaxis], affine=np.eye(4))
                                filename = name[0]

                                pred_path = os.path.join(os.path.dirname(dataset.images_path), 'pred')

                                if not os.path.exists(pred_path):
                                    os.makedirs(pred_path)

                                filename = os.path.join(pred_path, filename)
                                nib.save(nifti_img, filename)
                            if True:
                                stdd = self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy() )

                                # SAVE VARIANCE
                                stdd = np.swapaxes(np.squeeze(stdd), 0, 1)
                                nifti_img = nib.Nifti1Image(stdd[:, :, np.newaxis], affine=np.eye(4))
                                filename = name[0]
                                

                                variance_path = os.path.join(os.path.dirname(dataset.images_path), 'variance')

                                if not os.path.exists(variance_path):
                                    os.makedirs(variance_path)

                                filename = os.path.join(variance_path, filename)
                                nib.save(nifti_img, filename)

                                if False:
                                    # Save mask
                                    stack_path = '/home/jkalkhof_locale/Downloads/miccai_chestx8_mimic/stack/'

                                    if not os.path.exists(stack_path):
                                        os.makedirs(stack_path)

                                    filename = name[0]
                                    filename = os.path.join(stack_path, filename)
                                    stack = torch.squeeze(stack.detach().cpu())

                                    stack = np.swapaxes(np.squeeze(torch.sigmoid(stack).numpy())[..., np.newaxis], 0, 2)
                                    stack[stack > 0.5] = 1
                                    stack[stack <= 0.5] = 0

                                    print(stack.shape)
                                    nifti_img = nib.Nifti1Image(stack, affine=np.eye(4))
                                    nib.save(nifti_img, filename)


                print("PATIENT ID", name, out)
                patient_id, patient_3d_image, patient_3d_label = id, None, None
            # If first slice of volume
            if patient_3d_image == None:
                patient_id = id
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                patient_real_Img = inputs.detach().cpu()
            else:
                patient_3d_image = torch.vstack((patient_3d_image, outputs.detach().cpu()))
                patient_3d_label = torch.vstack((patient_3d_label, targets.detach().cpu()))
                patient_real_Img = torch.vstack((patient_real_Img, inputs.detach().cpu()))
            
            # Add image to aim
            if i in save_img: 
                self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                merge_img_label_gt_simplified(patient_real_Img[0:1, ...].transpose(1,3), torch.sigmoid(patient_3d_image[0:1, ...]), patient_3d_label[0:1, ...]),
                                self.exp.currentStep)
        # If 2D
        out = str(patient_id) + ", "
        for m in range(patient_3d_label.shape[-1]):
            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 
                out = out + str(loss_log[m][patient_id]) + ", "
            else:
                out = out + " , "
        print(out)
        # Print dice score per label
        for key in loss_log.keys():
            if len(loss_log[key]) > 0:
                average = sum(loss_log[key].values())/len(loss_log[key])
                print("Average Dice Loss 3d: " + str(key) + ", " + str(average))
                print("Standard Deviation 3d: " + str(key) + ", " + str(self.standard_deviation(loss_log[key])))
                self.exp.write_scalar('Loss/test/' + str(key), average, self.exp.currentStep)
                self.exp.write_scalar('Loss/test_std/' + str(key), self.standard_deviation(loss_log[key]), self.exp.currentStep)

        self.exp.set_model_state('train')
        return loss_log
    
    def labelVariance(self, images: torch.Tensor, median: torch.Tensor, img_mri: torch.Tensor, img_id: str, targets: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = np.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = np.sqrt(stdd)

        print("NQM Score: ", np.sum(stdd) / np.sum(median))
        return stdd