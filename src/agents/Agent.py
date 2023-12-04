import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image
from src.losses.LossFunctions import DiceLoss
import seaborn as sns
import math
import matplotlib.pyplot as plt

class BaseAgent():
    """Base class for all agents. Handles basic training and only needs to be adapted if special use cases are necessary.
    
    .. note:: In many cases only the data preparation and outputs need to be changed."""
    def __init__(self, model):
        self.model = model

    def set_exp(self, exp):
        r"""Set experiment of agent and initialize.
            #Args
                exp (Experiment): Experiment class"""
        self.exp = exp
        self.initialize()

    def initialize(self):
        r"""Initialize agent with optimizers and schedulers
        """
        self.device = torch.device(self.exp.get_from_config('device'))
        # If stacked NCAs
        if isinstance(self.model, list):
            self.optimizer = []
            self.scheduler = []
            for m in range(len(self.model)):
                self.optimizer.append(optim.Adam(self.model[m].parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas')))
                self.scheduler.append(optim.lr_scheduler.ExponentialLR(self.optimizer[m], self.exp.get_from_config('lr_gamma')))
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.exp.get_from_config('lr'), betas=self.exp.get_from_config('betas'))
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.exp.get_from_config('lr_gamma'))

    def printIntermediateResults(self, loss, epoch):
        r"""Prints intermediate results of training and adds it to tensorboard
            #Args 
                loss (torch)
                epoch (int) 
        """
        print(epoch, "loss =", loss.item())
        self.exp.save_model()
        self.exp.write_scalar('Loss/train', loss, epoch)

    def prepare_data(self, data, eval=False):
        r"""If any data preparation needs to be done do it here. 
            #Args
                data ([]): The data to be processed.
                eval (Bool): Whether or not its for evaluation. 
        """
        return data

    def get_outputs(self, data, **kwargs):
        r"""Get the output of the model.
            #Args 
                data (torch): The data to be passed to the model.
        """
        return self.model(data)

    def initialize_epoch(self):
        r"""Everything that should happen once before each epoch should be defined here.
        """
        return

    def conclude_epoch(self):
        r"""Everything that should happen once after each epoch should be defined here.
        """
        return

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
                if 1 in targets[..., m]:
                    loss_loc = loss_f(outputs[..., m], targets[..., m])
                    loss = loss + loss_loc
                    loss_ret[m] = loss_loc.item()

        if loss != 0:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        return loss_ret

    def intermediate_results(self, epoch, loss_log):
        r"""Write intermediate results to tensorboard
            #Args
                epoch (int): Current epoch
                los_log ([loss]): Array of losses
        """
        for key in loss_log.keys():
            if len(loss_log[key]) != 0:
                average_loss = sum(loss_log[key]) / len(loss_log[key])
            else:
                average_loss = 0
            print(epoch, "loss =", average_loss)
            self.exp.write_scalar('Loss/train/' + str(key), average_loss, epoch)

    def plot_results_byPatient(self, loss_log):
        r"""Plot losses in a per patient fashion with seaborn to display in tensorboard.
            #Args
                loss_log ({name: loss}: Dictionary of losses
        """
        print(loss_log)
        sns.set_theme()
        plot = sns.scatterplot(x=loss_log.keys(), y=loss_log.values())
        plot.set(ylim=(0, 1))
        plot = plot.get_figure()
        return plot

    def intermediate_evaluation(self, dataloader, epoch):
        r"""Do an intermediate evluation during training 
            .. todo:: Make variable for more evaluation scores (Maybe pass list of metrics)
            #Args
                dataset (Dataset)
                epoch (int)
        """
        diceLoss = DiceLoss(useSigmoid=True)
        loss_log = self.test(diceLoss)
        for key in loss_log.keys():
            img_plot = self.plot_results_byPatient(loss_log[key])
            self.exp.write_figure('Patient/dice/mask' + str(key), img_plot, epoch)
            if len(loss_log[key]) > 0:
                self.exp.write_scalar('Dice/test/mask' + str(key), sum(loss_log[key].values())/len(loss_log[key]), epoch)
                self.exp.write_histogram('Dice/test/byPatient/mask' + str(key), np.fromiter(loss_log[key].values(), dtype=float), epoch)
        param_lst = []
        # TODO: ADD AGAIN 
        #for param in self.model.parameters():
        #    param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)

    def getAverageDiceScore(self, useSigmoid=True, tag = "", pseudo_ensemble=False, showResults=False):
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=[], pseudo_ensemble=pseudo_ensemble, showResults=showResults)

        return loss_log

    def save_state(self, model_path):
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))

    def load_state(self, model_path):
        r"""Load state of current model
        """
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth')))
        self.scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pth')))

    def train(self, dataloader, loss_f):
        r"""Execute training of model
            #Args
                dataloader (Dataloader): contains training data
                loss_f (nn.Model): The loss for training"""
        for epoch in range(self.exp.currentStep, self.exp.get_max_steps()+1):
            print("Epoch: " + str(epoch))
            loss_log = {}
            for m in range(self.output_channels):
                loss_log[m] = []
            self.initialize_epoch()
            print('Dataset size: ' + str(len(dataloader)))
            for i, data in enumerate(dataloader):
                loss_item = self.batch_step(data, loss_f)
                for key in loss_item.keys():
                    loss_log[key].append(loss_item[key])
            self.intermediate_results(epoch, loss_log)
            if epoch % self.exp.get_from_config('evaluate_interval') == 0:
                print("Evaluate model")
                self.intermediate_evaluation(dataloader, epoch)
            #if epoch % self.exp.get_from_config('ood_interval') == 0:
            #    print("Evaluate model in OOD cases")
            #    self.ood_evaluation(epoch=epoch)
            if epoch % self.exp.get_from_config('save_interval') == 0:
                print("Model saved")
                self.save_state(os.path.join(self.exp.get_from_config('model_path'), 'models', 'epoch_' + str(self.exp.currentStep)))
            self.conclude_epoch()
            self.exp.increase_epoch()

    def prepare_image_for_display(self, image):
        r"""Prepare an image to be displayed in tensorboard. Since images need to be in a specific format these modifications these can be done here.
            #Args
                image (torch): The image to be processed for display. 
        """
        return image


    #def ood_evaluation(self, ood_cases=["random_noise", "random_spike", "random_anitrosopy"], epoch=0):
    #    print("OOD EVALUATION")
    #    dataset_train = self.exp.dataset
    #    diceLoss = DiceLoss(useSigmoid=True)
    #    for augmentation in ood_cases:
    #        dataset_eval = Nii_Gz_Dataset(aug_type=augmentation)
    #        self.exp.dataset = dataset_eval
    #        loss_log = self.test(diceLoss, tag='ood/' + str(augmentation) + '/')
    #        for key in loss_log.keys():
    #            self.exp.write_scalar('ood/Dice/' + str(key) + ", " + str(augmentation), sum(loss_log[key].values())/len(loss_log[key]), epoch)
    #            self.exp.write_histogram('ood/Dice/' + str(key) + ", " + str(augmentation) + '/byPatient', np.fromiter(loss_log[key].values(), dtype=float), epoch)
    #    self.exp.dataset = dataset_train


    def labelVariance(self, images, mean, img_mri, img_id, targets, showResults=False):
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                mean: The mean of all inferences
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

        if showResults:
            #print(img_mri)
            image1 = img_mri[0, :, img_mri.shape[2] // 2, :, 0]
            image2 = mean[0, :, mean.shape[2] // 2, :,  0]
            image3 = stdd[0, :, stdd.shape[2] // 2, :,  0]

            # Set up the matplotlib figure and axes
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))

            # Display the first image
            axs[0].imshow(image1, cmap='gray')
            axs[0].axis('off')  # Turn off axis

            # Display the second image
            axs[1].imshow(image2, cmap='Purples')
            axs[1].axis('off')  # Turn off axis

            # Display the second image
            im = axs[2].imshow(image3)
            axs[2].axis('off')  # Turn off axis

            plt.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

            # Add text below each image
            axs[0].text(0.5, -0.1, 'Middle image slice', ha='center', va='center', transform=axs[0].transAxes)
            axs[1].text(0.5, -0.1, 'Mean segmentation before sigmoid', ha='center', va='center', transform=axs[1].transAxes)
            axs[2].text(0.5, -0.1, 'Variance map', ha='center', va='center', transform=axs[2].transAxes)

            # Adjust layout to prevent overlapping
            plt.tight_layout()
            plt.show()

        print("NQM Score: ", np.sum(stdd) / np.sum(mean))

        # Save files refactor
        if False:
            nib_save = np.expand_dims(img_mri[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("path", str(img_id) + "_image.nii.gz"))
            
            nib_save = np.expand_dims(targets[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("path", str(img_id) + "_gt.nii.gz"))

            nib_save = np.expand_dims(stdd[0, ..., 0], axis=-1) 
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("path", str(img_id) + "_variance.nii.gz"))

            nib_save = np.expand_dims(mean[0, ..., 0], axis=-1) 
            nib_save[nib_save > 0.5] = 1 
            nib_save[nib_save != 1] = 0
            nib_save = nib.Nifti1Image(nib_save , np.array(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 4, 0), (0, 0, 0, 1))), nib.Nifti1Header()) #np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)))
            nib.save(nib_save, os.path.join("path", str(img_id) + "_label.nii.gz"))
        
            f = open(os.path.join("path", str(img_id) + "_score.txt"), "a")
            f.write(str(np.sum(stdd) / np.sum(mean)))
            f.close()

        return

    def test(self, loss_f, save_img = None, tag='test/img/', pseudo_ensemble=False, showResults=False, **kwargs):
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
            for m in range(self.output_channels):
                loss_log[m] = {}
            if save_img == None:
                save_img = [1, 2, 3, 4, 5, 32, 45, 89, 357, 53, 122, 267, 97, 389]

            # For each data sample
            for i, data in enumerate(dataloader):
                print("__________________________ CASE " + str(i) + " __________________________")
                data = self.prepare_data(data, eval=True)
                data_id, inputs, _ = data
                outputs, targets = self.get_outputs(data, full_img=True, tag="0")

                if isinstance(data_id, str):
                    _, id, slice = dataset.__getname__(data_id).split('_')
                else:
                    text = data_id[0].split('_')
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
                        
                        # Calculate mean
                        outputs = torch.mean(stack, dim=0)
                        self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy(), showResults=showResults)

                    else:
                        outputs, _ = torch.mean(torch.stack([outputs, outputs2, outputs3, outputs4, outputs5], dim=0), dim=0)

                # --------------- 2D ---------------------
                if dataset.slice is not None:
                    # If next patient
                    if id != patient_id and patient_id != None:
                        out = patient_id + ", "
                        for m in range(patient_3d_image.shape[3]):
                            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() #,, mask = patient_3d_label[...,4].bool()

                                if math.isnan(loss_log[m][patient_id]):
                                    loss_log[m][patient_id] = 0
                                out = out + str(loss_log[m][patient_id]) + ", "
                            else:
                                out = out + " , "
                        print(out)
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
                    # Add image to tensorboard
                    if i in save_img: 
                        self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                        convert_image(self.prepare_image_for_display(inputs.detach().cpu()).numpy(), 
                        self.prepare_image_for_display(outputs.detach().cpu()).numpy(), 
                        self.prepare_image_for_display(targets.detach().cpu()).numpy(), 
                        encode_image=False), self.exp.currentStep)
                # --------------------------------- 3D ----------------------------
                else: 
                    patient_3d_image = outputs.detach().cpu()
                    patient_3d_label = targets.detach().cpu()
                    patient_3d_real_Img = inputs.detach().cpu()
                    patient_id = id
                    print('ID:', patient_id)

                    #print(patient_3d_image.shape,patient_3d_label.shape )
                    for m in range(patient_3d_image.shape[-1]):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item()
                        print("Dice(", m, "): ", loss_log[m][patient_id], ",")
                        # Add image to tensorboard
                        if True: 
                            if len(patient_3d_label.shape) == 4:
                                patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                            self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)), 
                            convert_image(self.prepare_image_for_display(patient_3d_real_Img[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            self.prepare_image_for_display(patient_3d_image[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            self.prepare_image_for_display(patient_3d_label[:,:,:,5:6,:].detach().cpu()).numpy(), 
                            encode_image=False), self.exp.currentStep)

                            # REFACTOR: Save predictions
                            if False:
                                label_out = torch.sigmoid(patient_3d_image[0, ...])
                                nib_save = nib.Nifti1Image(label_out  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + ".nii.gz"))

                                nib_save = nib.Nifti1Image(torch.sigmoid(patient_3d_real_Img[0, ...])  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_real.nii.gz"))

                                nib_save = nib.Nifti1Image(patient_3d_label[0, ...]  , np.array(((0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))), nib.Nifti1Header())
                                nib.save(nib_save, os.path.join("path", str(len(loss_log[0])) + "_ground.nii.gz"))

            # If 2D
            if dataset.slice is not None:
                out = patient_id + ", "
                for m in range(patient_3d_image.shape[3]):
                    if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 
                        out = out + str(loss_log[m][patient_id]) + ", "
                    else:
                        out = out + " , " 
                print(out)
            # Print dice score per label
            for key in loss_log.keys():
                if len(loss_log[key]) > 0:
                    print("Average Dice Loss 3d: " + str(key) + ", " + str(sum(loss_log[key].values())/len(loss_log[key])))
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