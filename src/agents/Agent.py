import nibabel as nib
import numpy as np
import os
import torch
import torch.optim as optim
from src.utils.helper import convert_image, merge_img_label_gt
from src.losses.LossFunctions import DiceLoss
import seaborn as sns
import math

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

    def getAverageDiceScore(self, useSigmoid=True, tag = "", pseudo_ensemble=False):
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        diceLoss = DiceLoss(useSigmoid=useSigmoid)
        loss_log = self.test(diceLoss, save_img=[], pseudo_ensemble=pseudo_ensemble)

        return loss_log

    def save_state(self, model_path):
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))

    def load_state(self, model_path, device: str = ""):
        r"""Load state of current model

        If device is given, model is loaded onto device.
        Device should only be given for inference, during trainig model should 
        be loaded onto device specified in the stored model
        """
        if device != "":

            print(os.path.join(model_path, 'model.pth'))
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth'), map_location=device))
            self.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth'), map_location=device))
            self.scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pth'), map_location=device))
        else:     
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




    def labelVariance(self, images, median, img_mri, img_id, targets):
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

       
        return

    def test(self, loss_f, save_img = None, tag='test/img/', pseudo_ensemble=False, **kwargs):
        r"""Evaluate model on testdata by merging it into 3d volumes first

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
                        
                        # Calculate median
                        outputs, _ = torch.median(stack, dim=0)
                        self.labelVariance(torch.sigmoid(stack).detach().cpu().numpy(), torch.sigmoid(outputs).detach().cpu().numpy(), inputs.detach().cpu().numpy(), id, targets.detach().cpu().numpy() )

                    else:
                        outputs, _ = torch.median(torch.stack([outputs, outputs2, outputs3, outputs4, outputs5], dim=0), dim=0)

                # --------------- 2D ---------------------
                if dataset.slice is not None:
                    # If next patient
                    if id != patient_id and patient_id != None:
                        out = patient_id + ", "
                        for m in range(patient_3d_image.shape[3]):
                            if(1 in np.unique(patient_3d_label[...,m].detach().cpu().numpy())):
                                loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item() 

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
                                           merge_img_label_gt(inputs.detach().cpu().numpy(), torch.sigmoid(outputs.detach().cpu().numpy()), targets.detach().cpu().numpy()), 
                                           self.exp.currentStep)
                                           

                # --------------------------------- 3D ----------------------------
                else: 
                    patient_3d_image = outputs.detach().cpu()
                    patient_3d_label = targets.detach().cpu()
                    patient_3d_real_Img = inputs.detach().cpu()
                    patient_id = id
                    print(patient_id)

                    print(patient_3d_image.shape,patient_3d_label.shape )
                    for m in range(patient_3d_image.shape[-1]):
                        loss_log[m][patient_id] = 1 - loss_f(patient_3d_image[...,m], patient_3d_label[...,m], smooth = 0).item()
                        print(",",loss_log[m][patient_id])
                        # Add image to tensorboard
                        if True: 
                            if len(patient_3d_label.shape) == 4:
                                patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                            middle_slice = int(patient_3d_real_Img.shape[3] /2)
                            self.exp.write_img(str(tag) + str(patient_id) + "_" + str(len(patient_3d_image)),
                                            merge_img_label_gt(patient_3d_real_Img[:,:,:,middle_slice:middle_slice+1,0].numpy(), torch.sigmoid(patient_3d_image[:,:,:,middle_slice:middle_slice+1,m]).numpy(), patient_3d_label[:,:,:,middle_slice:middle_slice+1,m].numpy()), 
                                            self.exp.currentStep)

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