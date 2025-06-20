import torch
from src.agents.Agent_UNet import UNetAgent
from src.agents.Agent_MedSeg2D import Agent_MedSeg2D
import torch.nn as nn
from typing import List, Dict, Any
from src.utils.DatasetClassCreator import Extended_Loadable_Dataset_Annotations
from src.utils.Experiment_vis import Experiment_vis
import numpy as np
from src.utils.BasicNCA3DVis import VisualizationModel
from typing import Callable

class MedNCAAgent_simple_vis(UNetAgent):
    """Base agent for training UNet models
    """
    def __init__(self, dataset: Extended_Loadable_Dataset_Annotations, model: nn.Module|List[nn.Module], config: List[Dict[str, Any]]):
        super(MedNCAAgent_simple_vis, self).__init__(model=model)
        dataset_state = dataset.get_state()
        self.dataset = dataset
        self.config = config
        device = torch.device(self.config[0]['device'])
        self.device = device
        if isinstance(model, List):
            models = list()
            for m in model:
                models.append(m.to(device))
            model = models
        else:
            model = model.to(device)
        self.model = model
        exp = Experiment_vis(self.config, self.dataset, self.model, self)
        self.dataset.set_experiment(experiment=exp)
        exp.set_model_state('train')
        dataset.set_state(dataset_state)
        
    def prepare_data(self, data: tuple, eval: bool = False) -> tuple:
        r"""Prepare the data to be used with the model
            #Args
                data (int, tensor, tensor): identity, image, target mask
            #Returns:
                inputs (tensor): Input to model
                targets (tensor): Target of model
        """
        id, inputs, targets = data['id'], data['image'], data['label']
        inputs, targets = inputs.type(torch.FloatTensor), targets.type(torch.FloatTensor)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        if len(inputs.shape) == 4:
            inputs = inputs.permute(0, 3, 1, 2)
            targets = targets.permute(0, 3, 1, 2)
        
        #data = {'id': id, 'image': inputs, 'label': targets}
        data['image'] = inputs
        data['label'] = targets

        return data
        
        
                
    def render_slice(self, src_image: np.ndarray, prediction: torch.Tensor, label: torch.Tensor, storepath: str):
        #deprecated
        raise Exception("Deprecated")
    
    def get_output_for_image_monitored(self, image: str, output_path: str,  slice_number: int = None, instrumentation_function: Callable[[np.ndarray, int], bool] = None, altered_input: np.ndarray = None, 
                                   save_images: bool = True) -> Dict[int, np.ndarray]:
        """
        Renders all slices for prediction for specified image.
        Paramaters:
        image: file name of image from dataset
        output_path: path to folder in which outputs are to be stored
        altered_input: Optional altered input image for which output is computed instead 
        of the correspnding image from the dataset. Intended to manually alter the dataset. 
        The image has to be already preprocessed. the get_source_image methods handle all the preprocessing necessary. 
        When altering values, final intensities have to be in [0,1]
        """
        model: VisualizationModel|List[VisualizationModel] = self.model
        if isinstance(model, List):
            for m in model:
                m.set_instrumentation_function(instrumentation_function)
                m.set_state_dict({})    
        else:        
            model.set_instrumentation_function(instrumentation_function)
            model.set_state_dict({})
        self.get_output_for_image(image=image, output_path=output_path, slice_number=slice_number, altered_input=altered_input, save_image=save_images)
        state_dict: Dict[int, np.ndarray] = {}
        highest_key = 0
        if isinstance(model, List):
            for m in model:
                diict = m.export_state_dict()
                keeys = diict.keys()
                for k in list(keeys):
                    state_dict[k + highest_key] = diict[k]
                highest_key = highest_key + len(list(keeys))
        else:
            state_dict = model.export_state_dict()
        return state_dict
    def get_output_for_image(self, image: str, output_path: str, slice_number: int = None, altered_input: np.ndarray = None, save_image: bool = True):
        """
        Renders all slices for prediction for specified image.
        Paramaters:
        image: file name of image from dataset
        output_path: path to folder in which outputs are to be stored
        altered_input: Optional altered input image for which output is computed instead 
        of the correspnding image from the dataset. Intended to manually alter the dataset. 
        The image has to be already preprocessed. the get_source_image methods handle all the preprocessing necessary. 
        When altering values, final intensities have to be in [0,1]
        """
        with torch.no_grad():
            dataset: Extended_Loadable_Dataset_Annotations = self.dataset
            id = dataset.get_dataset_index_for_filename_slice(fname= image, slice_num=slice_number)

            data = dataset.__getitem__(id)
            fresh_image = dataset.get_source_image_for_id(id)
            id, input, label = data['id'], data['image'], data['label']
            if altered_input is not None:
                if altered_input.shape != input.shape:
                    raise Exception(
                        f"Shape of altered input image does not match shape \
                        of original image: \
                        {altered_input.shape} vs {input.shape}"
                    )
                input = altered_input
 
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)
            input = torch.from_numpy(input).to(self.device)
            label = torch.from_numpy(label).to(self.device)
            data = {'id': id, 'image': input,'label': label}
            _, inputs, _ = data
            
            data = self.prepare_data(data)
            outputs, targets = self.get_outputs(data)
            patient_3d_image = outputs.detach().cpu()
            patient_3d_label = targets.detach().cpu()

            return torch.sigmoid(patient_3d_image).numpy(), patient_3d_label



    
    def get_outputs(self, data: tuple, full_img=True, **kwargs) -> tuple:
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        inputs, targets = data['image'], data['label']
        
        inputs, targets = self.model(inputs, targets)
        return inputs, targets
        