from typing import Type, Dict, Any, TYPE_CHECKING, List, Union, TypeVar, Optional, Callable
from torch.utils.data import Dataset
import torch
import torchio
import numpy as np
from src.agents.Agent import BaseAgent
from PIL import Image
import types
from src.datasets.Data_Instance import Data_Container
from src.utils.DatasetClassCreator import Extended_Loadable_Dataset_Annotations
from src.utils.Experiment_vis import Experiment_vis
from src.utils.BasicNCA3DVis import VisualizationModel
from src.utils.ModelCreatorVis import Extended_Visualization_Model_Annotations
from src.utils.helper import merge_img_label_gt
import torch.nn as nn
import os
from os.path import join
import copy

class Extended_Visualization_Agent_Annotations(BaseAgent):
    """This is a pseudo base class for vizualization agents used only for type checking. Works in VSCOde/ PyCharm.
       In the current Setup, this is largely deprecated.
    """
    if TYPE_CHECKING:
        def __init__(self, model: Extended_Visualization_Model_Annotations, config: List[Dict[str, Any]], dataset: Extended_Loadable_Dataset_Annotations = None, 
                     dataset_type: Type = None, datase_kwargs_args: Dict[str, Any] = None):
            """Initializes the Agent. The agent should be connected to the dataset after this and be able to work normally

            Args:
                model (VisualizationModel): Model that has been augmented to allow visualization
                config (List[Dict[str, Any]]): Config for the neural network
                dataset (Extended_Loadable_Dataset_Annotations, optional)
                dataset_type (Type, optional): _description_. Defaults to None.
                datase_kwargs_args (Dict[str, Any], optional): _description_. Defaults to None.
            """
            pass
            
        def render_slice(self, src_image: np.ndarray, prediction: torch.Tensor, label: torch.Tensor, storepath: str):
            """
            Renders Slices for source image, prediction and label into an image and saves it to disc.

            Attributes:
            src_image: 2D slice of source image (input data)
            prediction: 2D slice of Network prediction
            label: 2D slice of corresponding label
            storepath: fully qualified storage path (including file name)
            """
            pass
    

        def get_output_for_image_monitored(self, image: str, output_path: str, slice_number: int = None, instrumentation_function: Callable[[np.ndarray, int], bool] = None, altered_input: np.ndarray = None, 
                                        save_images: bool = True) -> Dict[int, np.ndarray]:
            pass

        def get_output_for_image(self, image: str, output_path: str, altered_input: np.ndarray = None, save_image: bool = True):
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
            pass
        
        
        
class VisualizationAgentCreator():
    """
    Class used to dynamically create Agents to be used for vizualization. Can also be used for other purposes. Main advantages are decoupling from 
    the experiment, flexible save paths for model and augmented get_output functions that give access to data for each step.
    """
    
    @classmethod
    def create_visualization_agent(cls, base_class: Type[BaseAgent]) -> Type[Extended_Visualization_Agent_Annotations]:
        
        def anonymous_constructor(self, dataset: Extended_Loadable_Dataset_Annotations, model: nn.Module|List[nn.Module], config: List[Dict[str, Any]]): 
            super(NewVisualizationAgentClass, self).__init__(model=model)
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
            
        def render_slice(self, src_image: np.ndarray, prediction: torch.Tensor, label: torch.Tensor, storepath: str):
            """
            Renders Slices for source image, prediction and label into an image and saves it to disc.
            Remnant of an earlier development stage. DEPRECATED.
            Attributes:
            src_image: 2D slice of sourc image (input data)
            prediction: 2D slice of Network prediction
            label: 2D slice of corresponding label
            storepath: fully qualified storage path (including file name)
            """
            image = merge_img_label_gt(src_image, torch.sigmoid(prediction).numpy(), label.numpy())
            image = Image.fromarray(np.uint8(np.squeeze(image)*255)).convert('RGB')
            image.save(storepath, "PNG")
        
        def get_output_for_image_monitored(self, image: str, output_path: str,  slice_number: int = None, instrumentation_function: Callable[[np.ndarray, int], bool] = None, altered_input: np.ndarray = None, 
                                       save_images: bool = True) -> Dict[int, np.ndarray]:
            """
            Renders all slices for prediction for specified image.

            Paramaters:
            image: file name of image from dataset
            output_path: path to folder in which outputs are to be stored
            altered_input: Optional altered input image for whcih output is computed instead 
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
            altered_input: Optional altered input image for whcih output is computed instead 
            of the correspnding image from the dataset. Intended to manually alter the dataset. 
            The image has to be already preprocessed. the get_source_image methods handle all the preprocessing necessary. 
            When altering values, final intensities have to be in [0,1]
            """
            with torch.no_grad():
                dataset: Extended_Loadable_Dataset_Annotations = self.dataset
                id = dataset.get_dataset_index_for_filename_slice(fname= image, slice_num=slice_number)
                # this is going to make problems for 2D Data TODO: FIX 
                data = dataset.__getitem__(id)
                fresh_image = dataset.get_source_image_for_id(id)
                id, input, label = data
                if altered_input is not None:
                    if altered_input.shape != input.shape:
                        raise Exception(
                            f"Shape of altered input image does not match shape \
                            of original image: \
                            {altered_input.shape} vs {input.shape}"
                        )
                    input = altered_input
                    fresh_image = copy.deepcopy(input)
                input = np.expand_dims(input, axis=0)
                label = np.expand_dims(label, axis=0)
                input = torch.from_numpy(input).to(self.device)
                label = torch.from_numpy(label).to(self.device)
                data = (id, input, label)
                _, inputs, _ = data
                data = self.prepare_data(data)

                outputs, targets = self.get_outputs(data)
                patient_3d_image = outputs.detach().cpu()
                patient_3d_label = targets.detach().cpu()
                if save_image:
                    for m in range(patient_3d_image.shape[-1]):
                        if len(patient_3d_label.shape) == 4:
                            patient_3d_label = patient_3d_label.unsqueeze(dim=-1)
                        for i in range(0, fresh_image.shape[2]):
                            self.render_slice(fresh_image[:,:,i:i+1],patient_3d_image[:,:,:,i:i+1,m] ,patient_3d_label[:,:,:,i:i+1,m] , join(output_path, "out_" + str(i) + ".png"))
                        
                else:
                    return torch.sigmoid(patient_3d_image).numpy(), patient_3d_label

        
            
        
        NewVisualizationAgentClass = type("ConstructorInitializedDataset", (base_class, ), {
            # constructor
            "__init__": anonymous_constructor, 
            
            # function members
            "render_slice": render_slice,
            "get_output_for_image_monitored": get_output_for_image_monitored,
            "get_output_for_image": get_output_for_image,
            ""
            # object attribes
            "dataset": None,
            # Dataset has to be of type Extended loadable Dataset.
            "config": None,
            # List[Dict[str, Any]]
            "device": None
            # torch.device
            
        })
        return NewVisualizationAgentClass