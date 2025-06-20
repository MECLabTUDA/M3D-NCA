from typing import Type, Dict, Any, TYPE_CHECKING, List, Union, TypeVar, Optional, Tuple
import torchio
import numpy as np
from src.datasets.Dataset_Base import Dataset_Base
import types
from src.datasets.Data_Instance import Data_Container
import os
"""
Common interface for datasets.
"""
class Extended_Loadable_Dataset_Annotations(Dataset_Base):
    if TYPE_CHECKING:
        def get_from_config(self, tag: str) -> Any:
                r"""Get from config
                    #Args
                        tag (String): Key of requested value
                """
                pass
        def get_dataset_index_information(self) -> Dict[str, List[int]]:
            """
            Returns:
                    Returns Dict
                    maps file_name -> list(unique_ids). List of unique id's using which the individual slices of the source 
                    image can be quarried with __get__item. id's correspond to individual slices of the source image in ascending order. 
                    
            """
            pass
        def __init__(self, image_path: str, labels_path: str, config: Dict[str, Any], slices: int = None, slice_axis: int = None, is_3d: bool = True):
            """Takes a Dataset Object and sets an arbitrary underlying dataset for it

            Args:
                image_path (str): Fully qualified path to Folder containing source images
                labels_path (str): Fully qualified path to Folder containing labels corresponding to source images. 
                    Both folder have to contain only images from the same Dataset. Corresponding labels and images need equal names.
                slices (int): Sets whether the underlying 3D images be sliced along an axis for display 
                purposes, and gives length of the axis (e.g. how many slices per Image).
                config: config
                is_3D: whether the network receives 3D data or individual slices along the axis
            
            """
            pass
        
        def get_ids_for_filename(self, fname: str) -> list(int):
            """
            Returns all retreival IDs that correspond to given filename
            """ 
            pass


        def get_dataset_index_for_filename_slice(self, fname: str, slice_num: int = 0) -> int:
            """Returns the index that can be used with __get_item__ to retreive the specified slice of the 
            corresponding filename. 
            Also works for 3D, non-sliced images. the slice parameter is ignored in this case."""
            pass
        def get_source_image_for_id(self, id: int) -> np.ndarray:
            """
            Returns fresh 3D image for ID (same as given by data loader)
            Image is already a normalized 3D np array and mostly ready for display.
            """
            pass
        
        def get_state(self) -> Tuple[Any]:
            """Likely deprecated
            """
            pass            
        def set_state(self, state: Tuple[Any]):
            """llikely deprecated
            """
            pass

        def does_network_receive_3d(self) -> bool:
            """
            Returns whether the network wants to receive 3D data or individual slices.
            """
            pass


        def get_source_image_for_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
            """
            Returns the whole source image for the given file name as well as the whole label. 
            Independant off whether the dataset produces sliced data
            """
            pass
        
        
class DatasetClassCreator():
    """Deprecated, currently only serves as a housing for a single util method.
    """
    @classmethod
    def add_required_to_config(clas, config: Dict[str, Any]):
        r"""Fills config with basic setup if not defined otherwise

        """
        if 'Persistence' not in config:
            config['Persistence'] = False
        if 'batch_duplication' not in config:
            config['batch_duplication'] = 1
        if 'keep_original_scale' not in config:
            config['keep_original_scale'] = False
        if 'rescale' not in config:
            config['rescale'] = True
        if 'channel_n' not in config:
            config['channel_n'] = 16
        if 'cell_fire_rate' not in config:
            config['cell_fire_rate'] = 0.5
        if 'output_channels' not in config:
            config['output_channels'] = 1
    