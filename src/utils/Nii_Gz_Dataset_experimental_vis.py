from cmath import nan
from os import listdir
from os.path import join
import nibabel as nib
import numpy as np
import cv2
import os
from src.datasets.Dataset_Base import Dataset_Base
import random
import matplotlib.pyplot as plt
from src.datasets.Nii_Gz_Datasets_experimental import Nii_Gz_Dataset
from src.utils.DatasetClassCreator import DatasetClassCreator
from src.datasets.Data_Instance import Data_Container
from typing import Tuple, Dict, Any, List


class Nii_Gz_Dataset_vis(Nii_Gz_Dataset):
    def getFilesInPath(self, path):
        r"""Get files in path ordered by id and slice
            #Args
                path (string): The path which should be worked through
            #Returns:
                dic (dictionary): {key:patientID, {key:sliceID, img_slice}
        """
        dir_files = os.listdir(os.path.join(path))
        dic = {}
        for id_f, f in enumerate(dir_files):
            print(os.path.basename(f))
            if os.path.basename(f).startswith('.'):
                continue
            id = f
            # 2D 
            if self.slice is not None:
                for slice in range(self.getSlicesOnAxis(os.path.join(path, f), self.slice)):
                    if id not in dic:
                        dic[id] = {}
                    dic[id][slice] = (f, id_f, slice)
            # 3D
            else:
                if id not in dic:
                    dic[id] = {}
                dic[id][0] = (f, f, 0)           
        return dic
    
    
    
    def getSlicesOnAxis(self, path, axis):
        tmp = self.load_item(path)
        return tmp.shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()
   
    def __init__(self, image_path: str, labels_path: str, config: Dict[str, Any], slices: int = None, slice_axis: int = None):
        super(Nii_Gz_Dataset_vis, self).__init__(resize=True)
        self.config = config
        DatasetClassCreator.add_required_to_config(config)
        f_names = os.listdir(image_path)
        f_names_labels = os.listdir(labels_path)
        if len(f_names) != len(f_names_labels):
            raise Exception("Number of labels and images has to match")
        retDict: Dict[str, Any] = {}
        self.data = Data_Container()
        self.size = config["input_size"]
        self.slice = slice_axis
        @classmethod
        def get_from_config(self, tag: str) -> Any:
            r"""Get from config
                #Args
                    tag (String): Key of requested value
            """
            if tag in self.config.keys():
                return self.config[tag]
            else:
                return None
        self.exp = type("AnonymousConfigWrapper", (object, ), {"get_from_config": get_from_config, "config": config})
        labels_dict = self.getFilesInPath(image_path)
            
        images_dict = self.getFilesInPath(labels_path)
        l_keys = list(labels_dict.keys())
        i_keys = list(images_dict.keys())
        if len(l_keys) != len(i_keys):
            raise Exception("Cannot handle unequal number of lists and labels")
        images = list()
        labels = list()
        running_index = 0
        fname_id_dict: Dict[str, List[int]] = {}
        for i in range(len(l_keys)):
            new_labels = list(labels_dict[l_keys[i]].values())
            new_images = list(images_dict[i_keys[i]].values())
            images.extend(new_images)
            labels.extend(new_labels)
            unique_ids = [*range(running_index, running_index+len(new_labels))]
            fname_id_dict[l_keys[i]] = unique_ids
            running_index += len(new_labels)
        
        self.setPaths(image_path, images, labels_path, labels)
        
        if isinstance(config['input_size'][0], tuple):
            self.size = config['input_size'][-1]
        else:
            self.size = self.config['input_size']
        # dataset info is used to efficiently retrieve data points for a given file
        self.dataset_info =  fname_id_dict        
        self.slice = slice_axis
    
    
    def get_dataset_index_for_filename_slice(self, fname: str, slice_num: int = 0):
            """
            Returns index that is to be used with __getitem__ to retrieve the corresponding point. 
            If the Network receives unsliced 3D data, use slice_num=None
            """
            if not fname in self.dataset_info:
                return None
            if self.slice is None:
                return self.dataset_info[fname][0]
            else:
                if not slice_num in self.dataset_info[fname]:
                    return self.dataset_info[fname][0]
                else:
                    return self.dataset_info[fname][slice_num]
    def get_dataset_index_information(self):
        """
        Returns:
                Returns Dict
                maps file_name -> list(unique_ids). List of unique id's using which the individual slices of the source 
                image can be queried with __getitem__. id's correspond to individual slices of the source image in ascending order. 
                
        """
        return self.dataset_info
    
    def get_ids_for_filename(self, fname: str):
        d_set = getattr(self, "dataset_info")
        if fname in d_set:
            return d_set[fname]
        else:
            return None
        
    def set_experiment(self, experiment: Any):
        pass
    
    def does_network_receive_3d(self) -> bool:
        """
        Returns whether the network wants to receive 3D data or individual slices.
        """
        return self.slice is None
    
    def get_state(self) -> Tuple[Any]:
        return(self.images_path,
                self.images_list,
                self.labels_path,
                self.labels_list,
                self.length,
                self.size, 
                self.dataset_info)
        
    def set_state(self, state: Tuple[Any]):
        images_path, images_list, labels_path, labels_list, length, size, dataset_info = state
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = length
        self.size = size
        self.dataset_info = dataset_info
        
    def get_source_image_for_id(self, id: int) -> np.ndarray:
        """
        Deprecated
        """
        img_dict: Dict[str, Any] = self.__getitem__(id)
        return img_dict['image']
    
    
    def get_source_image_for_filename(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the whole source image for the given file name as well as the whole label. 
        Independant of whether the dataset represents sliced data. This method should be used when retreiving whole Data Points. 
        """
        label = None
        
        ids = self.dataset_info[filename]
         
        rdict: Dict[str, Any] = self.__getitem__(ids[0])
        fresh_image = rdict['image']
        label = rdict['label']
        return (fresh_image, label)