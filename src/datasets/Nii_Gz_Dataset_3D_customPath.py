from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio
from os import listdir

class Dataset_NiiGz_3D_customPath(Dataset_NiiGz_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """
    def __init__(self, resize: bool =True, imagePath = "", labelPath = "", size=[64,64,64], slice = None) -> None: 
        super().__init__(resize)

        self.images_path = imagePath
        self.labels_path = labelPath
        self.size = size
        self.slice=slice

        self.images_list = []
        self.labels_list = []

        if slice is None:
            for i, img in enumerate(listdir(self.images_path)):
                self.images_list.append((img, i, i))
            
            for i, img in enumerate(listdir(self.labels_path)):
                self.labels_list.append((img, i, i))
        else:
            for i, img in enumerate(listdir(self.images_path)):
                if self.slice is not None:
                    for slice in range(self.getSlicesOnAxis(os.path.join(self.images_path, img), self.slice)):
                        self.images_list.append((img, i, slice))
                    for slice in range(self.getSlicesOnAxis(os.path.join(self.labels_path, img), self.slice)):
                        self.labels_list.append((img, i, slice))

        
        self.length = len(self.images_list)

        # TODO ADD PATH AND INDIVIDUAL IMAGE AND LIST NAMES

    def getImagePaths(self) -> list:
        r"""Get a list of all images in dataset
            #Returns:
                list ([String]): List of images
        """
        return self.length