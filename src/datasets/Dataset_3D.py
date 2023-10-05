from src.datasets.Dataset_Base import Dataset_Base
import cv2
import numpy as np

class Dataset_3D(Dataset_Base):
    r"""Base class to load 3D datasets
        .. WARNING:: Not to be used directly!
    """
    def __init__(self, slice=None, resize=True): 
        self.slice = slice
        self.count = 42
        super().__init__(resize)

    def getImagePaths(self):
        r"""Get a list of all images in dataset
            #Returns:
                list ([String]): List of images
        """
        return self.images_list

    def getItemByName(self, name):
        r"""Get item by its name
            #Args
                name (String)
            #Returns:
                item (tensor): The image tensor
        """
        idx = self.images_list.index(name)
        return self.__getitem__(idx)
    
    def resize_image(self, img, isLabel):
        r"""TODO REMOVE OR USE"""
        if not isLabel:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC) 
        else:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_NEAREST) 
        return img
    
    def preprocessing(self, img, isLabel=False):
        r"""Preprocessing of image slices
            #Args
                img (tensor): the image
                isLabel (boolean): Whether its a mask or an image
            .. warning:: Likely there is a preprocessing problem since performance is worse than the already preprocessed slices. ( I imagine the scaling functionality of the mask is at fault)
        """
        if not isLabel:
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Remove different phases
        if len(img.shape) > 2:
            img = img[:, :, 0] 

        #img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = np.expand_dims(img, axis=-1)

        if isLabel:
            img[..., 0][img[...,0] != 0] = 1
            #img[...,1] = 0
            #img[...,2] = 0

        return img

    def __getitem__():
        r"""Placeholder function for getting a dataset value"""
        return None, None, None