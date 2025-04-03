import cv2
import os
from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import numpy as np
import torchvision.transforms as T
import torch
import random


class png_Dataset(Dataset_NiiGz_3D):

    normalize = True

    def __init__(self, crop=False, buffer=False, downscale=4):
        super().__init__()
        self.crop = crop
        self.buffer = buffer
        self.downscale = downscale

    def set_normalize(self, normalize=True):
        self.normalize = normalize

    def load_item(self, path: str) -> np.ndarray:
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.crop:
            img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, dsize=(img.shape[1]//self.downscale,img.shape[0]//self.downscale), interpolation=cv2.INTER_CUBIC)

        img = cv2.convertScaleAbs(img)
        return img

    def __getitem__(self, idx: int) -> tuple:
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        if self.buffer:
            img = self.data.get_data(key=self.images_list[idx])

            if not img:
                img_name, _, img_id = self.images_list[idx]

                img = self.load_item(os.path.join(self.images_path, img_name))
                label = 0

                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
                img = self.data.get_data(key=self.images_list[idx])
                img_id, img, label = img
                img = (img_id, img, img)
        else:
            img_name, _, img_id = self.images_list[idx]

            img = self.load_item(os.path.join(self.images_path, img_name))
            label = img
            img = (img_id, img, label)


        id, img, label = img

        if self.crop:
            pos_x = random.randint(0, img.shape[0] - self.size[0])
            pos_y = random.randint(0, img.shape[1] - self.size[1])

            img = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            label = img

        img = img[...,0:4]

        if self.normalize:
            img = img/128 -1

        return (id, img, label)

