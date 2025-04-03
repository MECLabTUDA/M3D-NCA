from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container
from src.datasets.Dataset_Base import Dataset_Base
import cv2
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
from PIL import Image
from os import listdir
from os.path import join
import os
import random

class Dataset_BCSS(Dataset_Base):

    def __init__(self):
        super().__init__()
        self.slice = 0
        self.color_label_dic = {
            # id, category
            0: 0,
            1: 1, 
            2: 2,
            3: 3,
            4: 4, 
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11, 
            12: 12,
            13: 13,
            14: 14, 
            15: 15,
            16: 16,
            17: 17,
            18: 18,
            19: 19,
            20: 20,
            21: 21, 
            22: 22,
            23: 23,
            24: 24, 
            25: 25,
            26: 26,
            27: 27,
            28: 28,
            29: 29,
        }
        self.color_label_dic = {
            # id, category
            1: 0, 
            2: 1,
            3: 2,
            4: 3,
            0: 4,
        }

    def getFilesInPath(self, path):
        r"""Get files in path
            Args:
                path (string): The path which should be worked through
            Returns:
                dic (dictionary): {key:file_name, value: file_name}
        """
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            id = f[:-4]
            dic[id] = {}
            dic[id][0] = f
        return dic

    def __getname__(self, idx):
        r"""Get name of item by id"""
        return self.images_list[idx]

    def __getitem__(self, idx):
        r"""Standard get item function
            Args:
                idx (int): Id of item to loa
            Returns:
                img (numpy): Image data
                label (numpy): Label data
        """

        img_id = self.__getname__(idx)
        out = self.data.get_data(key=img_id)
        if not out:
            img = Image.open(os.path.join(self.images_path, self.images_list[idx]))
            label = 0 
        

            img, label = self.preprocessing(img, label)


            img_id = "_" + str(img_id)[:-4].replace("_", "") + "_0"
            self.data.set_data(key=img_id, data=(img_id, img, label))

            print("LOAD DATA")
            out = self.data.get_data(key=img_id)

        # Random tile

        img_id, img, label = out

        pos_x = random.randint(0, img.shape[0] - self.size[0])
        pos_y = random.randint(0, img.shape[1] - self.size[1])

        img2 = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]


        return img_id, img2, img2  
    
    def preprocessing(self, img, label):
        r"""Preprocessing of image
            Args:
                img (numpy): Image to preprocess
                label (numpy): Label to preprocess
        """

        img = (np.array(img)/128) -1

        # Randomly Choose Tile
        pos_x = random.randint(0, img.shape[0] - self.size[0])
        pos_y = random.randint(0, img.shape[1] - self.size[1])

        img_loc = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]

        return img_loc, img_loc
