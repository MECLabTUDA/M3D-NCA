import cv2
import os
from  src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
import numpy as np
import torchvision.transforms as T
import torch
import random
import csv
import re


class png_Dataset(Dataset_NiiGz_3D):

    normalize = True
    

    def __init__(self, crop=False, buffer=False, downscale=4):
        super().__init__()
        self.crop = crop
        self.buffer = buffer
        self.downscale = downscale
        self.slice = 2
        self.csv = []
        with open('/home/jkalkhof_locale/Documents/Data/list_attr_celeba.txt', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    self.headers = next(reader)  # Read the headers
                    self.headers = [h for h in self.headers if h]  # Remove empty entries from headers
                    for row in reader:
                        cleaned_row = [r for r in row if r]
                        self.csv.append(cleaned_row)


    def set_normalize(self, normalize=True):
        self.normalize = normalize

    def get_properties(self, file_number):
        properties = []
        line_number = file_number  # Adding 1 because the first row is headers

        row = self.csv[line_number - 1]  # Get the desired line
        properties = np.array(row[1:], dtype=int)
        return properties

        with open(newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            headers = next(reader)  # Read the headers

            # Skip to the desired line
            for _ in range(line_number - 1):
                next(reader)

            # Read the desired line
            row = next(reader)
            row = [r for r in next(reader) if r]
            properties = np.array(row[1:], dtype=int)
            #print(properties)

            #properties = [headers[i] for i, value in enumerate(row[1:]) if value == '1']

        return properties

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

        # load csv
        match = re.search(r'(\d{6})\.jpg$', path)
        id = int(match.group(1))

        file_properties = self.get_properties(id)  # For 000001.jpg

        #img = img/256 
        #img = img*2 -1
        #print("MINMAX", torch.max(img), torch.min(img))
        return img, file_properties

    def getSlicesOnAxis(self, path: str, axis: int):
        return self.load_item(path)[0].shape[axis]

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
                img_name, p_id, img_id = self.images_list[idx]
                label_name, _, _ = self.labels_list[idx]

                img, file_properties = self.load_item(os.path.join(self.images_path, img_name))
                label = 0#self.load_item(os.path.join(self.labels_path, img_name))

                self.data.set_data(key=self.images_list[idx], data=(img_id, img, label, file_properties))
                img = self.data.get_data(key=self.images_list[idx])
                img_id, img, label, file_properties = img
                img = (img_id, img, img, file_properties)
        else:
            img_name, p_id, img_id = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            img, file_properties = self.load_item(os.path.join(self.images_path, img_name))
            label = img
            img = (img_id, img, label, file_properties)


        id, img, label, file_properties = img

        if self.crop:
            pos_x = random.randint(0, img.shape[0] - self.size[0])
            pos_y = random.randint(0, img.shape[1] - self.size[1])

            img = img[pos_x:pos_x+self.size[0], pos_y:pos_y+self.size[1], :]
            label = img

        #from matplotlib import pyplot as plt
        #plt.imshow(img[:,:,:])#outputs_fft[0, 0, :, :].real.detach().cpu().numpy())
        #plt.show()

        img = img[...,0:4]

        if self.normalize:
            if False:
                transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                img = torch.from_numpy(img).to(torch.float64)
                img = img.permute((2, 1, 0))
                #print(img.shape)
                img = transform(img)
                img = img.permute((2, 1, 0))
            #img = img * 2 -1
            #img = img
            img = img/128 -1#img/256/2.5 -1  #img/128 -1 #img/256/2.5 -1 #/2.5 -1

        #print(img.shape)

        data_dict = {}
        data_dict['id'] = id
        data_dict['image'] = img
        data_dict['label'] = label
        data_dict['file_properties'] = file_properties


        return data_dict

