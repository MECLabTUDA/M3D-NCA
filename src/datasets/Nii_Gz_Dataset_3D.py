from src.datasets.Dataset_3D import Dataset_3D
import nibabel as nib
import os
import numpy as np
import cv2
import random
import torchio

class Dataset_NiiGz_3D(Dataset_3D):
    """This dataset is used for all NiiGz 3D datasets. It can handle 3D data on its own, but is also able to split them into slices. """

    def getDataShapes():
        return

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
        return self.load_item(path).shape[axis]

    def load_item(self, path):
        r"""Loads the data of an image of a given path.
            #Args
                path (String): The path to the nib file to be loaded."""
        return nib.load(path).get_fdata()

    def rotate_image(self, image, angle, label = False):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        if label:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
        else:
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def preprocessing3d(self, img, isLabel=False):
        r"""Preprocess data to fit the required shape
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if not isLabel:
            # TODO: Currently only single volume, no multi phase
            if len(img.shape) == 4:
                img = img[..., 0]
            padded = np.zeros(self.size)#np.random.rand(*self.size) * 0.01
        else:
            padded = np.zeros(self.size)
        img_shape = img.shape
        padded[0:img_shape[0], 0:img_shape[1], 0:img_shape[2]] = img

        return padded

    def rescale3d(self, img, isLabel=False):
        r"""Rescale input image to fit training size
            #Args
                img (numpy): Image data
                isLabel (numpy): Whether or not data is label
            #Returns:
                img (numpy): numpy array
        """
        if len(self.size) == 3:
            size = (self.size[0], self.size[1])
            size2 = (self.size[2], self.size[0])
        else:
            size = (self.size[0], self.size[1])

        img_resized = np.zeros((self.size[0], self.size[1], img.shape[2])) 
        for x in range(img.shape[2]):
            if not isLabel:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_CUBIC) 
            else:
                img_resized[:, :, x] = cv2.resize(img[:, :, x], dsize=size, interpolation=cv2.INTER_NEAREST) 

        if len(self.size) == 3 and True:
            img = img_resized
            img_resized = np.zeros((self.size[0], self.size[1], self.size[2]))
            for x in range(img.shape[1]):
                if not isLabel:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_CUBIC) 
                else:
                    img_resized[:, x, :] = cv2.resize(img[:, x, :], dsize=size2, interpolation=cv2.INTER_NEAREST) 

        return img_resized

    def patchify(self, img, label):
        r"""Take a patch of the input. This should be used instead of rescaling if global information is not required.
            #Args
                img (numpy): Image data
                label (numpy): Label data
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        size = self.size

        containsMask = (random.uniform(0, 1) < self.exp.get_from_config('priotize_masks'))
        while True:
            pos_x = random.randint(0, img.shape[0] - size[0])
            pos_y = random.randint(0, img.shape[1] - size[1])
            pos_z = random.randint(0, img.shape[2] - size[2])

            if containsMask:
                if 1 in np.unique(label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]):
                    break
            else: 
                break
        
        img = img[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]
        label = label[pos_x:pos_x+size[0], pos_y:pos_y+size[1], pos_z:pos_z+size[2]]

        return img, label

    def __getitem__(self, idx):
        r"""Standard get item function
            #Args
                idx (int): Id of item to loa
            #Returns:
                img (numpy): Image data
                label (numpy): Label data
        """
        rescale = torchio.RescaleIntensity(out_min_max=(0,1), percentiles=(0.5, 99.5))
        znormalisation = torchio.ZNormalization()

        img = self.data.get_data(key=self.images_list[idx])
        if not img:
            img_name, p_id, img_id = self.images_list[idx]

            label_name, _, _ = self.labels_list[idx]

            img, label = self.load_item(os.path.join(self.images_path, img_name)), self.load_item(os.path.join(self.labels_path, img_name))
            # 2D
            if self.slice is not None:
                if len(img.shape) == 4:
                    img = img[..., 0]
                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.slice == 0:
                    img, label = img[img_id, :, :], label[img_id, :, :]
                elif self.slice == 1:
                    img, label = img[:, img_id, :], label[:, img_id, :]
                elif self.slice == 2:
                    img, label = img[:, :, img_id], label[:, :, img_id]
                # Remove 4th dimension if multiphase
                if len(img.shape) == 4:
                    img = img[...,0] 
                img, label = self.preprocessing(img), self.preprocessing(label, isLabel=True)
            # 3D
            else:
                if len(img.shape) == 4:
                    img = img[..., 0]
                img = np.expand_dims(img, axis=0)
                img = rescale(img) 
                img = np.squeeze(img)
                if self.exp.get_from_config('rescale') is not None and self.exp.get_from_config('rescale') is True:
                    img, label = self.rescale3d(img), self.rescale3d(label, isLabel=True)
                if self.exp.get_from_config('keep_original_scale') is not None and self.exp.get_from_config('keep_original_scale'):
                    img, label = self.preprocessing3d(img), self.preprocessing3d(label, isLabel=True)  
                # Add dim to label
                if len(label.shape) == 3:
                    label = np.expand_dims(label, axis=-1)
            img_id = "_" + str(p_id) + "_" + str(img_id)
            
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img = self.data.get_data(key=self.images_list[idx])
           

        id, img, label = img

        size = self.size 
        
        # Create patches from full resolution
        if self.exp.get_from_config('patchify') is not None and self.exp.get_from_config('patchify') is True and self.state == "train": 
            img, label = self.patchify(img, label) 

        if len(size) > 2:
            size = size[0:2] 

        # Normalize image
        img = np.expand_dims(img, axis=0)
        if np.sum(img) > 0:
            img = znormalisation(img)
        img = rescale(img) 
        img = img[0]

        # Merge labels -> For now single label
        label[label > 0] = 1

        # Number of defined channels
        if len(self.size) == 2:
            img = img[..., :self.exp.get_from_config('input_channels')]
            label = label[..., :self.exp.get_from_config('output_channels')]

        return (id, img, label)
