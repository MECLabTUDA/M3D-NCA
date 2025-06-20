from cProfile import label
import pickle
import json
from re import A
import cv2
import numpy as np
import seaborn as sns
import bz2
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io
import datetime
import nibabel as nib
import os
import torch
import warnings

def dump_pickle_file(file, path):
    r"""Dump pickle file in path
        #Args:
            file: the file to dump
            path: location to dump file to
    """
    with open(path, 'wb') as output_file:
        pickle.dump(file, output_file)

def load_pickle_file(path):
    r"""Load pickle file
        #Args:
            path: location to dump file to
    """
    with open(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file

def dump_compressed_pickle_file(file, path):
    r"""Dump compressed pickle file in path
        #Args:
            file: the file to dump
            path: location to dump file to
    """
    with bz2.BZ2File(path, 'w') as output_file:
        pickle.dump(file, output_file)

def load_compressed_pickle_file(path):
    r"""Load compressed pickle file
        #Args:
            path: location to dump file to
    """
    with bz2.BZ2File(path, 'rb') as input_file:
        file = pickle.load(input_file)
    return file
    
def dump_json_file(file, path):
    r"""Dump json file in path
        #Args:
            file: the json file to dump
            path: location to dump file to
    """
    with open(path, 'w') as output_file:
        json.dump(file, output_file)

def load_json_file(path):
    r"""Load json file
        #Args:
            path: location to dump file to
    """
    with open(path, 'r') as input_file:
        file =  json.load(input_file)
    return file


def normalize_image(image):
    image_float = image.to(torch.float32)

    # Normalize the image tensor to be in the range [0, 1]
    min_val = torch.min(image_float)
    max_val = torch.max(image_float)
    normalized = (image_float - min_val) / (max_val - min_val)

    return normalized

def merge_img_label_gt_simplified(img, label, gt, rgb=True):
    if label.size()[-1] != 1:
        label = label[..., 0]
        gt = gt[..., 0]
        warnings.warn("WARNING: Currently image output supports one label only")

    print(img.shape, label.shape, gt.shape)
    img = torch.squeeze(img)
    label = torch.squeeze(label)
    gt = torch.squeeze(gt)

    if len(img.shape) - len(label.shape) == 1:
        img = torch.squeeze(img)[..., 0]

    img, label, gt = normalize_image(img), normalize_image(label), normalize_image(gt)

    merged_image = torch.cat((img, label, gt)).numpy()
    # If 3D
    if len(img.shape) == 3:
       merged_image = merged_image[..., merged_image.shape[2]//2]
    


    return merged_image


def merge_img_label_gt(img, label, gt):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
    img, label, gt = np.squeeze(img), np.squeeze(label), np.squeeze(gt)

    img = np.stack((img, img, img), axis=-1)
    label_overlay = np.zeros(img.shape)

    label_overlay[..., 0] = label

    gt_overlay = np.zeros(img.shape)
    gt_overlay[..., 1] = gt

    img[label_overlay > 0.5] = img[label_overlay > 0.5]*0.5 + label_overlay[label_overlay > 0.5] * 0.5
    img[gt_overlay > 0] = img[gt_overlay > 0]*0.5 + gt_overlay[gt_overlay > 0] * 0.5
    return img

def overlay_sdf_field_nicer(image: torch.Tensor, sdf_field: torch.Tensor) -> np.ndarray:
    """OVerlays an SDF field over the Input image. Negative values are Coded in blue, positive values in red. 

    Args:
        imgage (torch.Tensor): 2 or 3D tensor, single phase representing image intensities. 
            Needs to have a value range between [0, 1]
        sdf_field (torch.Tensor): SDF field, 2 or 3D, equal size to the image, 
            accepts values in the range [-1, 1]

    Returns:
        torch.Tensor: _description_
    """
    if isinstance(image, torch.Tensor):
        img: np.ndarray = image.detach().cpu().numpy().squeeze()
        sdf: np.ndarray = sdf_field.detach().cpu().numpy().squeeze()
    else:
        img = image
        sdf = sdf_field
    img = np.clip(a=img, a_min=0, a_max=1)
    sdf = np.clip(a=sdf, a_min=-1, a_max=1)
    img = np.stack((img, img, img), axis=-1)
    def sigm(arr: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(arr*(-1)))
    sdf_color_field = np.stack((sdf, sdf, sdf), axis=-1)
    sdf_color_field[..., 2] = 0.0
    g_pos = sdf_color_field[..., 1]
    t = sigm(sdf_color_field[..., 1]*4)
    g_pos[t>0] = t[t > 0]
    t = sigm(sdf_color_field[..., 1]*8)
    g_pos[t<0] = t[t<0]
    sdf_color_field[..., 1] = g_pos
    r_neg = 1 - sdf_color_field[..., 0]
    t = sigm((1-sdf_color_field[..., 0])*4)
    r_neg[t>0] = t[t>0]
    t = sigm((1-sdf_color_field[..., 0])*8)
    r_neg[t<0] = t[t<0]
    sdf_color_field[..., 0] = r_neg
    img = img*0.5 + sdf_color_field*0.5
    return img

    
def convert_image(img, prediction, label=None, encode_image=True):
    r"""Convert an image plus an optional label into one image that can be dealt with by Pillow and similar to display
       
            """
    img_rgb = img 
    img_rgb = img_rgb - np.amin(img_rgb)
    img_rgb = img_rgb * img_rgb 
    img_rgb = img_rgb / np.amax(img_rgb)
    label_pred = prediction

    img_rgb, label, label_pred = [orderArray(v.squeeze()) for v in [img_rgb, label, label_pred]]

    
    label = np.amax(label, axis=-1)
    label_pred = np.amax(label_pred, axis=-1)
    label_pred = np.stack((label_pred, label_pred, label_pred), axis=-1)
    

    # Overlay Label on Image
    if label is not None:
        sobel_x = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(src=label, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel = sobel_x + sobel_y
        if len(sobel.shape) < 3:
            sobel = np.stack((sobel, sobel, sobel), axis=-1)

        sobel[:,:,2] = sobel[:,:,0]
        sobel[:,:,0] = 0
        sobel = np.abs(sobel)
        img_rgb[img_rgb < 0] = 0
        label_pred[label_pred < 0] = 0

        sobel = cv2.resize(sobel, dsize=(label_pred.shape[0], label_pred.shape[1])) 
        img_rgb = cv2.resize(img_rgb, dsize=(label_pred.shape[0], label_pred.shape[1]), interpolation=cv2.INTER_NEAREST) 

        img_rgb = np.clip((sobel  * 0.8 + img_rgb + 0.5 * label_pred), 0, 1)

    if sum(img_rgb.shape) > 2000:
        size = (int(img_rgb.shape[0]/6), int(img_rgb.shape[1]/6))
        img_rgb = cv2.resize(img_rgb, dsize=size, interpolation=cv2.INTER_CUBIC) 

    if encode_image:
        img_rgb = encode(img_rgb)
    return img_rgb 

def orderArray(array):

    if len(array.shape) < 3:
        array = np.stack((array, array, array), axis=-1)

    if array.shape[0] < array.shape[2]:
        return np.transpose(array, (1, 2, 0))
    if array.shape[1] < array.shape[2]:
        return np.transpose(array, (0, 2, 1))
    else:
         return array


def encode(img_rgb, size=(150, 100)):
    r"""Encode an image
        #Args:
            img_rgb: the input image
            size: size of the image
    """
    size_img = img_rgb.shape
    size_img = [1, size_img[0]/ size_img[1]]

    size_img_scaledX = [int(x * size[0] * 0.95) for x in size_img] 
    size_img_scaledY = [int(x * size[1] * 0.95) for x in size_img] 

    scale = (10, 10)

    for s in [size_img_scaledX, size_img_scaledY]:
        if s[0] <= size[0] and s[1] <= size[1] and s[0] > scale[0]:
            scale = s

    img_rgb = img_rgb * 255
    img_rgb[img_rgb > 255] = 255
    factor_y = img_rgb.shape[0] / img_rgb.shape[1] 
    img_rgb = cv2.resize(img_rgb, dsize=scale, interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.imencode(".png", img_rgb)[1].tobytes()
    return img_rgb
