
import os
import torch.nn.functional as F
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations

"""
CutBlur
Copyright 2020-present NAVER corp.
MIT license
"""
def im2tensor(im):
    np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_t).float()
    return tensor

def cutblur(im1, im2, prob=1.0, alpha=1.0):

    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha
    h, w = im2.size(2), im2.size(3)
    # h, w = im2.size(0), im2.size(1)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)
    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.target_path = os.path.join(data_path, 'Denoised')
        self.input_path = os.path.join(data_path, 'Noised')
        self.target_filename_list = os.listdir(self.target_path)
        self.input_filename_list = os.listdir(self.input_path)

    def __len__(self):
        return len(self.target_filename_list)

    def __getitem__(self, idx):
        t_path = os.path.join(self.target_path, self.target_filename_list[idx])
        i_path = os.path.join(self.input_path, self.input_filename_list[idx])
        # target
        target = cv2.imread(t_path)
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])
        target_img = target
        
        
        # input
        input_img = cv2.imread(i_path)
        b, g, r = cv2.split(input_img)
        input_img = cv2.merge([r, g, b])
        
        target_img = cv2.resize(target_img, (int(256), int(256)), interpolation=cv2.INTER_CUBIC)
        input_img = cv2.resize(input_img, (int(256), int(256)), interpolation=cv2.INTER_CUBIC)
        
        _, input_img = cutblur(im2tensor(target_img).unsqueeze(0), im2tensor(input_img).unsqueeze(0), alpha=0.5, prob=0.5)
        input_img = torch.squeeze(input_img)

        target_img = np.float32(normalize(target_img))
        input_img = np.float32(normalize(input_img))
        target_img = target_img.transpose(2,0,1)
        # input_img = input_img.transpose(2,0,1)
        return torch.Tensor(input_img), torch.Tensor(target_img)