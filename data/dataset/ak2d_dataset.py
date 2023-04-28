from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset
from data.transform import FMeshAugmentation
import os
from loguru import logger
from data.fmesh.image_tools import get_aligned_lmk, data_normalization, find_image_file
from typing import List


def read_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    res = list()
    root = os.path.dirname(filename)
    for x in data:
        image_path, mesh_path, kps_path = x.strip().split("\t")
        res.append(dict(image_path=os.path.sep.join([root, image_path]),
                        mesh_path=os.path.sep.join([root, mesh_path]),
                        kps_path=os.path.sep.join([root, kps_path])))

    return res


class AK2DFolder(object):

    def __init__(self, ak2d_dataset_path: str):
        """Input a folder object for an AK2D dataset"""
        self.train_txt = os.path.sep.join([ak2d_dataset_path, 'train.txt'])
        self.test_txt = os.path.sep.join([ak2d_dataset_path, 'test.txt'])
        self.val_txt = os.path.sep.join([ak2d_dataset_path, 'val.txt'])
        self.train_list = read_from_file(self.train_txt)
        self.test_list = read_from_file(self.test_txt)
        self.val_list = read_from_file(self.val_txt)

    def get_train_data(self):
        return self.train_list

    def get_val_data(self):
        return self.val_list

    def get_test_data(self):
        return self.test_list

    def get_data_from_mode(self, mode: str) -> List[dict]:
        if mode == 'train':
            return self.train_list
        elif mode == 'val':
            return self.val_list
        elif mode == 'test':
            return self.test_list
        else:
            return NotImplemented


class AK2DMeshDataset(Dataset):
    def __init__(self, ak2d_folder_list: List[AK2DFolder], mode: str = 'train', transform=None, is_show=False):
        self.data_list = list()
        for fd in ak2d_folder_list:
            self.data_list += fd.get_data_from_mode(mode)
        self.mode = mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = FMeshAugmentation()
        self.is_show = is_show

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        mate = self.data_list[idx]
        kp5 = np.load(mate['kps_path'])
        mesh = np.load(mate['mesh_path'])
        raw_image = cv2.imdecode(np.fromfile(mate['image_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
        image, mesh_points, _ = self.transform(raw_image, mesh, kp5, mode=self.mode)
        data, label = image, mesh_points
        if not self.is_show:
            data, label = data_normalization(image, mesh_points)

        return data, label

