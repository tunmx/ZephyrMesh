from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
from torch.utils.data import Dataset
from data.transform import FMeshAugmentation
import os


class FMeshDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, data_dir, mode='train', transform=None, is_show=False):
        self.data_dir = data_dir
        self.mode = mode
        if transform:
            self.transform = transform
        else:
            self.transform = FMeshAugmentation()
        self.is_show = is_show


    @abstractmethod
    def _load_data(self, path):
        candidate = [os.path.isdir(os.path.join(path, item)) for item in os.listdir(path) if
                     os.path.isdir(os.path.join(path, item))]
        selected = list()
        for idx, sub_dir in enumerate(candidate):
