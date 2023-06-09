from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset
from data.transform import FMeshAugmentation
import os
from loguru import logger
from data.fmesh.image_tools import get_aligned_lmk, data_normalization, find_image_file

def read_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data

class FMeshDataset(Dataset):
    """原图过于庞大，加载时间会耗时，所以选择放弃使用"""
    def __init__(self, data_dir, data_path, mode='train', transform=None, is_show=False):
        self.data_dir = data_dir
        self.data_path = data_path
        self.mode = mode
        if transform:
            self.transform = transform
        else:
            self.transform = FMeshAugmentation()
        self.is_show = is_show
        self.data_mate = self._load_data(data_dir, data_path)

    def __len__(self):
        return len(self.data_mate)

    @abstractmethod
    def _load_data(self, data_dir, data_path):
        data_map = read_from_file(data_path)
        candidate = [os.path.sep.join([data_dir, item]) for item in data_map if
                     os.path.isdir(os.path.sep.join([data_dir, item]))]
        selected = list()
        logger.info(f"Statistical candidate {self.mode} datasets...")
        for idx, sub_dir in enumerate(tqdm.tqdm(candidate)):
            basename = os.path.basename(sub_dir)
            kps5_path = os.path.sep.join([sub_dir, "kps5.npy"])
            mesh_path = os.path.sep.join([sub_dir, "raw_mesh.npy"])
            trans_matrix_path = os.path.sep.join([sub_dir, "to256_trans_matrix.npy"])
            raw_image_path = os.path.sep.join([sub_dir,  "raw_image"])
            raw_image_path = find_image_file(raw_image_path)
            for path in [kps5_path, mesh_path, trans_matrix_path, raw_image_path]:
                if path is not None:
                    if not os.path.exists(path):
                        logger.warning(f"[Not Found Data]{basename}")
                        continue
            mate = dict(kps5=kps5_path, mesh=mesh_path, trans_matrix=trans_matrix_path, raw_image=raw_image_path)
            selected.append(mate)

        return selected


    def __getitem__(self, idx):
        mate = self.data_mate[idx]
        kp5 = np.load(mate['kps5'])
        mesh = np.load(mate['mesh'])
        trans_matrix = np.load(mate['trans_matrix'])
        raw_image = cv2.imdecode(np.fromfile(mate['raw_image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        warped = cv2.warpAffine(raw_image, trans_matrix, (self.transform.image_size,) * 2, )
        warped_mesh = get_aligned_lmk(mesh, trans_matrix)
        warped_lmk5 = get_aligned_lmk(kp5, trans_matrix)
        image, mesh_points, _ = self.transform(warped, warped_mesh, warped_lmk5, mode=self.mode)
        data, label = image, mesh_points
        if not self.is_show:
            data, label = data_normalization(image, mesh_points)

        return data, label

