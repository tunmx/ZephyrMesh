import os

import cv2
import tqdm
from data import its
from data import ZephyrMeshDataset, FMeshAugmentation
from torch.utils.data import DataLoader

txt_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/train.txt"
dir_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/"
batch_size = 64

aug = FMeshAugmentation(image_size=256)
dataset = ZephyrMeshDataset(dir_path, txt_path, mode="val", transform=aug, is_show=False)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)

transform_data = tqdm.tqdm(dataloader)
images_tensor, kps_tensor = next(iter(transform_data))
_, _, h, w = images_tensor.shape
draw_images = its.visual_images(images_tensor, kps_tensor, w, h, swap=False)
for img in draw_images:
    print(img.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)

