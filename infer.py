import cv2
import numpy as np
import torch
import tqdm

from backbone import get_network
from easydict import EasyDict
from data import its
from data import ZephyrMeshDataset, FMeshAugmentation
from torch.utils.data import DataLoader

txt_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/val.txt"
dir_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/"
batch_size = 64

cfg = dict(
        use_onenetwork=True,
        width_mult=1.0,
        num_verts=1787,
        input_size=256,
        task=3,
        network='resnet_jmlr',
        no_gap=False,
        use_arcface=False,
    )
config = EasyDict(cfg)
net = get_network(config)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load("workplace/best_model.pth", map_location=device))
net.eval()

aug = FMeshAugmentation(image_size=256)
dataset = ZephyrMeshDataset(dir_path, txt_path, mode="val", transform=aug, is_show=False)

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)

transform_data = tqdm.tqdm(dataloader)
images_tensor, kps_tensor = next(iter(transform_data))
_, _, h, w = images_tensor.shape

output = net(images_tensor)
print(images_tensor.shape)

draw_images = its.visual_images(images_tensor, output, w, h, swap=False)
for img in draw_images:
    print(img.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)
