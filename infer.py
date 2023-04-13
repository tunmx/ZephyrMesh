import os.path

import cv2
import torch
import tqdm

from backbone import get_network
from easydict import EasyDict
from data import its
from data import ZephyrMeshDataset, FMeshAugmentation
from torch.utils.data import DataLoader
import torch.nn.functional as F

dir_path = "/Users/tunm/datasets/ffhq10k_zmesh_dataset"
txt_path = os.path.sep.join([dir_path, "val.txt"])
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
net.load_state_dict(torch.load("workplace/s2/best_model.pth", map_location=device))
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

print('MAE: ', F.l1_loss(output, kps_tensor))

predict_images = its.visual_images(images_tensor, output, w, h, swap=False)
gt_images = its.visual_images(images_tensor, kps_tensor, w, h, swap=False, color=(230, 100, 20))
for idx, img in enumerate(predict_images):
    print(img.shape)
    cv2.imshow("predict_images", img)
    gt = gt_images[idx]
    cv2.imshow("gt_images", gt)
    cv2.waitKey(0)
