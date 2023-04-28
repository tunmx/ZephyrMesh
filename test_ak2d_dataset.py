import tqdm
from data import its
from data.dataset.ak2d_dataset import AK2DFolder, AK2DMeshDataset
from torch.utils.data import DataLoader
import cv2

ak2d_folder = AK2DFolder("/Users/tunm/datasets/arkit_train/emotion")

dataset = AK2DMeshDataset([ak2d_folder], mode='val')
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True,
                        num_workers=0)

transform_data = tqdm.tqdm(dataloader)
images_tensor, kps_tensor = next(iter(transform_data))

_, _, h, w = images_tensor.shape
draw_images = its.visual_images(images_tensor, kps_tensor, w, h, swap=False)
for img in draw_images:
    print(img.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)
