import os
import shutil

raw_images_path = "/Users/tunm/datasets/BDLMK-ballFaceDataset20230317"
fitting_dataset_path = "/Users/tunm/datasets/ballFaceDataset20230317-FMESH"

suffix = "jpg"

dirs_list = [os.path.join(fitting_dataset_path, item) for item in os.listdir(fitting_dataset_path) if os.path.isdir(os.path.join(fitting_dataset_path, item))]

for item in dirs_list:
    basename = os.path.basename(item)
    img = os.path.join(raw_images_path, basename + f".{suffix}")
    dst = os.path.join(fitting_dataset_path, basename, "raw_image" + f".{suffix}")
    if not os.path.exists(img):
        print(f"{img} not found")
        continue
    shutil.copy(img, dst)

