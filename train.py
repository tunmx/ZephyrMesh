from backbone import get_network
from easydict import EasyDict
from data import ZephyrMeshDataset, FMeshAugmentation
from torch.utils.data import DataLoader
from trainer.task import TrainTask
import os

if __name__ == '__main__':
    dataset_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/"
    train_txt_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/train.txt"
    val_txt_path = "/Users/tunm/datasets/ballFaceDataset20230317-ZMESH/val.txt"
    save_dir = "./workplace"

    train_batch_size = 256
    val_batch_size = 64
    worker_num = 8
    epoch_num = 300
    os.makedirs(save_dir, exist_ok=True)

    aug = FMeshAugmentation(image_size=256)
    train_dataset = ZephyrMeshDataset(dataset_path, train_txt_path, mode="train", transform=aug, is_show=False)
    val_dataset = ZephyrMeshDataset(dataset_path, val_txt_path, mode="val", transform=aug, is_show=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=worker_num, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True,
                                num_workers=worker_num, pin_memory=True)

    schedule_opt = dict(
        name="ReduceLROnPlateau",
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    optimizer_opt = dict(
        name="Adam",
        lr=0.001,
    )

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

    print(net)

    task = TrainTask(net, save_dir, optimizer_opt, schedule_opt, )

    task.fitting(train_data=train_dataloader, val_data=val_dataloader, epoch_num=epoch_num, is_save=True)