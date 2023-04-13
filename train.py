from backbone import get_network
from easydict import EasyDict
from data import ZephyrMeshDataset, FMeshAugmentation
from torch.utils.data import DataLoader
from trainer.task import TrainTask
import os

if __name__ == '__main__':
    dataset_path = "dataset/"
    train_txt_path = "dataset/train.txt"
    val_txt_path = "dataset/val.txt"
    save_dir = "./workplace/s2"

    train_batch_size = 128
    val_batch_size = 64
    worker_num = 8
    epoch_num = 1000
    os.makedirs(save_dir, exist_ok=True)

    aug = FMeshAugmentation(image_size=256)
    train_dataset = ZephyrMeshDataset(dataset_path, train_txt_path, mode="train", transform=aug, is_show=False)
    val_dataset = ZephyrMeshDataset(dataset_path, val_txt_path, mode="val", transform=aug, is_show=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=worker_num, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=True,
                                num_workers=worker_num, pin_memory=True)

    epoch_steps = len(train_dataset) // train_batch_size
    total_steps = epoch_steps * epoch_num
    max_warmup_step = 4000

    schedule_opt = dict(
        name="ReduceLROnPlateau",
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=0.0001
    )

    optimizer_opt = dict(
        name="SGD",
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )
    # schedule_opt = dict(
    #     name="PolyScheduler",
    #     base_lr=optimizer_opt['lr'],
    #     max_steps=total_steps,
    #     warmup_steps=max_warmup_step,
    # )

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

    wandb_opt = dict(
        team_name="tunm",
        project_name="ZMesh",
        experiment_name="exp",
        scenario_name="training",
        folder="log",
        key="4b49a6b0286dcfb718a12360108a7a8578c3582c"
    )

    task = TrainTask(net, save_dir, optimizer_opt, schedule_opt, wandb_cfg=wandb_opt, )

    task.fitting(train_data=train_dataloader, val_data=val_dataloader, epoch_num=epoch_num, is_save=True, )
