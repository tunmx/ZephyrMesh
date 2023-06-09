import copy
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch.utils.data import DataLoader
from loguru import logger
import wandb
import socket
import torch.nn.functional as F
from data.fmesh.image_tools import visual_images

SHOW_IMAGE_NUM = 6

class TrainTask(object):
    def __init__(self, model, save_dir, optimizer_option, lr_schedule_option, loss_func=F.l1_loss, wandb_cfg=None, weight_path=None):
        self.save_dir = save_dir
        self.model = model
        self.task_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer_option = optimizer_option
        self.lr_schedule_option = lr_schedule_option
        self.optimizer, self.scheduler = self._configure_optimizers(model, optimizer_option=optimizer_option,
                                                                    lr_schedule_option=lr_schedule_option)
        self.loss_func = loss_func
        self.best_accuracy = 0.0
        self.best_loss = 1000.0
        self.time_tag = datetime.datetime.now()
        self.upload = True

        # if weight_path:
        #     self._load_pretraining_model(weight_path)
        self.model = self.model.to(self.task_device)

        self._wandb_log_init_(wandb_cfg)

    def fitting(self, train_data: DataLoader, val_data: DataLoader, epoch_num: int, is_save: bool = True):
        logger.info("Start training.")
        logger.info(f"Training Epochs Total: {epoch_num}")
        logger.info(f"Training Result Save to {self.save_dir}")
        for epoch in range(epoch_num):
            # train set
            train_loss = self.train_one_epoch(train_data, epoch, epoch_num)
            wandb.log({'train_loss': train_loss}, step=epoch + 1)
            # val set
            self.upload = True
            val_loss = self.validation_one_epoch(val_data, epoch)
            wandb.log({'val_loss': val_loss}, step=epoch + 1)
            logger.info(f"Train Epoch[{epoch + 1}/{epoch_num}] val_loss: {val_loss}")
            if is_save:
                self.save_model(val_loss, epoch, mode='min')

            # self.writer.add_scalar('loss/train', train_loss, epoch)
            # self.writer.add_scalar('loss/val', val_loss, epoch)
            # self.writer.add_scalar('lr/epoch', self.lr_scheduler.get_last_lr()[0], epoch)

        logger.info("This training is completed, a total of {} rounds of training, training time: {} minutes" \
                    .format(epoch_num, (datetime.datetime.now() - self.time_tag).seconds // 60))

    @staticmethod
    def _configure_optimizers(model, optimizer_option, lr_schedule_option) -> tuple:
        optimizer_cfg = copy.deepcopy(optimizer_option)
        logger.info("loading optimizer {}".format(optimizer_cfg.get('name')))
        name = optimizer_cfg.pop('name')
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=model.parameters(), **optimizer_cfg)


        schedule_cfg = copy.deepcopy(lr_schedule_option)
        name = schedule_cfg.pop('name')
        if name in ['PolyScheduler', ]:
            if name == "PolyScheduler":
                from .lr_scheduler import PolyScheduler
                scheduler = PolyScheduler(optimizer, **schedule_cfg)
        else:
            build_scheduler = getattr(torch.optim.lr_scheduler, name)
            scheduler = build_scheduler(
                optimizer=optimizer, **schedule_cfg)

        return optimizer, scheduler


    def save_model(self, loss, epoch, mode='min'):
        torch.save(self.model.state_dict(), os.path.join(
            self.save_dir, 'last.pth'))
        if mode == 'min':
            # Minimum save according to loss rate
            if loss < self.best_loss:
                # torch.save(self.model.state_dict(), os.path.join(
                #     self.save_dir, 'model_%d_loss%0.3f.pth' % (epoch + 1, loss)))
                with open(os.path.join(self.save_dir, "best_epoch.txt"), 'w') as f:
                    f.write(f"{epoch}: {loss}\n")
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_dir, 'best_model.pth'))
                self.best_loss = loss

    def train_one_epoch(self, train_data, epoch, epochs_total):
        global loss
        self.model.train()
        # train_acc = 0.0
        train_bar = tqdm(train_data)
        for step, data in enumerate(train_bar):
            samples, labels = data
            samples = samples.to(self.task_device)
            labels = labels.to(self.task_device)
            self.optimizer.zero_grad()
            # print('shape: ', samples.shape)
            outputs = self.model(samples.to(self.task_device))
            # print("debug: outputs", outputs.shape)
            # print("outputs: labels", labels.shape)
            loss = self.loss_func(outputs, labels.to(self.task_device))
            loss.backward()
            self.optimizer.step()

            train_bar.set_description('Epoch: [{}/{}] loss: {:.6f}'.format(epoch + 1, epochs_total, loss))

        if self.lr_schedule_option['name'] in ["ReduceLROnPlateau", ]:
            # callable
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

        return loss.item()

    def validation_one_epoch(self, val_data, epoch) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            logger.info(f"Learning Rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            val_bar = tqdm(val_data)
            for step, data in enumerate(val_bar):
                val_images, val_labels = data
                n, c, h, w = val_images.shape
                # val_images[0] = np.
                outputs = self.model(val_images.to(self.task_device))
                loss = self.loss_func(outputs, val_labels.to(self.task_device))
                val_loss += loss.item()

                val_bar.set_description(
                    'Val: loss: {:.3f}'.format(val_loss / (step + 1)))
                predict_images = visual_images(val_images.cpu()[:SHOW_IMAGE_NUM], outputs.cpu()[:SHOW_IMAGE_NUM], w, h)
                predict_images = np.asarray(predict_images)
                gt_images = visual_images(val_images.cpu()[:SHOW_IMAGE_NUM], val_labels.cpu()[:SHOW_IMAGE_NUM], w, h, color=(230, 100, 20))
                gt_images = np.asarray(gt_images)
                self._upload_images_(predict_images, epoch + 1, images_gt=gt_images)
                self.upload = False

        wandb.log({'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch + 1)
        return val_loss / len(val_bar)


    def _wandb_log_init_(self, wandb_cfg):
        dir_path = os.path.join(self.save_dir, wandb_cfg['folder'])
        os.makedirs(dir_path, exist_ok=True)
        wandb_init_config = dict(team_name=wandb_cfg['team_name'], project_name=wandb_cfg['project_name'],
                                 experiment_name=wandb_cfg['experiment_name'], scenario_name=wandb_cfg['scenario_name'])
        if wandb_cfg:
            wandb.login(key=wandb_cfg['key'])
        wandb.init(config=wandb_init_config,
                   project=wandb_cfg['project_name'],
                   entity=wandb_cfg['team_name'],
                   notes=socket.gethostname(),
                   group="training",
                   dir=dir_path,
                   job_type="training",
                   reinit=True,
                   )

    def _upload_images_(self, images: np.ndarray, step: int, images_gt: np.ndarray = None, ):
        img_list = list()
        gt_list = list()
        if len(images.shape) == 4:
            for img in images:
                img_list.append(wandb.Image(img))
        elif len(images.shape) == 3:
            img_list.append(wandb.Image(images))
        else:
            pass
        if images_gt is not None:
            # show gt images
            if len(images_gt.shape) == 4:
                for img in images_gt:
                    gt_list.append(wandb.Image(img))
            elif len(images_gt.shape) == 3:
                gt_list.append(wandb.Image(images_gt))
            else:
                pass
        if self.upload:
            wandb.log({
                "results": img_list
            }, step=step)

            if gt_list:
                wandb.log({
                    "ground truth": gt_list
                }, step=step)