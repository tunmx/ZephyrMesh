import random
from imgaug import augmenters as iaa
import imgaug as ia
from .fmesh.image_tools import *

class FMeshAugmentation(object):

    def __init__(self,
                 image_size=256,
                 sometimes_rate=0.5,
                 crop_percent=(0, 0.1),
                 flip_lr=0.0,
                 gaussian_blur=(0, 0.5),
                 multiply=(0.25, 1.55),
                 contrast_normalization=(0.8, 1.2),
                 gamma_contrast=(0.6, 1.5),
                 scale=(0.8, 1.6),
                 translate_percent_x=(-0.15, 0.15),
                 translate_percent_y=(-0.15, 0.15),
                 rotate=(-25, 25),
                 shear=(-25, 25),
                 order=(0, 1),
                 cval=0,
                 mode="constant",
                 ):
        # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强
        sometimes = lambda aug: iaa.Sometimes(sometimes_rate, aug)
        self.image_size = image_size
        self.flip_lr = flip_lr
        self.val_seq = iaa.Sequential([
            iaa.Resize((image_size, image_size), ),
        ])

        self.train_seq = iaa.Sequential([  # 建立一个名为seq的实例，定义增强方法，用于增强
            sometimes(iaa.Crop(percent=tuple(crop_percent))),
            # iaa.Fliplr(flip_lr),  # 对百分之五十的图像进行做左右翻转
            iaa.GaussianBlur(tuple(gaussian_blur)),  # 在模型上使用0均值1方差进行高斯模糊
            # iaa.Rotate((-30, 30)),
            iaa.Multiply(tuple(multiply)),  # 改变亮度, 不影响bounding box
            iaa.ContrastNormalization(tuple(contrast_normalization)),  # 对比度
            iaa.GammaContrast(tuple(gamma_contrast), per_channel=True),  # 随机颜色变换
            # iaa.Sequential([
            #         iaa.Dropout(p=0.005),  # 随机删除像素点
            #     ]),

            # 对一部分图像做仿射变换
            sometimes(iaa.Affine(
                scale=random.uniform(*scale),  # 图像缩放为80%到120%之间
                translate_percent={"x": tuple(translate_percent_x), "y": tuple(translate_percent_y)},  # 平移±20%之间
                rotate=tuple(rotate),  # 旋转±45度之间
                shear=tuple(shear),  # 剪切变换±16度，（矩形变平行四边形）
                order=list(order),  # 使用最邻近差值或者双线性差值
                cval=cval,  # 全白全黑填充
                mode=mode  # 定义填充图像外区域的方法
            )),

            iaa.Resize(image_size, ),
        ])

        self.seq_map = dict(train=self.train_seq, val=self.val_seq, test=self.val_seq)

        self.flipper = iaa.Sequential([
            iaa.Fliplr(1.0)
        ])


    def _transform_one(self, image: np.ndarray, mesh_points: np.ndarray, kps5_points: np.ndarray, is_align=True, mode='train') -> tuple:
        kps = list()
        for kx, ky in mesh_points:
            kps.append(ia.Keypoint(x=kx, y=ky))
        k5 = list()
        for kx, ky in kps5_points:
            k5.append(ia.Keypoint(x=kx, y=ky))
        kpsoi = ia.KeypointsOnImage(kps, shape=image.shape)
        kpsoi5 = ia.KeypointsOnImage(k5, shape=image.shape)
        aug_det = self.seq_map[mode].to_deterministic()
        img_aug = aug_det.augment_image(image)
        kps_aug = aug_det.augment_keypoints(kpsoi)
        kps5_aug = aug_det.augment_keypoints(kpsoi5)
        kps_out = kps_aug.to_xy_array()

        kps5_out = kps5_aug.to_xy_array()

        if mode == 'train':
            # To preserve the original order of the landmark, flip is handled separately
            if random.uniform(0, 1) < self.flip_lr:
                img_aug = self.flipper.augment_image(img_aug)
                width = img_aug.shape[1]
                kps5_out = kps_flip(width, kps5_out)
                kps_out = kps_flip(width, kps_out)

        if is_align:
            tfrom, img_aug = trans_to_align(img_aug, kps5_out, self.image_size)
            kps_out = get_aligned_lmk(kps_out, tfrom)
            kps5_out = get_aligned_lmk(kps5_out, tfrom)

        return img_aug, kps_out, kps5_out


    def __call__(self, image: np.ndarray, mesh_points: np.ndarray, kps5_points: np.ndarray, is_align=True, mode='train', *args, **kwargs) -> tuple:
        return self._transform_one(image, mesh_points, kps5_points, is_align, mode=mode)