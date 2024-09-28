import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

def random_mirror(gt, modal_x, lf_images):
    if random.random() >= 0.5:
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)
        lf_images = {key: cv2.flip(img, 1) for key, img in lf_images.items()}

    return gt, modal_x, lf_images


def random_scale(gt, modal_x, lf_images, scales):

    scale = random.choice(scales)
    # sh = int(lf_images[0].shape[0] * scale)
    # sw = int(lf_images[0].shape[1] * scale)
    sh = int(next(iter(lf_images.values())).shape[0] * scale)
    sw = int(next(iter(lf_images.values())).shape[1] * scale)

    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    lf_images = {key: cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR) for key, img in lf_images.items()}

    return gt, modal_x, lf_images, scale

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config

    def __call__(self, gt, modal_x, lf_images):
        gt, modal_x, lf_images = random_mirror(gt, modal_x, lf_images)
        if self.config.train_scale_array is not None:
            gt, modal_x, lf_images, scale = random_scale(gt, modal_x, lf_images, self.config.train_scale_array)

        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)
        lf_images = {key: normalize(img, self.norm_mean, self.norm_std) for key, img in lf_images.items()}

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(next(iter(lf_images.values())).shape[:2], crop_size)

        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        p_lf_images = {key: random_crop_pad_to_shape(img, crop_pos, crop_size, 0)[0] for key, img in lf_images.items()}

        p_modal_x = p_modal_x.transpose(2, 0, 1)
        p_lf_images = {key: img.transpose(2, 0, 1) for key, img in p_lf_images.items()}

        return p_gt, p_modal_x, p_lf_images

class ValPre(object):
    def __call__(self, gt, modal_x, lf_images):
        return gt, modal_x, lf_images

def get_train_loader(engine, dataset, config):
    data_setting = {'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'lf_root': config.lf_root_folder,
                    'lf_format': config.lf_format,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'lf_image_names': config.lf_image_names}
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config)

    train_dataset = dataset(data_setting, "train", train_preprocess, config.batch_size * config.niters_per_epoch)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler