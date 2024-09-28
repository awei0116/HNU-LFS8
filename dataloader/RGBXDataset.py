import os
from pickletools import uint8
import cv2
import torch
import torch.nn as nn
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._lf_path = setting['lf_root']
        self._lf_format = setting['lf_format']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self.lf_image_names = setting['lf_image_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        lf_path = os.path.join(self._lf_path, item_name)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)

        # Check the following settings if necessary

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        lf_images = self._concat_lf(lf_path, mode=cv2.COLOR_BGR2RGB)

        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
           gt, x, lf_images = self.preprocess(gt, x, lf_images)

        if self._split_name == 'train':
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
            lf_images = {key: torch.from_numpy(np.ascontiguousarray(img)).float() for key, img in lf_images.items()}

        output_dict = dict(label=gt, modal_x=x, lf_images = lf_images, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    def _concat_lf(self, filepath, mode=cv2.IMREAD_COLOR):
        image_filenames = self.lf_image_names
        lf_images = {}  # 使用一个字典来存储图像

        for filename in image_filenames:
            filepath1 = os.path.join(filepath, filename)
            if os.path.isfile(filepath1):
                img = RGBXDataset._open_image(filepath1, mode=mode)
                # 从文件名提取键名（如 '1_1' 从 '1_1.png'）
                key = filename.split('.')[0]
                lf_images[key] = img
            else:
                raise FileNotFoundError(f"Image file path {filepath1} does not exist.")

        return lf_images

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img


    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
