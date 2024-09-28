import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'UrbanLF_Real'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'UrbanLF_Real')
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'        #111
C.lf_root_folder = osp.join(C.dataset_path, 'LF')
C.lf_format ='.png'       #111
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Depth')    #111
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
C.lf_is_single_channel = False
C.train_source = osp.join(C.dataset_path, "train.txt")   #111
C.eval_source = osp.join(C.dataset_path, "test.txt")     #111
C.is_test = False                                         #111
C.num_train_imgs = 580                                        #111
C.num_eval_imgs = 80                                        #111
C.num_classes = 14                                        #111
C.class_names = ["bike", "building", "fence", "others", "person", "pole", "road", "sidewalk", "traffic sign", "vegetation", "vehicle", "bridge", "rider", "sky"]
# C.lf_image_names = ['1_1.png', '1_5.png', '1_9.png', '2_2.png', '2_5.png', '2_8.png', '3_3.png', '3_5.png', '3_7.png', '4_4.png',
#                     '4_5.png', '4_6.png', '5_1.png', '5_2.png', '5_3.png', '5_4.png', '5_5.png', '5_6.png', '5_7.png', '5_8.png',
#                     '5_9.png', '6_4.png', '6_5.png', '6_6.png', '7_3.png', '7_5.png', '7_7.png', '8_2.png', '8_5.png', '8_8.png',
#                     '9_1.png', '9_5.png', '9_9.png']
C.lf_image_names = ['5_1.png', '5_2.png', '5_3.png', '5_4.png', '5_5.png', '5_6.png', '5_7.png', '5_8.png', '5_9.png']

"""Image Config"""
C.background = 255
C.image_height = 432
C.image_width = 623
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 3
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size  + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.75, 1, 1.25]                                         #111
C.eval_flip = True                                        #111
C.eval_crop_size = [432, 623] # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 50
C.checkpoint_step = 5

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.abspath('log_final/log_UrbanLF_Real/' + 'log_' + C.dataset_name + '_' + C.backbone + '_' + 'lfcromb_lfconmb_cvssdecoder')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()