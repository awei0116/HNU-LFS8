import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device, config):

        label = data['label']
        modal_x = data['modal_x']
        lf = data['lf_images']

        name = data['fn']

        pred = self.sliding_eval_rgbX(modal_x, lf, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            # Real_colors
            # class_colors = [[168, 198, 168], [198, 0, 0], [202, 154, 198], [0, 0, 0], [100, 198, 198], [198, 100, 0], [52, 42, 198],
            #                 [154, 52, 192], [198, 0, 168], [0, 198, 0], [198, 186, 90], [108, 107, 161], [156, 200, 26], [158, 179, 202]]

            #Syn_colors
            class_colors = [ [0, 0, 0],[168, 198, 168], [198, 0, 0], [202, 154, 198], [100, 198, 198], [198, 100, 0],[52, 42, 198],
                            [154, 52, 192], [198, 0, 168], [0, 198, 0], [198, 186, 90], [108, 107, 161], [156, 200, 26],[158, 179, 202]]
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors()
            # colors = [[168, 198, 168], [198, 0, 0], [202, 154, 198], [0, 0, 0], [100, 198, 198], [198, 100, 0], [52, 42, 198],
            #           [154, 52, 192], [198, 0, 168], [0, 198, 0], [198, 186, 90], [108, 107, 161], [156, 200, 26], [158, 179, 202]]
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results, config):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        return result_line, mean_IoU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='UrbanLF_Syn', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    
    dataset_name = args.dataset_name
    if dataset_name == 'UrbanLF_Real':
        from configs.config_UrbanLF_Real import config
    elif dataset_name == 'UrbanLF_Syn':
        from configs.config_UrbanLF_Syn import config
    else:
        raise ValueError('Not a valid dataset name')

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'lf_root': config.lf_root_folder,
                    'lf_format': config.lf_format,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names,
                    'lf_image_names': config.lf_image_names}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image, config)
        _, mean_IoU = segmentor.run_eval(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)