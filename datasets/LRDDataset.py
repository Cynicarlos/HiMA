import cv2
import exifread
import numpy as np
import os
import re
import rawpy
import torch
from torch.utils import data
import sys
sys.path.append('/data/models/Carlos/HiMA/')
sys.path.append('E:\Deep Learning\HiMA\HiMA')

from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class LRDDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train',
                transpose=False, h_flip=False, v_flip=False, ratio=False, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()#./LRD/Clean/00001_00_0.25s.DNG ./LRD/Noisy/00001_01_0.00625s.DNG ISO1600 -1EV
                gt_path, input_path, iso, _ = img_pair.split(' ')
                gt_path, input_path = gt_path[6:], input_path[6:]
                ISO = int(''.join([x for x in iso if x.isdigit()]))
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.00625
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 0.25
                ratio = (100 * gt_exposure) / (ISO * input_exposure)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio),
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./LRD/Clean/00001_00_0.25s.DNG
        gt_path = info['gt_path']#./LRD/Clean/00001_00_0.25s.DNG
        
        input_raw = rawpy.imread(os.path.join(self.data_dir, input_path))
        input_raw = self.pack_raw_LRD(input_raw) * info['ratio'] #(4,h/2,w/2)
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0)

        gt_raw = rawpy.imread(os.path.join(self.data_dir, gt_path))
        gt_rgb = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
        gt_rgb = np.float32(gt_rgb) / np.float32(65535)
        
        gt_raw = self.pack_raw_LRD(gt_raw) #(4,h/2,w/2)

        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
                gt_rgb = np.flip(gt_rgb, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
                gt_rgb = np.flip(gt_rgb, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1))
                gt_rgb = np.transpose(gt_rgb, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)
        gt_rgb = np.ascontiguousarray(gt_rgb)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        gt_rgb = torch.from_numpy(gt_rgb).float()

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def pack_raw_LRD(self, raw):
        # pack Bayer image to 4 channels
        im = raw.raw_image_visible.astype(np.float32)

        white_point = 65535.0

        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((im[0:H:2, 0:W:2],  # RGGB
                        im[0:H:2, 1:W:2],
                        im[1:H:2, 0:W:2],
                        im[1:H:2, 1:W:2]), axis=0).astype(np.float32)
        black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)
        out = (out - black_level) / (white_point - black_level)
        out = np.clip(out, 0, 1)
        return out

    def crop_random_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, 2*yy:2*(yy + patch_size), 2*xx:2*(xx + patch_size)]

        return input_raw, gt_raw, gt_rgb

if __name__=='__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    dataset = LRDDataset(data_dir='E:\Deep Learning\datasets\RAW\LRD\LRD',
                         image_list_file='LRD_test_list.txt',
                         patch_size=None,
                         split='test',
                         ratio=False)
    data = dataset[1]
    input_raw, gt_raw, gt_rgb, input_path, gt_path = data['input_raw'],data['gt_raw'],data['gt_rgb'], data['input_path'], data['gt_path']
    print(input_path, gt_path)
    print(input_raw.shape, gt_raw.shape, gt_rgb.shape)
    print(input_raw.min(), input_raw.max(), gt_raw.min(), gt_raw.max(), gt_rgb.min(), gt_rgb.max())
    exit(0)
