import os
import re
import torch
import rawpy
import numpy as np
from torch.utils.data import Dataset

#Notice: The ELD dataset is only for evaluation
from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class ELDDataset(Dataset):
    def __init__(self, data_dir, camera, pairs_file_path, patch_size=None, split='train',
                transpose=False, h_flip=False, v_flip=False, **kwargs):
        super(ELDDataset, self).__init__()
        assert camera in ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
        self.data_dir = data_dir
        self.camera = camera
        self.split = split
        self.patch_size = patch_size
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.pairs_file_path=os.path.join(data_dir, pairs_file_path)

        self.img_info=[]
        with open(self.pairs_file_path, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()
                input_path, gt_path, ratio = img_pair.split(' ')
                input_path = os.path.join(data_dir, self.camera, input_path)
                gt_path = os.path.join(data_dir, self.camera, gt_path)
                _id = os.path.basename(input_path)#10003_00_10s.ARW
                _id, extension = os.path.splitext(_id)#10003_00_10s    .ARW
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
                    'ratio': np.float32(ratio),
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']
        gt_path = info['gt_path']
        
        with rawpy.imread(input_path) as raw:
            input_raw = self.pack_raw_bayer(raw)
        with rawpy.imread(gt_path) as raw:
            gt_raw = self.pack_raw_bayer(raw)
            gt_rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_rgb = gt_rgb.transpose(2, 0, 1)#未归一 (3,2848,4256) numpy
            gt_rgb = np.float32(gt_rgb) / np.float32(65535)

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
                input_raw_patch, gt_raw_patch, gt_rgb_patch = self.crop_random_patch(input_raw, gt_raw, gt_rgb, self.patch_size)
                input_raw = input_raw_patch.copy()
                gt_raw = gt_raw_patch.copy()
                gt_rgb = gt_rgb_patch.copy()

        input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)
        gt_rgb = np.ascontiguousarray(gt_rgb)

        input_raw = torch.from_numpy(input_raw)
        gt_raw = torch.from_numpy(gt_raw)
        gt_rgb = torch.from_numpy(gt_rgb)


        data = {
            'input_raw': input_raw, 
            'gt_raw': gt_raw,
            'gt_rgb': gt_rgb,
            'input_path':input_path, 
            'gt_path': gt_path
        }
        
        return data

    def __len__(self):
        return len(self.img_info)
    
    def crop_random_patch(self, input_raw, gt_raw, gt_rgb, patch_size):
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_rgb = gt_rgb[:, yy*2:(yy + patch_size)*2 , xx*2:(xx + patch_size)*2]

        return input_raw, gt_raw, gt_rgb
    
    def pack_raw_bayer(self, raw):
        im = raw.raw_image_visible.astype(np.float32)
        
        raw_pattern = raw.raw_pattern
        R = np.where(raw_pattern==0)
        G1 = np.where(raw_pattern==1)
        B = np.where(raw_pattern==2)
        G2 = np.where(raw_pattern==3)
        
        white_point = 16383
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                        im[G1[0][0]:H:2,G1[1][0]:W:2],
                        im[B[0][0]:H:2,B[1][0]:W:2],
                        im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

        black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)
        out = (out - black_level) / (white_point - black_level)
        out = np.clip(out, 0, 1)
    
        return out

