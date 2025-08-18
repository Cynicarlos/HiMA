'''
import torch
import rawpy
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from utils.utils import pack_raw, getInputMetaInfo

#Notice: The ELD dataset is only for evaluation


class ELDDataset(Dataset):
    def __init__(self, basedir, camera_suffix, scenes=None, img_ids=None,input_channel = 4):
        super(ELDDataset, self).__init__()
        self.basedir = basedir #'datasets/ELD'
        self.camera_suffix = camera_suffix # ('SonyA7S2', '.ARW')
        self.scenes = scenes #(1,2,3,4,5,6,7,8,9,10)
        self.img_ids = img_ids #[4, 9, 14] and [5, 10, 15]
        self.input_channel = input_channel
    def __len__(self):
        return 30
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix #'SonyA7S2', '.ARW'
        
        scene_id = i // len(self.img_ids) # i // 3 = 0, 1,2,.....9
        img_id = i % len(self.img_ids) # i % 3 = 0, 1,2

        scene = 'scene-{}'.format(self.scenes[scene_id])#scene-1 to scene-10

        datadir = self.basedir + '/' + camera + '/' + scene #'datasets/ELD/SonyA7S2/scene-1'
        input_path = datadir + '/IMG_{:04d}{}'.format(self.img_ids[img_id], suffix)
        #like 'datasets/ELD/SonyA7S2/scene-1/IMG_0004.ARW'

        gt_ids = [6, 11, 16]#
        #ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        gt_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[img_id], suffix))
        #'datasets/ELD/SonyA7S2/scene-1/IMG_0006.ARW'

        metainfo = getInputMetaInfo(gt_path,input_path)
        ISO = metainfo['ISO']
        Exposure = metainfo['Exposure']
        Ratio = metainfo['Ratio']
        BrightnessValue = metainfo['BrightnessValue']
        FocalLength = metainfo['FocalLength']
        with rawpy.imread(input_path) as raw:
            if self.input_channel == 4:
                input = pack_raw(raw) * Ratio #(4,h/2,w/2)          
            elif self.input_channel == 1:
                input = raw.raw_image_visible.astype(np.float32)
                black_level = np.mean(raw.black_level_per_channel).astype(np.float32)
                input = (input - black_level) / (16383 - black_level)
                input = np.clip(input, 0, 1) * Ratio #(1,h,w)
        with rawpy.imread(gt_path) as raw:
            gt = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True,output_bps=16)
            gt = (gt / 65535.0).transpose(2,0,1)#(3,h,w) numpy

        input = np.maximum(np.minimum(input, 1.0), 0) 
        gt = np.maximum(np.minimum(gt.astype(np.float32), 1.0), 0)
        input = np.ascontiguousarray(input)#按行存储
        gt = np.ascontiguousarray(gt)  
        data = {'input': input, 'gt': gt, 'input_path':input_path, 'gt_path': gt_path,
                'metainfo':{'ISO':(ISO-np.float32(800))/np.float32(3200-800),
                            'Exposure': (Exposure-np.float32(0.0005))/np.float32(0.005-0.001),
                            'Ratio':(Ratio-np.float32(100))/np.float32(200),
                            'BrightnessValue':(BrightnessValue+np.float32(2553)/np.float32(1280))/(np.float32(-783)/np.float32(1280)+np.float32(2553)/np.float32(1280)),
                            'FocalLength':(FocalLength-np.float32(35))/np.float32(47-35)}}
        
        return data


if __name__ == '__main__':
    databasedir = 'E:\Deep Learning\datasets\ELD\ELD'
    scenes = list(range(1, 11))#(1,2,3,4,5,6,7,8,9,10)
    #cameras = ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']     
    #suffixes = ['.CR2', '.CR2', '.nef', '.ARW']
    cameras = ['SonyA7S2']     
    suffixes = ['.ARW']
    img_ids_set = [[4, 9, 14], [5, 10, 15]]# ratio 100 and 200
    # for scene in scenes:

    for img_ids in img_ids_set:# ratio 100 and 200
        #4 datasets, each camera has one dataset
        eval_datasets = [ELDDataset(databasedir, camera_suffix, scenes=scenes, img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]
        #4 dataloaders
        eval_dataloaders = [torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False) for eval_dataset in eval_datasets]
        psnrs = []
        ssims = []
        for camera, dataloader in zip(cameras, eval_dataloaders):
            print('Eval camera {}'.format(camera))
            for data in dataloader:
                matainfo = data['metainfo']
                a = matainfo['BrightnessValue']
                b = matainfo['FocalLength']
                print(a,b)
                if a < 0:
                    print(data['input_path'])
                    exit(0)
            # res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=True, crop=False, savedir='res-eld/{}_scene_{}'.format(camera, scene))
            
            # we evaluate PSNR/SSIM on full size images
            #res = engine.eval(dataloader, dataset_name='eld_eval_{}'.format(camera), correct=True, crop=False)
            #psnrs.append(res['PSNR'])
            #ssims.append(res['SSIM'])
'''