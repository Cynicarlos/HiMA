import argparse
import os
import torch
from datasets.ELDDataset import ELDDataset
from tqdm import tqdm
import yaml
from models import build_model
from tqdm import tqdm
from utils import set_random_seed
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch


def crop_tow_patch(x):
    _, _, H, W = x.shape
    res = [x[:, :, :, :W//2], x[:, :, :, W//2:]]
    return res


@torch.no_grad()
def test(model, dataloader, camera, ratio, merge_test=False):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    total_samples = len(dataloader.dataset)
    tqdm_loader = tqdm(dataloader, desc=f"Evaluating Camara {camera}:", leave=False)
    
    with open(f'results/ELD/{camera}_{ratio}.txt', 'w') as f:
        for idx, data in enumerate(tqdm_loader):
            input = data['input_raw'].cuda()
            gt = data['gt_rgb'].cuda()
            #print(input.shape, gt.shape)
            input_path = data['input_path'][0]
            gt_path = data['gt_path'][0]
            if merge_test:
                inputs = crop_tow_patch(input)
                gts = crop_tow_patch(gt)
                preds = [model(patch)[0] for patch in inputs]
                preds = [torch.clamp(pred, 0, 1) for pred in preds]
                psnrs = [get_psnr_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                ssims = [get_ssim_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                psnr = sum(psnrs) / len(psnrs)
                ssim = sum(ssims) / len(ssims)
            else:
                pred, _ = model(input)
                pred = torch.clamp(pred, 0, 1)
                psnr = get_psnr_torch(pred, gt, data_range=1.0)
                ssim = get_ssim_torch(pred, gt, data_range=1.0)
                pred = torch.clamp(pred, 0, 1)
                psnr = get_psnr_torch(pred, gt, data_range=1.0)
                ssim = get_ssim_torch(pred, gt, data_range=1.0)

            f.write(f"input:{input_path}    gt:{gt_path}    psnr:{psnr.item():.4f}   ssim:{ssim.item():.4f}\n")

            psnr_sum += psnr.item()
            ssim_sum += ssim.item()
            
            tqdm_loader.set_postfix({'psnr':f'{psnr.item():.4f}','ssim':f'{ssim.item():.4f}', 'avg_psnr': f'{psnr_sum/(idx+1):.4f}', 'avg_ssim':f'{ssim_sum/(idx+1):.4f}'}, refresh=True)
        
        average_psnr = psnr_sum / total_samples
        average_ssim = ssim_sum / total_samples

        f.write(f'psnr:{average_psnr:.4f}       ssim:{average_ssim:.4f}')
    print(f'Camera {camera}    Ratio {ratio}    PSNR:{average_psnr:.4f}    SSIM:{average_ssim:.4f}')
    return psnr_sum, ssim_sum, total_samples

if __name__ == "__main__":
    os.makedirs('results/ELD', exist_ok=True)
    data_dir='/data/dataset/Carlos/ELD' #30
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_test', action='store_true', default=False)
    args = parser.parse_args()
    
    with open('configs/sony.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    model_name, model = build_model(config['model'])
    model = model.cuda()
    checkpoint = torch.load('Sony.pth') #30
    model.load_state_dict(checkpoint['model'])

    
    #cameras = ['SonyA7S2', 'NikonD850', 'CanonEOS70D', 'CanonEOS700D']
    cameras = ['SonyA7S2', 'NikonD850']
    
    #ratios = [1, 10, 100, 200]
    ratios = [100, 200]
    for camera, num_patch in zip(cameras, num_patches):
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0
        for ratio in ratios:
            pairs_file_path = os.path.join(data_dir, f'{camera}_{ratio}.txt')
            dataset = ELDDataset(data_dir=data_dir, camera=camera, pairs_file_path=pairs_file_path, split='test')
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
            psnr, ssim, samples = test(model, dataloader, camera=camera, ratio=ratio, 
                                    merge_test=args.merge_test)
            total_psnr += psnr
            total_ssim += ssim
            total_samples += samples
        all_ratio_avg_psnr, all_ratio_avg_ssim = total_psnr / total_samples, total_ssim / total_samples
        print(f'total samples:{total_samples}    all_ratio_psnr:{all_ratio_avg_psnr}    all_ratio_ssim:{all_ratio_avg_ssim}')

