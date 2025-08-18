import argparse
import os
import shutil
import time
import torch
import yaml

from datasets import build_train_loader, build_valid_loader, build_test_loader
from models import build_model
from timm.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import format_time, get_grad_norm, load_checkpoint, save_checkpoint, set_random_seed
from utils.logger import CustomLogger
from utils.loss import build_loss
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler

from models.LiMamba import get_local_mean_and_std

def main(config, args):
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(os.path.join(config['output'], 'tb_logs'))
    train_dataloader = build_train_loader(config['data'])

    valid_dataloader = build_test_loader(config['data'], 8)

    logger.info(f"Creating model:{config['name']}/{config['model']['name']}")
    model_name, model = build_model(config['model'])
    model.cuda()

    optimizer = build_optimizer(config['train'], model)
    lr_scheduler = build_scheduler(config['train'], optimizer)
    critierion = build_loss(config['loss'])

    metrics = {
        'psnr': 0.0,
        'max_psnr': 0.0,
        'ssim': 0.0,
        'max_ssim': 0.0,
        'lpips': 1.0e7,
        'min_lpips': 1.0e7,
    }

    total_epochs = config['train']['epochs']

    if args.auto_resume:
        auto_resume_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pth')
        if os.path.exists(auto_resume_path):
            metrics['psnr'], metrics['max_psnr'] = load_checkpoint(config, auto_resume_path, model, optimizer, lr_scheduler, logger)
            #validate(config, model, critierion, valid_dataloader, config['train'].get('start_epoch', 0), writer)
        else:
            raise ValueError(f"Auto resume failed, no checkpoint found at {auto_resume_path}")
    elif args.resume:
        metrics['psnr'], metrics['max_psnr'] = load_checkpoint(config, args.resume, model, optimizer, lr_scheduler, logger)
        #validate(config, model, critierion, valid_dataloader, config['train'].get('start_epoch', 0), writer)


    logger.info("----------------------------------------Start training----------------------------------------")

    for epoch in range(config['train'].get('start_epoch', 0)+1, total_epochs+1):
        epoch_start = time.time()
        train_one_epoch(config, model, critierion, train_dataloader, optimizer, epoch, lr_scheduler, writer)

        if epoch == 1 or epoch % 10 == 0 or (epoch >= 200 and epoch % 5 == 0) or epoch >= 250:
            metrics['psnr'] , metrics['ssim'], metrics['lpips'], valid_time = validate(config, model, valid_dataloader, epoch, writer)
            metrics['max_psnr'] = max(metrics['max_psnr'], metrics['psnr'])
            metrics['max_ssim'] = max(metrics['max_ssim'], metrics['ssim'])
            metrics['min_lpips'] = min(metrics['min_lpips'], metrics['lpips'])

            save_checkpoint(config, epoch, model, metrics, optimizer, lr_scheduler, is_best=(metrics['psnr'] == metrics['max_psnr']))

            logger.info(
                f"Valid: [{epoch}/{config['train']['epochs']}]\
                PSNR:{metrics['psnr']:.2f} Max_PSNR:{metrics['max_psnr']:.2f}\
                SSIM:{metrics['ssim']:.4f}  Max_SSIM:{metrics['max_ssim']:.4f}\
                LPIPS:{metrics['lpips']:.4f}  Min_LPIPS:{metrics['min_lpips']:.4f}\
                valid_time: {valid_time}"
            )
            writer.add_scalar('eval/max_psnr', metrics['max_psnr'], epoch)
            writer.add_scalar('eval/max_ssim', metrics['max_ssim'], epoch)
            writer.add_scalar('eval/min_lpips', metrics['min_lpips'], epoch)

        epoch_end = time.time()
        epoch_time = format_time(epoch_end - epoch_start)
        logger.info(f'Epoch {epoch} time: {epoch_time}')

def train_one_epoch(config, model, critierion, data_loader, optimizer, epoch, lr_scheduler, writer):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    losses_meter = [AverageMeter() for _ in range(len(critierion))]

    tqdm_loader = tqdm(data_loader,desc=f"Train: [{epoch}/{config['train']['epochs']}]",leave=False)
    for idx, data in enumerate(tqdm_loader):
        input_raw = data['input_raw'].cuda(non_blocking=True)
        gt_rgb = data['gt_rgb'].cuda(non_blocking=True)
        if config['with_raw_loss']:
            gt_raw = data['gt_raw'].cuda(non_blocking=True)
            if config['with_local_mean_and_std_loss']:
                gt_raw_mean, gt_raw_std = get_local_mean_and_std(gt_raw, 16, 16)
                #gt_rgb_mean, gt_rgb_std = get_local_mean_and_std(gt_rgb, 32, 32)

        if config['with_metadata']:
            metadata = data['input_metadata'].cuda(non_blocking=True)
            pred = model(input_raw, metadata)
        else:
            pred = model(input_raw)
        if config['with_raw_loss']:
            if config['with_local_mean_and_std_loss']:
                #losses = critierion(pred, [gt_rgb, gt_raw, gt_raw_mean, gt_raw_std, gt_rgb_mean, gt_rgb_std])
                losses = critierion(pred, [gt_rgb, gt_raw, gt_raw_mean, gt_raw_std])
            else:
                losses = critierion(pred, [gt_rgb, gt_raw])
        else:
            losses = critierion(pred, gt_rgb)

        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()

        if config['train'].get('clip_grad'):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()

        batch_size = config['data']['train']['batch_size']
        for _loss_meter, _loss in zip(losses_meter, losses):
            _loss_meter.update(_loss.item(), batch_size)    
        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        if config['with_raw_loss']:
            if config['with_local_mean_and_std_loss']:
                tqdm_loader.set_postfix(
                    {
                        'al_rgb':f'{losses_meter[0].avg:.6f}',
                        'al_raw':f'{losses_meter[1].avg:.6f}',
                        'al_mean1':f'{losses_meter[2].avg:.5f}',
                        'al_std1':f'{losses_meter[3].avg:.5f}',
                        #'al_mean2':f'{losses_meter[4].avg:.5f}',
                        #'al_std2':f'{losses_meter[5].avg:.5f}',
                        'al_t':f'{loss_meter.avg:.5f}',
                        'lr':f"{optimizer.param_groups[0]['lr']:.6f}"
                    },
                    refresh=True
                )
            else:
                tqdm_loader.set_postfix(
                    {
                        'al_rgb':f'{losses_meter[0].avg:.6f}',
                        'al_raw':f'{losses_meter[1].avg:.6f}',
                        'al_t':f'{loss_meter.avg:.6f}',
                        'lr':f"{optimizer.param_groups[0]['lr']:.6f}"
                    },
                    refresh=True
                )
        else:
            tqdm_loader.set_postfix(
                {
                    'al':f'{loss_meter.avg:.6f}', 
                    'lr':f"{optimizer.param_groups[0]['lr']:.6f}"
                },
                refresh=True
            )

    lr_scheduler.step(epoch)
    tensor_board_dict = {'train/loss_total':loss_meter.avg}
    for index, (_loss_meter) in enumerate(losses_meter):
        tensor_board_dict[f'train/loss_{index}'] = _loss_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)

    end_time = time.time()
    train_time = format_time(end_time-start_time)

    if config['with_raw_loss']:
        if config['with_local_mean_and_std_loss']:
            logger.info(
                f'Train: [{epoch}/{config["train"]["epochs"]}]\
                al_rgb:{losses_meter[0].avg:.6f} al_raw:{losses_meter[1].avg:.6f}\
                al_mean1:{losses_meter[2].avg:.6f} al_std1:{losses_meter[3].avg:.6f}\
                al_total: {loss_meter.avg:.6f} lr: {optimizer.param_groups[0]["lr"]:.6f} train_time: {train_time}'
            )
        else:
            logger.info(
                f'Train: [{epoch}/{config["train"]["epochs"]}]\
                al_rgb:{losses_meter[0].avg:.6f} al_raw:{losses_meter[1].avg:.6f}\
                al_total: {loss_meter.avg:.6f} lr: {optimizer.param_groups[0]["lr"]:.6f} train_time: {train_time}'
            )
    else:
        logger.info(f'Train: [{epoch}/{config["train"]["epochs"]}] total loss: {loss_meter.avg:.6f} lr: {optimizer.param_groups[0]["lr"]:.6f} train_time: {train_time}')


@torch.no_grad()
def validate(config, model, data_loader, epoch, writer):
    start_time = time.time()
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    lpips_meter = AverageMeter()
    tqdm_loader = tqdm(data_loader,desc=f"Valid: [{epoch}/{config['train']['epochs']}]",leave=False)
    for idx, data in enumerate(tqdm_loader):
        input_path, gt_path = data['input_path'][0], data['gt_path'][0]
        input_raw = data['input_raw'].cuda(non_blocking=True)
        gt_rgb = data['gt_rgb'].cuda(non_blocking=True)
        
        if config['with_metadata']:
            input_metadata = data['input_metadata'].cuda(non_blocking=True)
            pred = model(input_raw, input_metadata)
        else:
            pred = model(input_raw)
        if config['with_raw_loss']:
            pred = pred[0]

        pred = torch.clamp(pred, 0, 1)
        gt_rgb = torch.clamp(gt_rgb, 0, 1)
        lpips = get_lpips_torch(pred, gt_rgb)
        
        pred =  pred * 255
        gt_rgb = gt_rgb * 255
        pred = pred.round()
        gt_rgb = gt_rgb.round()

        psnr = get_psnr_torch(pred, gt_rgb, data_range=255.0)
        ssim = get_ssim_torch(pred, gt_rgb, data_range=255.0)

        batch_size = config['data']['test']['batch_size']
        
        psnr_meter.update(psnr.item(), batch_size)
        ssim_meter.update(ssim.item(), batch_size)
        lpips_meter.update(lpips.item(), batch_size)

        tqdm_loader.set_postfix(
            {
            'psnr':f'{psnr_meter.val:.2f}', 'a_psnr': f'{psnr_meter.avg:.2f}',
            'ssim':f'{ssim_meter.val:.4f}', 'a_ssim': f'{ssim_meter.avg:.4f}',
            'lpips':f'{lpips_meter.val:.3f}', 'a_lpips': f'{lpips_meter.avg:.3f}'
            },
            refresh=True)
        os.makedirs(f"{config['output']}/results", exist_ok=True)
        with open(f"{config['output']}/results/Epoch_{epoch}_results.txt", 'a') as f:
            f.write(f"input:{input_path}    gt:{gt_path}    psnr:{psnr_meter.val:.2f}   ssim:{ssim_meter.val:.4f} lpips:{lpips_meter.val:.4f}\n")

    tensor_board_dict = {}
    tensor_board_dict['eval/psnr'] = psnr_meter.avg
    tensor_board_dict['eval/ssim'] = ssim_meter.avg
    tensor_board_dict['eval/lpips'] = lpips_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)
    
    end_time = time.time()
    valid_time = format_time(end_time-start_time)
    return psnr_meter.avg, ssim_meter.avg, lpips_meter.avg, valid_time

@torch.no_grad()
def validate_metric(pred, gt):
    pred = torch.clamp(pred, 0, 1) 
    gt = torch.clamp(gt, 0, 1)
    psnrs = get_psnr_torch(pred, gt, data_range=1.0)
    ssims = get_ssim_torch(pred, gt, data_range=1.0)
    return psnrs.mean(), ssims.mean()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sony.yaml',type=str, help='Path to option YAML file.')
    parser.add_argument('--auto-resume', action='store_true', default=False, help='Auto resume from latest checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    
    config['output'] = os.path.join(config.get('output', 'runs'), config['name'])
    os.makedirs(config['output'], exist_ok=True)
    os.makedirs(os.path.join(config['output'],'checkpoints'), exist_ok=True)

    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = CustomLogger(log_file_path=f"{config['output']}/training_log.txt")
    path = os.path.join(config['output'], f"{start_time}.yaml")
    shutil.copy(args.config, path)

    model_name = config['model']['name']
    shutil.copy(f"models/{model_name}.py", config['output'])
    current_cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
    logger.info(f"Current CUDA Device: {current_cuda_device.name}, Total Mem: {int(current_cuda_device.total_memory / 1024 / 1024)}MB")
    main(config, args)


