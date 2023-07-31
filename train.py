import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import logging
import sys
import time
import importlib
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR

    
from utils import save_img
import numpy as np
from source.models.get_model import get_model
from source.losses import get_criterion
from source.optimizer import get_optimizer
from source.scheduler import get_scheduler
from source.datas.utils import create_datasets
import copy
from torch.cuda import amp

import warnings
warnings.filterwarnings("ignore")

#Wandb and mlflow
import wandb

try:
    ## Add wandb key
    anonymous = None
except:
    anonymous = "must"
    print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

parser = argparse.ArgumentParser(description='Simple Super Resolution')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/x2/repConv_x2_m4c32_relu_div2k_warmup_lr5e-4_b8_p384_normalize.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
parser.add_argument('--gpu_ids', type=int, default=1, help = 'gpu_ids')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    
    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(gpu_ids_str))
        device = torch.device('cuda:{}'.format(gpu_ids_str))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    if args.wandb:
        run_log_wandb = wandb.init(entity="iilab", project='Realt_Time_Super_Resolution',
                    config={k:v for k, v in dict(opt).items() if '__' not in k},
                    anonymous=anonymous,
                    name=f"{args.model}|ps-{args.patch_size}|m-{args.m_plainsr}|c-{args.c_plainsr}|{args.loss}|{args.optimizer}|lr{str(args.lr)}|e{str(args.epochs)}",
                    # name=f"{args.model}_{args.scale}",
                    group=args.comment,
                    )
    else: 
        run_log_wandb = 0

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    model = get_model(args, device)

    ## definition of loss and optimizer & scheduler
    loss_func = get_criterion(args, device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    ## load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    
    if args.wandb:
        wandb.watch(model, criterion=loss_func, log='all')
        
    ## resume training
    start_epoch = 1
    if args.resume is not None:
        ckpt_files = os.path.join(args.resume, 'models', "model_x3_last.pt")
        if len(ckpt_files) != 0:
            #ckpt_files = sorted(ckpt_files, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            ckpt = torch.load(ckpt_files)
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('select {}, resume training from epoch {}.'.format(ckpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}_x{}_p{}_m{}_c{}_{}_{}_{}_lr{}_e{}_t{}'.format(args.model, args.scale, args.patch_size, args.m_plainsr, args.c_plainsr, args.act_type, args.loss, args.optimizer, args.lr, args.epochs, timestamp)
            # experiment_name = '{}_x{}'.format(args.model, args.scale)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = utils.get_stat_dict()
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)

    ## print architecture of model
    time.sleep(3) # sleep 3 seconds 
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    scaler = amp.GradScaler()
    max_norm = 0.1
    
    ## start training
    timer_start = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        epoch_loss = 0.0
        stat_dict['epochs'] = epoch
        model = model.train()
        #opt_lr = scheduler.get_last_lr()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Train {}:'.format(args.model))
        #print('##===========training, Epoch: {}, lr: {} =============##'.format(epoch, opt_lr))
        #for iter, batch in enumerate(train_dataloader):
        mem = torch.cuda.memory_reserved(device=args.gpu_ids) / 1E9 if torch.cuda.is_available() else 0
        try:
            current_lr = scheduler.get_lr()[0]
        except:
            current_lr = scheduler.get_last_lr()[0]
                
        for step, (data) in pbar:
            if args.mixed_pred:
                lr, hr = data
                lr, hr = lr.to(device), hr.to(device)
                ###Use Unscaled Gradiendts instead of 
                ### https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
                with amp.autocast(enabled=True):
                    sr = model(lr)
                    #_pred = torch.clamp(_pred, 0, 1)
                    loss = loss_func(sr, hr)   
                scaler.scale(loss).backward()
                
                # Unscales the gradients of optimizer's assigned params i100n-place
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad() # set_to_none=True here can modestly improve performance
            else: 
                optimizer.zero_grad()
                lr, hr = data
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                loss = loss_func(sr, hr)
                loss.backward()
                optimizer.step()
            epoch_loss += float(loss)

            pbar.set_postfix(epoch=f'{epoch}',
                train_loss=f'{epoch_loss/(step+1):0.4f}',
                lr=f'{current_lr:0.5f}', 
                gpu_mem=f'{mem:0.2f} GB')
        
        if args.wandb:
            # Log the metrics
            wandb.log({"train/Loss": epoch_loss/len(train_dataloader),  
                "train/LR":current_lr})

        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()
            for valid_dataloader in valid_dataloaders:
                avg_psnr, avg_ssim = 0.0, 0.0
                name = valid_dataloader['name']
                loader = valid_dataloader['dataloader']
                count = 0
                
                pbar = tqdm(loader, total=len(loader), desc='Valid: {}'.format(args.model))
                
                #for lr, hr in tqdm(loader, ncols=80):
                for lr, hr in pbar:
                    lr, hr = lr.to(device), hr.to(device)
                    sr = model(lr)
  
                    if args.normalize:
                        hr = hr.clamp(0, 1) * 255
                        sr = sr.clamp(0, 1) * 255
                    else: 
                        hr = hr.clamp(0, 255)
                        sr = sr.clamp(0, 255)

                    # calculate psnr and ssim
                    psnr = utils.calc_psnr(sr, hr)       
                    ssim = utils.calc_ssim(sr, hr)         
                    avg_psnr += psnr
                    avg_ssim += ssim
                    
                    count += 1
                    if args.save_val_image and count < 20:
                        fname = str(count + 801).zfill(4) + '.jpg'
                        save_img(os.path.join(experiment_model_path,'./result_img/', str(epoch)+'_rec', fname), sr[0].cpu().numpy().transpose(1,2,0).astype(np.uint8), color_domain='rgb')
                    pbar.set_postfix(epoch=f'{epoch}', psnr=f'{psnr:0.2f}')
                    
                avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
                avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
                stat_dict[name]['psnrs'].append(avg_psnr)
                stat_dict[name]['ssims'].append(avg_ssim)
                if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                    stat_dict[name]['best_psnr']['value'] = avg_psnr
                    stat_dict[name]['best_psnr']['epoch'] = epoch
                    
                    saved_model_path = os.path.join(experiment_model_path, 'model_x{}_best.pt'.format(args.scale))
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'stat_dict': stat_dict
                    }, saved_model_path)
                    
                    saved_model_path = os.path.join(experiment_model_path, 'model_x{}_best_submission.pt'.format(args.scale))
                    
                    torch.save(model.state_dict(), saved_model_path)
                    
                    saved_model_path = os.path.join(experiment_model_path, 'model_x{}_best_submission_deploy.pt'.format(args.scale))
                    
                    model_deploy = copy.deepcopy(model)
                    model_deploy.fuse_model()
                    torch.save(model_deploy.state_dict(), saved_model_path)
                    
                    if args.wandb:
                        wandb.summary["Best PSNR"]      = avg_psnr
                        wandb.summary["Best SSIM"]      = avg_ssim
                        wandb.summary["Best Epoch"]     = epoch
                if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                    stat_dict[name]['best_ssim']['value'] = avg_ssim
                    stat_dict[name]['best_ssim']['epoch'] = epoch
                test_log += '[{}-X{}], PSNR/SSIM: {:.2f}/{:.4f} (Best: {:.2f}/{:.4f}, Epoch: {}/{})\n'.format(
                    name, args.scale, float(avg_psnr), float(avg_ssim), 
                    stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
                    stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])
            
            if args.wandb:
                # Log the metrics
                wandb.log({
                    "val/Valid PSNR {} - ".format(name): avg_psnr,
                    "val/Valid SSIM {} - ".format(name): avg_ssim,
                    })
            
            # print log & flush out
            print(test_log)
            sys.stdout.flush()
            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_x{}_last.pt'.format(args.scale))
            # torch.save(model.state_dict(), saved_model_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)
            
            
            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        scheduler.step()
    