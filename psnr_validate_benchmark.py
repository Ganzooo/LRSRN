import torch
import pickle
import numpy as np
import math
import cv2
import os
import os.path as osp
from shutil import copyfile
import glob
import argparse
import source.utils.dataset as dd
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import parse
import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from skimage.metrics import structural_similarity as ssim_calc
from ssim import ssim_matlab

from source.models.get_model import get_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# def evaluate_benchmark_fp32(best_model_path, save_path, hrPath,lrPath, datasetsName):
def evaluate_benchmark_fp32(args, model, hrPath,lrPath, datasetsName):

    outpath = args.outpath
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    total_txt = open(os.path.join(outpath, 'total.txt'), 'w')
    print(args.scale)
    
    for i, (datasetName) in enumerate(datasetsName):

        clip_psnr_list = []
        clip_ssim_list = []

        clip_psnr_txt = open(os.path.join(outpath, '{}_psnr.txt'.format(datasetsName[i])), 'w')
        clip_ssim_txt = open(os.path.join(outpath, '{}_ssim.txt'.format(datasetsName[i])), 'w')

        dataset = dd.Eval_Dataset(lr_images_dir=lrPath[datasetName], hr_images_dir=hrPath[datasetName], n_channels=3, transform=None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

        for img_L, lr_img_path, img_H, hr_img_path in tqdm(dataloader, desc='{}'.format(datasetsName[i])):

            img_L = img_L.to(device)
            img_H = img_H.to(device)
            sr = model(img_L)
            
            sr_ = sr[0]
            hr_ = img_H[0]

            sr_ = (np.round((sr_ * 255).detach().cpu().numpy())).astype('uint8').transpose(1, 2, 0)
            hr_ = (np.round((hr_ * 255).detach().cpu().numpy())).astype('uint8').transpose(1, 2, 0)

            # psnr = 20 * math.log10(((sr_ - hr_) * (sr_ - hr_)).mean())
            # ssim = ssim_matlab(sr, img_H).detach().cpu().numpy()
            psnr = psnr_calc(sr_, hr_)
            ssim = ssim_calc(sr_, hr_, channel_axis = -1, multichannel=True)
            clip_psnr_txt.write("psnr : {} \n".format(psnr))
            clip_psnr_list.append(psnr)

            clip_ssim_txt.write("ssim : {} \n".format(ssim))
            clip_ssim_list.append(ssim)

        clip_psnr_txt.write("avg psnr : {} \n".format(np.mean(clip_psnr_list)))
        clip_ssim_txt.write("avg ssim : {} \n".format(np.mean(clip_ssim_list)))
        total_txt.write("{}PSNR: {}\t SSIM: {}\n".format(datasetsName[i], np.mean(clip_psnr_list), np.mean(clip_ssim_list)))


    return psnr

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./configs/cvprw2023/ablation/plainsr_x2_m4c32_192_relu.yml')
    parser.add_argument('--weight', type=str, default='./weight/plainsr_x2_p192_m4_c32_relu_l1_adam_lr0.0005_e800_t2023-0324-1345/models/model_x2_best.pt')
    parser.add_argument('--outpath', type=str, default='./benchmark')
    parser.add_argument('--gpu_ids', type=int, default=0)
    parser.add_argument('--scale', type=int, default=2)

    args = parser.parse_args()
    if args.config:
        opt = vars(args)
        yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(yaml_args)

    gpu_ids_str = str(args.gpu_ids)

    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print("use cuda & cudnn for acceleration!")
        print("the gpu id is: {}".format(gpu_ids_str))
        device = torch.device('cuda:{}'.format(gpu_ids_str))
        torch.backends.cudnn.benchmark = True
    else:
        print("use cpu for training!")
        device = torch.device('cpu')

    
    model = get_model(args, device)

    if args.weight is not None:
        print('load pretrained model: {}!'.format(args.weight))
        ckpt = torch.load(args.weight, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    dataset_evaluation_path_HR = {
                                "Set5":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Set5/X{}_compressed/HR_crop/".format(args.scale),
                                "Set14":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Set14/X{}_compressed/HR_crop/".format(args.scale),
                                "B100":"/dataset/SR/RLSR/benchmark_compressed/benchmark/B100/X{}_compressed/HR_crop/".format(args.scale),
                                "Urban100":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Urban100/X{}_compressed/HR_crop/".format(args.scale),
                                "DIV2K":"/dataset/SR/RLSR/benchmark_compressed/benchmark/DIV2K_valid/X{}_compressed/HR_crop/".format(args.scale),
                                "NTIRE2023":"/dataset/SR/RLSR/benchmark_compressed/benchmark/NTIRE2023/X{}_compressed/HR_crop/".format(args.scale),
                                }
    dataset_evaluation_path_LR = {
                                "Set5":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Set5/X{}_compressed/LR/".format(args.scale),
                                "Set14":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Set14/X{}_compressed/LR/".format(args.scale),
                                "B100":"/dataset/SR/RLSR/benchmark_compressed/benchmark/B100/X{}_compressed/LR/".format(args.scale),
                                "Urban100":"/dataset/SR/RLSR/benchmark_compressed/benchmark/Urban100/X{}_compressed/LR/".format(args.scale),
                                "DIV2K":"/dataset/SR/RLSR/benchmark_compressed/benchmark/DIV2K_valid/X{}_compressed/LR/".format(args.scale),
                                "NTIRE2023":"/dataset/SR/RLSR/benchmark_compressed/benchmark/NTIRE2023/X{}_compressed/LR/".format(args.scale),
                                }
    

    
    datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'DIV2K', 'NTIRE2023']

    psnrs_fp32 = evaluate_benchmark_fp32(args, model, dataset_evaluation_path_HR, dataset_evaluation_path_LR, datasets)

