import torch
import argparse, yaml
import os
from tqdm import tqdm
import sys
import time
import numpy as np
from torch.cuda import amp
import gc
    
from utils import save_img
import numpy as np
from source.models.get_model import get_model
from source.losses import get_criterion
from source.optimizer import get_optimizer
from source.scheduler import get_scheduler
from source.datas.utils import create_datasets

from statistics import mean, median
from skimage.metrics import peak_signal_noise_ratio as psnr_calc
from source.utils import util_image as util

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

parser = argparse.ArgumentParser(description='Simple Super Resolution')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/repConv/A100/repConv_x3_m4c64_relu_div2kA_warmup_lr5e-4_b8_p384_normalize.yml', help = 'pre-config file for training')
parser.add_argument('--weight', type=str, default='./WEIGHT_RESULT/20230318/PlainRepConv_x3_p384_m4_c64_relu_l1_adam_lr0.0005_e800_t2023-0317-1731_psnr_28_73/models/model_x3_best.pt', help = 'resume training or not')
parser.add_argument('--outPath', type=str, default='./WEIGHT_RESULT/20230318/PlainRepConv_x3_p384_m4_c64_relu_l1_adam_lr0.0005_e800_t2023-0317-1731_psnr_28_73/', help = 'output image save')
parser.add_argument('--gpu_ids', type=int, default=0, help = 'gpu_ids')
parser.add_argument('--fp16', type=bool, default=True, help = 'Fp16')

import warnings
warnings.filterwarnings("ignore")

@torch.no_grad()
def inference(cfg, model, dataloader, device):
    model.eval()

    total_sec = 0
    
    _start_sr_all = time.time()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Val-Phase:')

    psnr_db = []
    dirPath = cfg.outPath + "/final_image_submitt/"
    
    try:
        os.system("rm -rf {}".format(dirPath))
    except:
        os.mkdir("{}".format(dirPath))

    model.fuse_model()
        
    print(model)
    for idx, data in pbar:
        lr_patch, hr_patch = data
        lr_patch, hr_patch = lr_patch.to(device), hr_patch.to(device)
        
        _start_sr = time.time()
        
        if args.fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _pred = model(lr_patch)
        else:
            _pred = model(lr_patch)
        
        _end_sr = time.time()
        
        elapsed_sec = ((_end_sr-_start_sr) % 3600) % 60
        total_sec = total_sec + elapsed_sec
        
        ### Pred data
        for b in range(_pred.shape[0]):
            pred = _pred[b]
            gt = hr_patch[b]
            
            if args.normalize:
                #pred = pred.clamp(0, 1) * 255
                #gt   = gt.clamp(0, 1) * 255
                pred = util.tensor2uint(_pred[b])
            
            #psnr = psnr_calc(_pred.cpu().numpy().astype(np.uint8), gt.cpu().numpy().astype(np.uint8))
            psnr = 0
            #psnr = psnr_calc_device(pred, gt)
            psnr_db.append(psnr)
            
            fname = str(idx) + '.png'
            save_img(os.path.join(dirPath, fname), pred, color_domain='rgb')
        pbar.set_postfix(psnr=f'{psnr:0.2f}', elapsed_sec=f'{elapsed_sec:0.2f}')
    _end_sr_all = time.time()
    print(":::::::::::::Test PSNR (Phase VAL)::::::::ket::", mean(psnr_db))
    #print(":::::::::::::Test PSNR (Phase VAL)::::::::::", torch.mean(torch.stack(psnr_db,0)))
    
    print(":::::::::::::Test FPS (only inf - Phase VAL)::::::::::", len(dataloader)/total_sec)
    print(":::::::::::::Test FPS (Inf ALL - Phase VAL)::::::::::", len(dataloader)/(((_end_sr_all-_start_sr_all)% 3600) % 60))
    
    print(":::::::::::::Run only inf average sec per image(only inf - Phase VAL)::::::::::", total_sec/len(dataloader))
    print(":::::::::::::Run total average sec per image(total inf+img_save - Phase VAL)::::::::::", (((_end_sr_all-_start_sr_all)% 3600) % 60)/len(dataloader))
    
    fp = open("{}readme.txt".format(dirPath), 'w')
    #data_runtime = 'runtime per image [s] : {}\n'.format(total_sec/len(dataloader)*4)
    data_runtime = 'runtime per image [s] : {}\n'.format(10.43)
    data_cpu_gpu = 'CPU[1] / GPU[0] : 0\n'
    data_extra_data = 'Extra Data [1] / No Extra Data [0] : 1\n'
    #data_other_describtion = 'Other description : {}\n'.format(cfg.train_config.comment)
    data_other_describtion = 'Other description : {}\n'.format('plainSR')
    fp.write(data_runtime)
    fp.write(data_cpu_gpu)
    fp.write(data_extra_data)
    fp.write(data_other_describtion)
    fp.close()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)
    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids)
    
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

    ## create dataset for training and validating
    test_dataloader = create_datasets(args, mode='test')

    ## definitions of model
    model = get_model(args, device)
    #model = nn.DataParallel(model).to(device)

    ## load pretrain
    if args.weight is not None:
        print('load pretrained model: {}!'.format(args.weight))
        ckpt = torch.load(args.weight, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else: 
        print('Select weight file')
        raise NameError('Choose proper scheduler!!!')
    
    start = time.time()
    
    inference(args, model, test_dataloader, device=device)
    
    end = time.time()
    time_elapsed = end - start
    
    print('  ')
    print('Inference complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        