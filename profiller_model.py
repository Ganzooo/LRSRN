import torch 
import os 
import numpy 
from source.models.get_model import get_model
from source.models.rtsrn import RealTimeSRNet
from collections import OrderedDict
from tqdm import tqdm

import argparse, yaml

from torchsummary import summary as summary_
import torch.autograd.profiler as profiler

parser = argparse.ArgumentParser(description='Simple Super Resolution')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/repConv/repConv_x3_m4c48_relu_combined2.yml', help = 'pre-config file for training')
parser.add_argument('--gpu_ids', type=int, default=0, help = 'gpu_ids')

def profile_model(cfg):
    device = torch.device('cuda:{}'.format(0))
    warmup = True
    
    x = torch.rand(1,3,720, 1280).cuda(0)
    
    model_baseline = RealTimeSRNet(num_channels=3, num_feats=64, num_blocks=4, upscale=3).cuda()
    model = get_model(cfg, device)
    
    ###Model -> RepConv model
    model.fuse_model()   #Inference mode
    
    #summary_(model,(3,128,128),batch_size=10)  
    model.eval()
    model_baseline.eval()
    if warmup == True:
        for _ in range(224):
            _ = model(x)
            _ = model_baseline(x)
    
    # with profiler.profile(with_stack=False, use_cuda=True, profile_memory=False, record_shapes=False) as prof:
    #     _ = model(x)
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", top_level_events_only=True))
    
    with profiler.profile(use_kineto=False, with_stack=False, use_cuda=True, profile_memory=True, record_shapes=True) as prof2:
        _ = model_baseline(x)
    print("::::::::::Baseline model:::::::::::::::::::::")
    
    print(prof2.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", top_level_events_only=True))
    print("Baseline model profiler result: 19.77\n\n")
    
    with profiler.profile(use_kineto=False, with_stack=False, use_cuda=True, profile_memory=True, record_shapes=True) as prof2:
        _ = model(x)
    print("::::::::::Proposed model:::::::::::::::::::::")
    print(prof2.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", top_level_events_only=True))
    


def inference_test_model(cfg, device):
    device = torch.device('cuda:{}'.format(0))
    
    x = torch.rand(1,3,720, 1280).cuda(device=device)
    
    #model_baseline = RealTimeSRNet(num_channels=3, num_feats=64, num_blocks=4, upscale=3).cuda(device=device)
    model = get_model(cfg, device)
    
    ###Model -> RepConv model
    model.fuse_model()   #Inference mode
    
    model.eval()
    #model_baseline.eval()
    
    print(model)
    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    half_inference = True
    if half_inference:
        x = x.half()
        model = model.half()
        #model_baseline = model_baseline.half()
        
    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(x) 
            
    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(244)):       
            start.record()
            _ = model(x)
            end.record()

            torch.cuda.synchronize()
              
            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        
        if half_inference:
            print('------> Average runtime of FP16({}) is : {:.6f} ms'.format('Base test', ave_runtime))
        else: 
            print('------> Average runtime of FP32({}) is : {:.6f} ms'.format('Base test', ave_runtime))

if __name__ == "__main__":
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
        
    #profile_model()
    inference_test_model(args, device)