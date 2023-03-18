import os
import pathlib
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import center_crop

def main(args):
    
    pathlib.Path(os.path.join(args.lr_out_dir)).mkdir(parents=True, exist_ok=True)
    
    if args.gt_out_dir is not None:
        pathlib.Path(os.path.join(args.gt_out_dir)).mkdir(parents=True, exist_ok=True)
    
    count = 0
    for filename in tqdm(os.listdir(args.image_dir)):
        # load image
        img = Image.open(os.path.join(args.image_dir, filename)).convert('RGB')
        img_name, ext = os.path.splitext(filename)
        
        #img_name = 'Midjorney_' + str(count).zfill(5)
        #ext = '.png'
        #count = count + 1
        
        nW, nH = img.size      
        if nW >= 2040 and nH >= 1080:        
            # pre-crop image to DIV2K dimensions
            img = center_crop(img, output_size=[1080, 2040])
        elif nW >= 1920 and nH >= 1080:        
            # pre-crop image to GTA dimensions
            img = center_crop(img, output_size=[1080, 1920])
        elif nW >= 1020 and nH >= 1020:        
            # pre-crop image to LSDIR and ARTBENCH dimensions
            img = center_crop(img, output_size=[1020, 1020])
        elif nW >= 1080 and nH >= 510:        
            # pre-crop image to LSDIR and ARTBENCH dimensions
            img = center_crop(img, output_size=[510, 1080])
        elif nW >= 510 and nH >= 510:        
            # pre-crop image to LSDIR and ARTBENCH dimensions
            img = center_crop(img, output_size=[510, 510])
        
        try:
            # check sizes
            w, h = img.size
            assert w % args.downsample_factor == 0
            assert h % args.downsample_factor == 0
            
            if args.gt_out_dir is not None:
                img.save(os.path.join(args.gt_out_dir, f"{img_name+ext}"))


            
            # bicubic downsampling
            img = img.resize((int(w/args.downsample_factor), int(h/args.downsample_factor)), resample=Image.Resampling.BICUBIC)
            
            if ext == ".jpg":
                img.save(os.path.join(args.lr_out_dir, f"{img_name}.jpg"), "JPEG", quality=100)
            else:
                # save to JPEG
                img.save(os.path.join(args.lr_out_dir, f"{img_name}.jpg"), "JPEG", quality=args.jpeg_level)
        except:
            pass

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default='/dataset/SR/RLSR/Combined_x2/train_HR/')
    parser.add_argument("--gt-out-dir", type=str, default='/dataset/SR/RLSR/Combined_x2/train_HR_cropped/')
    parser.add_argument("--lr-out-dir", type=str, default='/dataset/SR/RLSR/Combined_x2/train_LR/')
    parser.add_argument("--jpeg-level", type=int, default=90)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=[1080, 2040], nargs="+")
    args = parser.parse_args()
    
    main(args)