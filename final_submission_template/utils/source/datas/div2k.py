import os
import glob
import random
import pickle

import numpy as np
import imageio
import torch
import torch.utils.data as data
import skimage.color as sc
import time
from utils import ndarray2tensor

def crop_patch(lr, hr, patch_size, scale, augment=True):
    # crop patch randomly
    lr_h, lr_w, _ = lr.shape
    hp = patch_size
    lp = patch_size // scale
    lx, ly = random.randrange(0, lr_w - lp + 1), random.randrange(0, lr_h - lp + 1)
    hx, hy = lx * scale, ly * scale
    lr_patch, hr_patch = lr[ly:ly+lp, lx:lx+lp, :], hr[hy:hy+hp, hx:hx+hp, :]
    # augment data
    if augment:
        hflip = random.random() > 0.5
        vflip = random.random() > 0.5
        rot90 = random.random() > 0.5
        if hflip: lr_patch, hr_patch = lr_patch[:, ::-1, :], hr_patch[:, ::-1, :]
        if vflip: lr_patch, hr_patch = lr_patch[::-1, :, :], hr_patch[::-1, :, :]
        if rot90: lr_patch, hr_patch = lr_patch.transpose(1,0,2), hr_patch.transpose(1,0,2)
        # numpy to tensor
    lr_patch, hr_patch = ndarray2tensor(lr_patch), ndarray2tensor(hr_patch)
    return lr_patch, hr_patch

class DIV2K(data.Dataset):
    def __init__(
        self, HR_folder, LR_folder, CACHE_folder, 
        train=True, augment=True, scale=2, colors=1, 
        patch_size=96, repeat=168, normalize=True
    ):
        super(DIV2K, self).__init__()
        self.HR_folder = HR_folder
        self.LR_folder = LR_folder
        self.augment   = augment
        self.img_postfix = '.png'
        self.scale = scale
        self.colors = colors
        self.patch_size = patch_size
        self.repeat = repeat
        self.nums_trainset = 0
        self.train = train
        self.cache_dir = CACHE_folder
        self.normalize = normalize

        ## for raw png images
        self.hr_filenames = []
        self.lr_filenames = []
        ## for numpy array data
        self.hr_npy_names = []
        self.lr_npy_names = []
        ## store in ram
        self.hr_images = []
        self.lr_images = []

        self.lr_filenames = sorted(glob.glob(self.LR_folder + '*.jpg'))

        for idx, lr_name in enumerate(self.lr_filenames):
            _name = os.path.basename(lr_name)[:-4] + '.png'
            self.hr_filenames.append(os.path.join(self.HR_folder, _name))
            # if _name[1] == 'pr':
            #     base_name = _name[4] + '_' + _name[5] + '_' + _name[6] + '_' + _name[7]
            # else: 
            #     base_name = _name[3] + '_' + _name[4] + '_' + _name[5] + '_' + _name[6]
            #self.imageOrgName.append(os.path.join(os.path.join(self.org_path,self.image_org_folder),base_name))
            
        assert len(self.hr_filenames) == len(self.lr_filenames)
          
        self.nums_trainset = len(self.hr_filenames)

        LEN = self.nums_trainset
        hr_dir = os.path.join(self.cache_dir, 'hr', 'ycbcr' if self.colors==1 else 'rgb')
        lr_dir = os.path.join(self.cache_dir, 'lr_x{}'.format(self.scale), 'ycbcr' if self.colors==1 else 'rgb')
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)
        else:
            for i in range(LEN):
                hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
                hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                self.hr_npy_names.append(hr_npy_name)
            
        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)
        else:
            for i in range(LEN):
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)

        ## prepare hr images
        if len(glob.glob(os.path.join(hr_dir, "*.npy"))) != len(self.hr_filenames):
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("convert {} hr images to npy data!".format(i+1))
                hr_image = imageio.imread(self.hr_filenames[i], pilmode="RGB")
                if self.colors == 1:
                    hr_image = sc.rgb2ycbcr(hr_image)[:, :, 0:1]
                hr_npy_name = self.hr_filenames[i].split('/')[-1].replace('.png', '.npy')
                hr_npy_name = os.path.join(hr_dir, hr_npy_name)
                self.hr_npy_names.append(hr_npy_name)
                np.save(hr_npy_name, hr_image)
        else:
            print("hr npy datas have already been prepared!, hr: {}".format(len(self.hr_npy_names)))
        ## prepare lr images
        if len(glob.glob(os.path.join(lr_dir, "*.npy"))) != len(self.lr_filenames):
            for i in range(LEN):
                if (i+1) % 50 == 0:
                    print("convert {} lr images to npy data!".format(i+1))
                lr_image = imageio.imread(self.lr_filenames[i], pilmode="RGB")
                if self.colors == 1:
                    lr_image = sc.rgb2ycbcr(lr_image)[:, :, 0:1]
                lr_npy_name = self.lr_filenames[i].split('/')[-1].replace('.jpg', '.npy')
                lr_npy_name = os.path.join(lr_dir, lr_npy_name)
                self.lr_npy_names.append(lr_npy_name)
                np.save(lr_npy_name, lr_image)
        else:
            print("lr npy datas have already been prepared!, lr: {}".format(len(self.lr_npy_names)))

    def __len__(self):
        if self.train:
            return self.nums_trainset * self.repeat
        else:
            return self.nums_trainset

    def __getitem__(self, idx):
        # get periodic index
        idx = idx % self.nums_trainset
        # get whole image
        hr, lr = np.load(self.hr_npy_names[idx]), np.load(self.lr_npy_names[idx])
        if self.train:
            train_lr_patch, train_hr_patch = crop_patch(lr, hr, self.patch_size, self.scale, True)
            if self.normalize:
                return train_lr_patch / 255., train_hr_patch / 255.
            else: 
                return train_lr_patch, train_hr_patch
                
        if self.normalize:
            return ndarray2tensor(lr / 255.), ndarray2tensor(hr / 255.)
        else: 
            return ndarray2tensor(lr), ndarray2tensor(hr)

if __name__ == '__main__':
    HR_folder = '/dataset/SR/RLSR/DIV2K/train_HR'
    LR_folder = '/dataset/SR/RLSR/DIV2K/train_LR'
    argment   = True
    div2k = DIV2K(HR_folder, LR_folder, augment=True, scale=2, colors=3, patch_size=96, repeat=168, store_in_ram=True)

    print("numner of sample: {}".format(len(div2k)))
    start = time.time()
    for idx in range(10):
        tlr, thr, vlr, vhr = div2k[idx]
        print(tlr.shape, thr.shape, vlr.shape, vhr.shape)
    end = time.time()
    print(end - start)