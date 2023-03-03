import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss
        
class VGG_gram(nn.Module):
    def __init__(self):
        super(VGG_gram, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False
    
    def gram_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n*c, h*w)
        gram = torch.mm(x,x.t()) # 행렬간 곱셈 수행
        return gram


    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        vgg_output = self.gram_matrix(vgg_output)

        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())
            vgg_gt = self.gram_matrix(vgg_gt)
            
        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss


def get_criterion(cfg, device):
    if cfg.loss == 'l1':
        return nn.L1Loss().cuda(device)
    elif cfg.loss == 'l2':
        return nn.MSELoss()
    else: 
        raise NameError('Choose proper model name!!!')

if __name__ == "__main__":
    # true = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # pred = np.array([1.0, 1.2, 1.1, 1.4, 1.5, 1.8, 1.9])
    # loss = get_criterion(pred, true)
    # print(loss)
    loss = nn.L1Loss()
    
    predict = torch.tensor([1.0, 2, 3, 4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1.0, 1, 1, 1], dtype=torch.float64,  requires_grad=True)
    mask = torch.tensor([0, 0, 0, 1], dtype=torch.float64, requires_grad=True)
    out = loss(predict, target, mask)
    out.backward()
    print(out)