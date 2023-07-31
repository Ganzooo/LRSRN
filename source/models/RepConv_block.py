import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RepBlock(nn.Module):
    def __init__(self, inp_planes, out_planes, feature_size=256, mode='train', act_type = 'linear', with_idt = True):
        super(RepBlock, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mode = mode
        self.mid_planes = feature_size
        self.act_type = act_type
       
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')
        
        if self.mode == 'train':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=1)
            self.w0 = self.conv0.weight
            self.b0 = self.conv0.bias

            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.w1 = self.conv1.weight
            self.b1 = self.conv1.bias
        else:
            self.repconv = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

    def forward(self, x):
        if self.mode == 'train':
            if self.with_idt:
                y0 = self.conv0(x)
                y0 = self.conv1(y0)
                y0 = y0 + x                    
            else:
                y0 = self.conv1(self.conv0(x))
        else: 
            RepW, RepB = self.repblock_convert()
            y0 = self.repconv(input=x, weight=RepW, bias=RepB)
        
        if self.act_type != 'linear':
            y0 = self.act(y0)    
        return y0
    
    def repblock_convert(self):         
        # get_weights
        weight0 = self.conv0.weight
        bias0 = self.conv0.bias if self.conv0.bias is not None else None
        weight1 = self.conv1.weight
        bias1 = self.conv1.bias if self.conv1.bias is not None else None

        device = weight0.get_device()
        if device < 0:
            device = None
                    #tf# (kx, ky, in, out) -> (out, in, kx, ky)
        #torch (out, in, kx, ky,) -> (out, in, kx, ky)

        # weights: [k, k, in, out]
        # [out, in, kx, ky] -> [out, in, kx*ky]
        weight0_ = weight0.reshape([weight0.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        # [out, in, kx, ky] -> [out, in*kx*ky]
        weight1_ = weight1.reshape([weight1.shape[0], weight1.shape[1]*weight1.shape[2]*weight1.shape[3]])
        # [out, in, kx*ky]
        
            
        init_val = np.zeros([weight1.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        new_weight_ = torch.tensor(init_val, requires_grad=False, dtype=torch.float, device=device)

        for i in range(weight0.shape[1]):
            tmp = weight0_[:, i, :].reshape([weight0.shape[0], weight0.shape[2]*weight0.shape[3]])
            #new_weight_[:, i, :].copy_(torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(tmp, requires_grad=False)))
            new_weight_[:, i, :].copy_(torch.matmul(weight1_, tmp))
        new_weight_ = new_weight_.reshape([weight1.shape[0], weight0.shape[1], weight0.shape[2], weight0.shape[3]])

        #residual
        residual = torch.zeros(self.out_planes, self.inp_planes, 3, 3, requires_grad=False, device=device)
        if self.with_idt:
            for i in range(self.out_planes):
                residual[i, i, 1, 1] = 1.0
        # final weight
        wt_tensor = new_weight_ + residual.cuda(device=device)

        # biases
        if bias0 is not None and bias1 is not None:
            bias0_ = bias0.reshape([bias0.shape[0], 1]) # 1d -> 2d
            bs_tensor = torch.matmul(weight1_, bias0_).reshape([bias1.shape[0]]) + bias1  # with bias
        elif bias0 is not None:
            bs_tensor = torch.tensor(bias1, requires_grad=False, device=device)  #without Bias
        else:
            bs_tensor = torch.zeros([self.out_planes], requires_grad=False, device=device)

        return wt_tensor, bs_tensor
    
class RepBlockV2(nn.Module):
    def __init__(self, inp_planes, out_planes, feature_size=256, mode='train', act_type = 'linear', with_idt = True):
        super(RepBlockV2, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mode = mode
        self.mid_planes = feature_size
        self.act_type = act_type
       
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')
        
        if self.mode == 'train':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=1)
            self.w0 = self.conv0.weight
            self.b0 = self.conv0.bias

            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.w1 = self.conv1.weight
            self.b1 = self.conv1.bias
        else:
            self.repconv = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            
        if self.with_idt == False: 
            self.conv1_res = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)

    def forward(self, x):
        if self.mode == 'train':
            if self.with_idt:
                y0 = self.conv0(x)
                y0 = self.conv1(y0)
                y0 = y0 + x                    
            else:
                y0 = self.conv1(self.conv0(x)) + self.conv1_res(x)
        else: 
            RepW, RepB = self.repblock_convert()
            y0 = self.repconv(input=x, weight=RepW, bias=RepB)
        
        if self.act_type != 'linear':
            y0 = self.act(y0)    
        return y0
    
    def repblock_convert(self):         
        # get_weights
        weight0 = self.conv0.weight
        bias0 = self.conv0.bias if self.conv0.bias is not None else None
        weight1 = self.conv1.weight
        bias1 = self.conv1.bias if self.conv1.bias is not None else None

        device = weight0.get_device()
        if device < 0:
            device = None
                    #tf# (kx, ky, in, out) -> (out, in, kx, ky)
        #torch (out, in, kx, ky,) -> (out, in, kx, ky)

        # weights: [k, k, in, out]
        # [out, in, kx, ky] -> [out, in, kx*ky]
        weight0_ = weight0.reshape([weight0.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        # [out, in, kx, ky] -> [out, in*kx*ky]
        weight1_ = weight1.reshape([weight1.shape[0], weight1.shape[1]*weight1.shape[2]*weight1.shape[3]])
        # [out, in, kx*ky]
        
            
        init_val = np.zeros([weight1.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        new_weight_ = torch.tensor(init_val, requires_grad=False, dtype=torch.float, device=device)

        for i in range(weight0.shape[1]):
            tmp = weight0_[:, i, :].reshape([weight0.shape[0], weight0.shape[2]*weight0.shape[3]])
            #new_weight_[:, i, :].copy_(torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(tmp, requires_grad=False)))
            new_weight_[:, i, :].copy_(torch.matmul(weight1_, tmp))
        new_weight_ = new_weight_.reshape([weight1.shape[0], weight0.shape[1], weight0.shape[2], weight0.shape[3]])

        #residual
        residual = torch.zeros(self.out_planes, self.inp_planes, 3, 3, requires_grad=False, device=device)
        if self.with_idt:
            for i in range(self.out_planes):
                residual[i, i, 1, 1] = 1.0
        # final weight
        wt_tensor = new_weight_ + residual.cuda(device=device)

        # biases
        if bias0 is not None and bias1 is not None:
            bias0_ = bias0.reshape([bias0.shape[0], 1]) # 1d -> 2d
            #bs_tensor = torch.tensor((torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(bias0_, requires_grad=False))).reshape([bias1.shape[0]]), requires_grad=False) + torch.tensor(bias1, requires_grad=False)  # with bias
            bs_tensor = torch.matmul(weight1_, bias0_).reshape([bias1.shape[0]]) + bias1  # with bias
        elif bias0 is not None:
            bs_tensor = torch.tensor(bias1, requires_grad=False, device=device)  #without Bias
        else:
            bs_tensor = torch.zeros([self.out_channels], requires_grad=False, device=device)

        if self.with_idt == False: 
            weight1_res = self.conv1_res.weight
            bias1_res = self.conv1_res.bias if self.conv1.bias is not None else None
            
            #Weigh add
            tmp_conv3x3_init = torch.zeros(weight1_res.shape[0], weight1_res.shape[1], 3, 3, requires_grad=False, device=device)
            
            for in_ch in range(weight1_res.shape[1]):
                for out_ch in range(weight1_res.shape[0]):
                    tmp_conv3x3_init[out_ch, in_ch, 1, 1] = weight1_res[out_ch, in_ch, 0, 0]
            # final weight
            wt_tensor = wt_tensor + tmp_conv3x3_init
            
            #Bias
            if bias1_res is not None:
                bs_tensor = bs_tensor + bias1_res
                    
        return wt_tensor, bs_tensor

class RepBlockV3(nn.Module):
    def __init__(self, inp_planes, out_planes, feature_size=256, mode='train', act_type = 'linear', with_idt = True):
        super(RepBlockV3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mode = mode
        self.mid_planes = feature_size
        self.act_type = act_type
       
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')
        
        if self.mode == 'train':
            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=1)
            self.w0 = self.conv0.weight
            self.b0 = self.conv0.bias

            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.w1 = self.conv1.weight
            self.b1 = self.conv1.bias
        else:
            self.repconv = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            
        if self.with_idt == False: 
            self.conv1_res = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)

    def forward(self, x):
        if self.mode == 'train':
            if self.with_idt:
                y0 = self.conv0(x)
                y0 = self.conv1(y0)
                y0 = y0 + x                    
            else:
                y0 = self.conv1(self.conv0(x)) + self.conv1_res(x)
        else: 
            RepW, RepB = self.repblock_convert()
            y0 = self.repconv(input=x, weight=RepW, bias=RepB)
        
        if self.act_type != 'linear':
            y0 = self.act(y0)    
        return y0
    
    def repblock_convert(self):         
        # get_weights
        weight0 = self.conv0.weight
        bias0 = self.conv0.bias if self.conv0.bias is not None else None
        weight1 = self.conv1.weight
        bias1 = self.conv1.bias if self.conv1.bias is not None else None

        device = weight0.get_device()
        if device < 0:
            device = None
                    #tf# (kx, ky, in, out) -> (out, in, kx, ky)
        #torch (out, in, kx, ky,) -> (out, in, kx, ky)

        # weights: [k, k, in, out]
        # [out, in, kx, ky] -> [out, in, kx*ky]
        weight0_ = weight0.reshape([weight0.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        # [out, in, kx, ky] -> [out, in*kx*ky]
        weight1_ = weight1.reshape([weight1.shape[0], weight1.shape[1]*weight1.shape[2]*weight1.shape[3]])
        # [out, in, kx*ky]
        
            
        init_val = np.zeros([weight1.shape[0], weight0.shape[1], weight0.shape[2]*weight0.shape[3]])
        new_weight_ = torch.tensor(init_val, requires_grad=False, dtype=torch.float, device=device)

        for i in range(weight0.shape[1]):
            tmp = weight0_[:, i, :].reshape([weight0.shape[0], weight0.shape[2]*weight0.shape[3]])
            #new_weight_[:, i, :].copy_(torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(tmp, requires_grad=False)))
            new_weight_[:, i, :].copy_(torch.matmul(weight1_, tmp))
        new_weight_ = new_weight_.reshape([weight1.shape[0], weight0.shape[1], weight0.shape[2], weight0.shape[3]])

        #residual
        residual = torch.zeros(self.out_planes, self.inp_planes, 3, 3, requires_grad=False, device=device)
        if self.with_idt:
            for i in range(self.out_planes):
                residual[i, i, 1, 1] = 1.0
        # final weight
        wt_tensor = new_weight_ + residual.cuda(device=device)

        # biases
        if bias0 is not None and bias1 is not None:
            bias0_ = bias0.reshape([bias0.shape[0], 1]) # 1d -> 2d
            #bs_tensor = torch.tensor((torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(bias0_, requires_grad=False))).reshape([bias1.shape[0]]), requires_grad=False) + torch.tensor(bias1, requires_grad=False)  # with bias
            bs_tensor = torch.matmul(weight1_, bias0_).reshape([bias1.shape[0]]) + bias1  # with bias
        elif bias0 is not None:
            bs_tensor = torch.tensor(bias1, requires_grad=False, device=device)  #without Bias
        else:
            bs_tensor = torch.zeros([self.out_channels], requires_grad=False, device=device)

        if self.with_idt == False: 
            weight1_res = self.conv1_res.weight
            bias1_res = self.conv1_res.bias if self.conv1.bias is not None else None
            
            #Weigh add
            tmp_conv3x3_init = torch.zeros(weight1_res.shape[0], weight1_res.shape[1], 3, 3, requires_grad=False, device=device)
            
            for in_ch in range(weight1_res.shape[1]):
                for out_ch in range(weight1_res.shape[0]):
                    tmp_conv3x3_init[out_ch, in_ch, 1, 1] = weight1_res[out_ch, in_ch, 0, 0]
            # final weight
            wt_tensor = wt_tensor + tmp_conv3x3_init
            
            #Bias
            if bias1_res is not None:
                bs_tensor = bs_tensor + bias1_res
                    
        return wt_tensor, bs_tensor
    
if __name__ == '__main__':
    # test ecb
    x = torch.randn(1, 3, 5, 5, dtype=torch.float, requires_grad=False).cuda() * 200
    ecb = RepBlockV2(inp_planes=3, out_planes=5, feature_size=256, mode='train', act_type= 'linear', with_idt = True).cuda().eval()
    #ecb = RepBlock(3, 3, 2, act_type='linear', with_idt=True, with_bn=False).cuda().eval()
    y0 = ecb(x)

    RK, RB = ecb.repblock_convert()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, padding=1)
    print(y0-y1)
    
    print('->Matching Error: {}'.format(np.mean((y0.detach().cpu().numpy() - y1.detach().cpu().numpy()) ** 2)))    # Will be around 1e-10