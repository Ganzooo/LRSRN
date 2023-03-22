import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act  = None

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

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

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
            #bs_tensor = torch.tensor((torch.matmul(torch.tensor(weight1_, requires_grad=False), torch.tensor(bias0_, requires_grad=False))).reshape([bias1.shape[0]]), requires_grad=False) + torch.tensor(bias1, requires_grad=False)  # with bias
            bs_tensor = torch.matmul(weight1_, bias0_).reshape([bias1.shape[0]]) + bias1  # with bias
        elif bias0 is not None:
            bs_tensor = torch.tensor(bias1, requires_grad=False, device=device)  #without Bias
        else:
            bs_tensor = torch.zeros([self.out_channels], requires_grad=False, device=device)

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
    
class PlainRepConv(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainRepConv, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [RepBlock(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0) 

        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y
    
    def fuse_model(self):
        ## reparam as plainsrcd cd
        for idx, blk in enumerate(self.backbone):
            if type(blk) == RepBlock:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone
                self.backbone[idx] = conv.to(RK.device)

class PlainRepConv_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainRepConv_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0) 

        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y
                
class PlainRepConv_BlockV2(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainRepConv_BlockV2, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.head = RepBlockV2(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [RepBlockV2(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        RepBlockV2(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0) 

        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y
    
    def fuse_model(self):
        ## reparam as plainsrcd cd
        for idx, blk in enumerate(self.backbone):
            if type(blk) == RepBlockV2:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone
                self.backbone[idx] = conv.to(RK.device)
        #for idx, blk in enumerate(self.head):
        if type(self.head) == RepBlockV2:
            RK, RB  = self.head.repblock_convert()
            conv = Conv3X3(self.head.inp_planes, self.head.out_planes, act_type=self.head.act_type)
            ## update weights & bias for conv3x3
            conv.conv3x3.weight.data = RK
            conv.conv3x3.bias.data   = RB
            ## update weights & bias for activation
            if self.head.act_type == 'prelu':
                conv.act.weight = self.head.act.weight
            ## update block for backbone
            self.head = conv.to(RK.device)
        for idx, blk in enumerate(self.transition):
            if type(blk) == RepBlockV2:
                RK, RB  = blk.repblock_convert()
                conv = Conv3X3(blk.inp_planes, blk.out_planes, act_type=blk.act_type)
                ## update weights & bias for conv3x3
                conv.conv3x3.weight.data = RK
                conv.conv3x3.bias.data   = RB
                ## update weights & bias for activation
                if blk.act_type == 'prelu':
                    conv.act.weight = blk.act.weight
                ## update block for backbone
                self.transition[idx] = conv.to(RK.device)
        
class PlainRepConv_BlockV2_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainRepConv_BlockV2_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.transition = nn.Sequential(#torch.nn.Conv2d(self.channel_nums, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0) 

        y = self.transition(y + y0)
        y = self.upsampler(y) 
        return y
                
    
# if __name__ == "__main__":
#     x = torch.rand(1,3,128,128).cuda()
#     model = PlainRepConv(module_nums=6, channel_nums=64, act_type='prelu', scale=3, colors=3).cuda().eval()
#     y0 = model(x)

#     model.fuse_model()
#     y1 = model(x)

#     print(model)
#     print(y0-y1)
#     print('->Matching Error: {}'.format(np.mean((y0.detach().cpu().numpy() - y1.detach().cpu().numpy()) ** 2)))    # Will be around 1e-10
