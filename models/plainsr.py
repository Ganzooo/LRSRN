import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PlainSR(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        #self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type='linear')
        self.head = nn.Sequential(
            nn.Conv2d(colors, channel_nums, 3, padding=1)
        )
        backbone = []
        for i in range(self.module_nums):
            backbone.append(nn.Conv2d(channel_nums, channel_nums, 3, padding=1))
            if i > self.module_nums:
                backbone.append(nn.ReLU(True))
            #backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]

        self.backbone = nn.Sequential(*backbone)

        self.transition = nn.Sequential(nn.Conv2d(channel_nums, colors * (self.scale ** 2), 3, padding=1))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0)
        y = self.transition(y + y0)
        
        y = self.upsampler(y)
        return y

class PlainSR2(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainSR2, self).__init__()
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
        
        self.transition = nn.Sequential(torch.nn.Conv2d(self.channel_nums+3, self.channel_nums, kernel_size=1, padding=0),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear'))
        
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        #y = self.backbone(x) + x
        #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
        y = self.head(x)
        y = self.backbone(y) 
        
        y = torch.cat([y, x], dim=1)
        
        y = self.transition(y)
        y = self.upsampler(y)
        
        return y