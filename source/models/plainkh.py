import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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
        elif self.act_type == 'sg':
            self.act = SimpleGate()
            
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

class PlainKH(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors):
        super(PlainKH, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        ### HEAD
        self.head = nn.Sequential(nn.Conv2d(colors, channel_nums, 3, padding=1))
        
        ###BackBone
        backbone = []
        
        backbone.append(nn.Conv2d(channel_nums, 2*channel_nums, 3, padding=1))
        backbone.append(nn.ReLU(True))

        backbone.append(nn.Conv2d(2*channel_nums, 4*channel_nums, 3, padding=1))
        backbone.append(nn.ReLU(True))

        backbone.append(nn.Conv2d(4*channel_nums, 4*channel_nums, 3, padding=1))
        backbone.append(SimpleGate())

        backbone.append(nn.Conv2d(2*channel_nums, 2*channel_nums, 3, padding=1))
        backbone.append(SimpleGate())

        self.backbone = nn.Sequential(*backbone)

        ### Transition
        self.transition = nn.Sequential(nn.Conv2d(channel_nums, colors * (self.scale ** 2), 3, padding=1))
        
        ### Upsample
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        y0 = self.head(x)
        y = self.backbone(y0)
        y = self.transition(y + y0)
        
        y = self.upsampler(y)
        return y

# class PlainKH2(nn.Module):
#     def __init__(self, module_nums, channel_nums, act_type, scale, colors):
#         super(PlainKH2, self).__init__()
#         self.module_nums = module_nums
#         self.channel_nums = channel_nums
#         self.scale = scale
#         self.colors = colors
#         self.act_type = act_type
#         self.backbone = None
#         self.upsampler = None
#         self.sg = SimpleGate()

#         ### HEAD
#         self.head = nn.Sequential(nn.Conv2d(colors, channel_nums, 3, padding=1))
        
#         ###BackBone        
#         self.backbone1 = nn.Conv2d(channel_nums, channel_nums, 3, padding=1)
#         self.backbone2 = nn.Conv2d(channel_nums, channel_nums, 3, padding=1)
#         self.backbone3 = nn.Conv2d(channel_nums, channel_nums, 3, padding=1)
#         self.backbone4 = nn.Conv2d(channel_nums, channel_nums, 3, padding=1)

#         ### Transition
#         self.transition = nn.Sequential(nn.Conv2d(channel_nums, colors * (self.scale ** 2), 3, padding=1))
        
#         ### Upsample
#         self.upsampler = nn.PixelShuffle(self.scale)
    
#     def forward(self, x):
#         #y = self.backbone(x) + x
#         #_x = torch.cat([x, x, x, x, x, x, x, x, x], dim=1)
        
#         x = self.head(x)
#         y1 = self.backbone1(x)
#         y1 = self.sg(y1)


        
#         y = torch.cat([y, x], dim=1)
        
#         y = self.transition(y)
#         y = self.upsampler(y)
        
#         return y