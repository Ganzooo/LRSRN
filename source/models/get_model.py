from .plainsr import PlainSR, PlainSR2
from .plainRepConv import PlainRepConv, PlainRepConv_st01, PlainRepConv_BlockV2, PlainRepConv_All, PlainRepConvClip, PlainRepConv_deploy, PlainRepConv_BlockV2_deploy
from .plainkh import PlainKH
# from .imdn_baseline import IMDN
from .fsrcnn import FSRCNN
from .rtsrn import RTSRN

def get_model(cfg, device, mode='Train'):
    if cfg.model == 'plainsr':
        model = PlainSR(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'plainsr2':
        model = PlainSR2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv':
        if mode == 'Train':
            model = PlainRepConv(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConvClip':
        model = PlainRepConvClip(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_st01':
        model = PlainRepConv_st01(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_All':
        model = PlainRepConv_All(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)    
    elif cfg.model == 'PlainRepConv_BlockV2':
        if mode == 'Train':
            model = PlainRepConv_BlockV2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_BlockV2_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'Plainkh':
        model = PlainKH(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    # elif cfg.model == 'IMDN':
    #     model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=cfg.c_plainsr, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05)
    elif cfg.model == "FSRCNN":
        model = FSRCNN(colors=3, upscale_factor=cfg.scale)
    elif cfg.model == "RTSRN":
        model = RTSRN(num_channels=3, num_feats=64, num_blocks=5, upscale=cfg.scale)
    
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
