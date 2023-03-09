from .plainsr import PlainSR, PlainSR2
from .plainRepConv import PlainRepConv, PlainRepConv_st01, PlainRepConv_BlockV2, PlainRepConv_All
from .plainkh import PlainKH

def get_model(cfg, device):
    if cfg.model == 'plainsr':
        model = PlainSR(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'plainsr2':
        model = PlainSR2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv':
        model = PlainRepConv(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_st01':
        model = PlainRepConv_st01(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_All':
        model = PlainRepConv_All(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)    
    elif cfg.model == 'PlainRepConv_BlockV2':
        model = PlainRepConv_BlockV2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'Plainkh':
        model = PlainKH(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
