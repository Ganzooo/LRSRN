from .plainRepConv import PlainRepConv, PlainRepConv_BlockV2,PlainRepConv_deploy, PlainRepConv_BlockV2_deploy

def get_model(cfg, device, mode='Train'):
    if cfg.model == 'PlainRepConv':
        if mode == 'Train':
            model = PlainRepConv(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    elif cfg.model == 'PlainRepConv_BlockV2':
        if mode == 'Train':
            model = PlainRepConv_BlockV2(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
        else: 
            model = PlainRepConv_BlockV2_deploy(module_nums=cfg.m_plainsr, channel_nums=cfg.c_plainsr, act_type=cfg.act_type, scale=cfg.scale, colors=cfg.colors)
    else: 
        raise NameError('Choose proper model name!!!')
    model.to(device)
    return model

# if __name__ == "__main__":
#     cfg.model.name = 'unet_res50'
#     modelTrain = get_model(cfg)
#     print(modelTrain)
