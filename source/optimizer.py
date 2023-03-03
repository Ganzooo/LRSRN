import torch.optim as optim

from adamp import AdamP
from adabelief_pytorch import AdaBelief

def get_optimizer(cfg, model):
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd)
    elif cfg.optimizer == 'adamp':
        optimizer = AdamP(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd)
    elif cfg.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=cfg.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple = True, rectify = False)
    elif cfg.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    else:
        raise NameError('Choose proper optimizer!!!')
    return optimizer
