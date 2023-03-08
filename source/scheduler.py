import torch.optim as optim
from torch.optim import lr_scheduler

from adamp import AdamP
from adabelief_pytorch import AdaBelief
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def get_scheduler(cfg, optimizer):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = 2, 
                                                   eta_min=cfg.optimizer.min_lr)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
#        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epochs//5, T_mult=2, eta_min=cfg.min_lr)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=cfg.epochs//5,
                                                  cycle_mult=1.0,
                                                  max_lr=cfg.lr,
                                                  min_lr=cfg.min_lr,
                                                  #warmup_steps=cfg.train_config.epochs//8,
                                                  gamma=cfg.gamma)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.optimizer.min_lr.min_lr,)
    elif cfg.optimizer.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif cfg.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)
    else:
        raise NameError('Choose proper scheduler!!!')
    return scheduler