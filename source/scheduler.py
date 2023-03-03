import torch.optim as optim
from torch.optim import lr_scheduler

from adamp import AdamP
from adabelief_pytorch import AdaBelief
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

def get_scheduler(cfg, optimizer):
    # if cfg.optimizer.scheduler == 'CosineAnnealingLR':
    #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max = _T_max, 
    #                                                eta_min=cfg.optimizer.min_lr)
    # elif cfg.optimizer.scheduler == 'CosineAnnealingWarmRestarts':
    #     scheduler = CosineAnnealingWarmupRestarts(optimizer,
    #                                               first_cycle_steps=cfg.train_config.epochs//4,
    #                                               cycle_mult=1.0,
    #                                               max_lr=cfg.optimizer.lr,
    #                                               min_lr=cfg.optimizer.min_lr,
    #                                               #warmup_steps=cfg.train_config.epochs//8,
    #                                               gamma=cfg.optimizer.gamma,)
    # elif cfg.optimizer.scheduler == 'ReduceLROnPlateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                mode='min',
    #                                                factor=0.1,
    #                                                patience=7,
    #                                                threshold=0.0001,
    #                                                min_lr=cfg.optimizer.min_lr.min_lr,)
    # elif cfg.optimizer.scheduler == 'ExponentialLR':
    #     scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    if cfg.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decays, gamma=cfg.gamma)
    else:
        raise NameError('Choose proper scheduler!!!')
    return scheduler