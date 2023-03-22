import os
from source.datas.benchmark import Benchmark
from source.datas.div2k import DIV2K
from source.datas.val_data_loader import VALPhaseLoader
from torch.utils.data import DataLoader

def create_datasets(args, mode='train'):
    if mode == 'train':
        div2k = DIV2K(
            os.path.join(args.data_path, 'train_HR/'), 
            os.path.join(args.data_path, 'train_LR/'), 
            os.path.join(args.data_path, 'cache'),
            train=True, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            normalize=args.normalize
        )
        train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
        
        valid_dataloaders = []
        if 'DIV2K' in args.eval_sets:
            div2k_val = DIV2K(
                os.path.join(args.data_path, 'val_HR/'), 
                os.path.join(args.data_path, 'val_LR/'), 
                os.path.join(args.data_path, 'cache_val'),
                train=False, 
                augment=args.data_augment, 
                scale=args.scale, 
                colors=args.colors, 
                patch_size=args.patch_size, 
                repeat=args.data_repeat, 
                normalize=args.normalize
            )
            valid_dataloaders += [{'name': 'DIV2K', 'dataloader': DataLoader(dataset=div2k_val, batch_size=1, shuffle=False)}]
    else: #test mode
        # test_loader = DIV2K(
        #         os.path.join(args.test_path, 'val_phase_HR/'), 
        #         os.path.join(args.test_path, 'val_phase_LR/'), 
        #         os.path.join(args.test_path, 'val_phase_cache'),
        #         train=False, 
        #         augment=args.data_augment, 
        #         scale=args.scale, 
        #         colors=args.colors, 
        #         patch_size=args.patch_size, 
        #         repeat=args.data_repeat, 
        #     )
        # test_dataloader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)
            
        test_loader = VALPhaseLoader(
            os.path.join(args.test_path, 'val_phase_HR/'), 
            os.path.join(args.test_path, 'val_phase_LR/'), 
            os.path.join(args.test_path, 'val_phase_cache'),
            train=False, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
            normalize=args.normalize
        )
        test_dataloader = DataLoader(dataset=test_loader, batch_size=1, shuffle=False)
        return test_dataloader
    return train_dataloader, valid_dataloaders