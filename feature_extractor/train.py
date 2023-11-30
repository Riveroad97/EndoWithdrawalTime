import os
import json
import torch
import pathlib
import argparse
import numpy as np

from core.utils import *
from core.engine import *
from core.create_model import *
from core.lr_scheduler import *
from core.datasets.H5Dataset import HDF5Dataset


parser = argparse.ArgumentParser('Endo Withdrawal Framewise Train', add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='configs/train/cross.json',
                    type=str)
args = parser.parse_args()

# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()


def main(config):

    # GPU Setting
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    print(device)
    
    
    print("Loading dataset ....")

    video_list = np.load('data/train_video_list.npy')
    
    testlist = ['REC-201007-1513', 'REC-201022-1348', 'REC-201023-1626', 'REC-201027-1502']
    trainlist = [i for i in video_list if i not in testlist]

    train_dataset = HDF5Dataset(mode='train', 
                                file_path=config['data'],
                                blacklist=testlist)
    
    valid_dataset = HDF5Dataset(mode='valid',
                                file_path=config['data'],
                                blacklist=trainlist)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, pin_memory=True, drop_last=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)

    # Select Model
    print(f"Creating model  : {config['name']}")
    model = create_model(name=config['name'], num_class=config['num_class'])
    print(model)

    # Multi GPU
    if config['gpu_mode'] == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif config['gpu_mode'] == 'Single':
        model.to(device)
    else :
        raise Exception('Error...! gpu_mode')
    
    # Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)])
    scheduler = create_scheduler(optimizer=optimizer, config=config)

    # Save Weight & Log
    save_path = os.path.join(config['save_path'], 'weight')
    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)

    log_save_path = os.path.join(config['save_path'], 'logs')
    pathlib.Path(log_save_path).mkdir(exist_ok=True, parents=True)

    # Etc training setting
    print(f"Start training for {config['epochs']} epochs")

    start_epoch = 0

    # Resume
    if config['resume'] == 'on':
        model, optimizer, scheduler, start_epoch = load(ckpt_dir=save_path, model=model, optimizer=optimizer, lr_scheduler=scheduler)
        fix_optimizer(optimizer)

    # Whole LOOP Train & Valid 
    for epoch in range(start_epoch, config['epochs']):
        # Train & Valid    
        if config['name'] == 'Framewise_Cross':
            train_logs = train_Framewise_Cross(model, trainloader, optimizer, device, epoch, config)
            print("Averaged train_stats: ", train_logs)
            valid_logs = valid_Framewise_Cross(model, validloader, device, epoch, config)
            print("Averaged valid_stats: ", valid_logs)

        elif config['name'] == 'Framewise_ArcFace':
            train_logs = train_Framewise_ArcFace(model, trainloader, optimizer, device, epoch, config)
            print("Averaged train_stats: ", train_logs)
            valid_logs = valid_Framewise_ArcFace(model, validloader, device, epoch, config)
            print("Averaged valid_stats: ", valid_logs)

        elif config['name'] == 'Framewise_MagFace':
            train_logs = train_Framewise_MagFace(model, trainloader, optimizer, device, epoch, config)
            print("Averaged train_stats: ", train_logs)
            valid_logs = valid_Framewise_MagFace(model, validloader, device, epoch, config)
            print("Averaged valid_stats: ", valid_logs)
        
        if epoch % 1 == 0:
            save(ckpt_dir=save_path, model=model, optimizer=optimizer, lr_scheduler=scheduler, epoch=epoch+1, config=config)

        log_stats = {**{f'train_{k}': v for k, v in train_logs.items()},
                    **{f'valid_{k}': v for k, v in valid_logs.items()},
                    'epoch': epoch}
        
        with open(log_save_path +'/log.txt', 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
    
        scheduler.step()


if __name__ == '__main__':
    config = json.load(open(args.config))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
   
    main(config)
    

    


