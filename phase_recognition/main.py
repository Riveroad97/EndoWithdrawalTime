import os
import random
import argparse
import numpy as np
from enum import Flag

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

from utils import *
from dataset import *
from refine_model import *
from base_model import BaseCausalTCN
from hierarch_causal_tcn import Hierarch_CausalTCN
from prototype import hierarch_train, hierarch_predict, refine_train, refine_predict, base_train, base_predict


## Fix random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def get_args_parser():
    parser = argparse.ArgumentParser('SAHC_HardFrame', add_help=False)
    parser.add_argument('--action', default='hierarch_train')
    parser.add_argument('--name', default="cross_aug3")
    parser.add_argument('--dataset', default="cecum4")
    parser.add_argument('--dataset_path', default="/workspace/nasEndo/endoscope_video/하부/2번방/cecum_recognition")

    parser.add_argument('--sample_rate', default=5, type=int)
    parser.add_argument('--test_sample_rate', default=5, type=int)
    parser.add_argument('--num_classes', default=4)
    parser.add_argument('--model', default="Hierarch_TCN")
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--combine_loss', default=False, type=bool)
    parser.add_argument('--ms_loss', default=True, type=bool)

    parser.add_argument('--fpn', default=True, type=bool)
    parser.add_argument('--output', default=False, type=bool)
    parser.add_argument('--feature', default=False, type=bool)
    parser.add_argument('--trans', default=True, type=bool)
    parser.add_argument('--prototype', default=False, type=bool)
    parser.add_argument('--last', default=False, type=bool)
    parser.add_argument('--first', default=True, type=bool)
    parser.add_argument('--hier', default=True, type=bool)
    
    ####ms-tcn2
    parser.add_argument('--dim', default="2048", type=int)
    parser.add_argument('--num_layers_PG', default="11", type=int)
    parser.add_argument('--num_layers_R', default="10", type=int)
    parser.add_argument('--num_f_maps', default="64", type=int)
    parser.add_argument('--num_R', default="3", type=int)

    ##Transformer
    parser.add_argument('--head_num', default=8)
    parser.add_argument('--embed_num', default=512)
    parser.add_argument('--block_num', default=1)
    parser.add_argument('--positional_encoding_type', default="learned", type=str, help="fixed or learned")

    ## Hard Frame
    parser.add_argument('--hard_frame', default='off', type=str)

    ## Refinement
    parser.add_argument('--refine_model', default='gru')
    parser.add_argument('--refine_epochs', default=40)
    parser.add_argument('--refine_learning_rate', default=1e-4)
    parser.add_argument('--num_stages', default="3", type=int)
    parser.add_argument('--num_layers', default="12", type=int)

    ## GPU
    parser.add_argument('--cuda_visible_devices', default="0")

    return parser
    

def main(args):
    
    print(args)

    # GPU Setting
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if args.model == 'Hierarch_CausalTCN':
        base_model = Hierarch_CausalTCN(args, args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, args.dim, args.num_classes)

    elif args.model == 'Base_CausalTCN':
        base_model = BaseCausalTCN(args.num_R, args.num_f_maps, args.dim, args.num_classes)

    if args.action == 'hierarch_train':
        video_traindataset = TestVideoDataset(args.dataset, args.dataset_path, args.sample_rate, args.name, 'train')
        video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
        video_testdataset = TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'valid')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False) 
        model_save_dir = f'experiments/models/{args.name}/train/'
        hierarch_train(args, base_model, video_train_dataloader, video_test_dataloader, device, save_dir=model_save_dir, debug=True)

    elif args.action == 'hierarch_predict':
        if args.hard_frame=='on':
            model_path = f'experiments/models/{args.name}/train/{args.model}/hard_frame/best.model'
        else:
            model_path = f'experiments/models/{args.name}/train/{args.model}/no_hard_frame/best.model'
    
        base_model.load_state_dict(torch.load(model_path))
        video_testdataset =TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'test')
        # video_testdataset =TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'valid')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        hierarch_predict(base_model, args, device, video_test_dataloader, args.name)
    
    elif args.action == 'base_train':
        video_traindataset = TestVideoDataset(args.dataset, args.dataset_path, args.sample_rate, args.name, 'train')
        video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
        video_testdataset = TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'valid')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False) 
        model_save_dir = f'experiments/models/{args.name}/train/'
        base_train(args, base_model, video_train_dataloader, video_test_dataloader, device, save_dir=model_save_dir, debug=True)
    
    elif args.action == 'base_predict':
        if args.hard_frame=='on':
            model_path = f'experiments/models/{args.name}/train/{args.model}/hard_frame/best.model'
        else:
            model_path = f'experiments/models/{args.name}/train/{args.model}/no_hard_frame/best.model'
        
        base_model.load_state_dict(torch.load(model_path))
        video_testdataset =TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'test')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        base_predict(base_model, args, device, video_test_dataloader, args.name)


    elif args.action == 'refine_train':

        if args.refine_model == 'gru':
            refine_model = MultiStageRefineGRU(num_stage=args.num_stages, num_f_maps=128, num_classes=args.num_classes)
        elif args.refine_model == 'causal_tcn':
            refine_model = MultiStageRefineCausalTCN(args.num_stages, args.num_layers, args.num_f_maps, args.num_classes, args.num_classes)
        elif args.refine_model == 'tcn':
            refine_model = MultiStageRefineTCN(args.num_stages, args.num_layers, args.num_f_maps, args.num_classes, args.num_classes)

        base_model_path = f'experiments/models/{args.name}/train/{args.model}/best.model'
        base_model.load_state_dict(torch.load(base_model_path))

        video_traindataset = TestVideoDataset(args.dataset, args.dataset_path, args.sample_rate, args.name, 'train')
        video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=True, drop_last=False)
        video_testdataset = TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'valid')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        
        model_save_dir =  f'experiments/models/{args.name}/refine/'
        refine_train(args, base_model, refine_model, video_train_dataloader, video_test_dataloader, device, model_save_dir, debug=True)

    elif args.action == 'refine_predict':
        base_model_path = f'experiments/models/{args.name}/train/{args.model}/best.model'
        refine_model_path = f'experiments/models/{args.name}/refine/{args.refine_model}/best.model'

        base_model.load_state_dict(torch.load(base_model_path))
        refine_model.load_state_dict(torch.load(refine_model_path))

        video_testdataset =TestVideoDataset(args.dataset, args.dataset_path, args.test_sample_rate, args.name, 'test')
        video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False)
        refine_predict(base_model, refine_model, args, device, video_test_dataloader, args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Endo Phase Recognition', parents=[get_args_parser()])
    args   = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    main(args)


