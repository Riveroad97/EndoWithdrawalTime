import os
import json
import glob
import torch
import pathlib
import natsort
import argparse
import numpy as np
from tqdm import tqdm

from core.utils import *
from core.create_model import *
from core.datasets.H5Dataset import HDF5Dataset_Test

seed_everything(42)

parser = argparse.ArgumentParser('Endo Withdrawal Framewise feature extraction', add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='configs/framewise_cross.json',
                    type=str)
args = parser.parse_args()


def main(config):
    # GPU Setting
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Select Model
    print(f"Creating model  : {config['name']}")    
    checkpoint = torch.load(config['checkpoint'])
    model = create_model(name=config['name'], num_class=config['num_class']).to(device)
    model.load_state_dict(checkpoint['model'])

    all_video_h5 = natsort.natsorted(glob.glob('./data/all_video_h5/*.hdf5'))

    for h5 in tqdm(all_video_h5):
        video_name = h5.split('/')[-1].split('.')[0]
        testset = HDF5Dataset_Test(video_name=video_name, 
                                   file_path=h5)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, pin_memory=True, drop_last=False)
        with torch.no_grad():
            model.eval()
            video_feature = []
            for batch in tqdm(testloader):
                save_path = config['save_path']
                pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
                image = batch[0].to(device).float()
                feature = model.extract_feat(image).detach().cpu().numpy()
                video_feature.append(feature)

            save_feature = os.path.join(save_path, video_name + '.npy')
            video_feature = np.concatenate(video_feature, axis=0)
            np.save(save_feature, video_feature)
    

if __name__ == '__main__':
    config = json.load(open(args.config))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
   
    main(config)