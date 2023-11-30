import os
import numpy as np

import torch
from torch.utils.data import Dataset


phase2label_dicts = {
    'cecum3':{
    'cecum':0,
    'background':1,
    'out':2
    },
    'cecum4':{
    'cecum':0,
    'background':1,
    'out':2,
    'surgery':3
    }
}

def phase2label(phases, phase2label_dict):
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else 1 for phase in phases] # 나머지 것들은 1로 처리
    return labels

def label2phase(labels, phase2label_dict):
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases


class TestVideoDataset(Dataset):
    def __init__(self, dataset, root, sample_rate, name, mode):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []
        self.hard_frames = [] 
        self.video_names = []
        if name == '':
            name = ''
        else:
            name = f'_{name}'
        video_feature_folder = os.path.join(root, 'video_feature' + f'{name}') + f'/{mode}'
        label_folder = os.path.join(root, 'frame_label')
        hard_frames_folder = os.path.join(root, 'hard_frame')

        for v_f in os.listdir(video_feature_folder):
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            v_hard_frames_abs_path = os.path.join(hard_frames_folder, v_f)
            
            labels = self.read_labels(v_label_file_abs_path) 
            
            labels = labels[::sample_rate]
            videos = np.load(v_f_abs_path)[::sample_rate,]
            masks = np.load(v_hard_frames_abs_path)[::sample_rate]
            assert len(labels) == len(masks)

            self.videos.append(videos)
            self.labels.append(labels)
            self.hard_frames.append(masks)
            
            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, item):
        video, label, mask, video_name = self.videos[item], self.labels[item], self.hard_frames[item], self.video_names[item]
        return video, label, mask, video_name
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()][1:] # 첫번째 줄 날림
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels