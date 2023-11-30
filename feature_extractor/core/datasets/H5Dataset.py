import h5py
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset


train_transform = A.Compose([
                A.OneOf([
                A.MedianBlur(blur_limit=5, p=1),
                A.MotionBlur(blur_limit=7, p=1)],
                p=0.9),
                
                A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.4), contrast_limit=(-0.3, 0.4), p=1), # 내부 brightness_limit=(-1, 1) #노말라이즈 했기 때문에(-1, 1)사이값

                A.OpticalDistortion(distort_limit=0.05, p=0.2),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
                ])


test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

    ToTensorV2()])


class HDF5Dataset(Dataset):
    def __init__(self, file_path, mode, blacklist=[]):
        self.file_path = file_path
        self.mode = mode
        self.length = None
        self._idx_to_name = {}
        self.blacklist = blacklist
        
        self.transform = train_transform if mode == 'train' else test_transform

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname not in self.blacklist:
                    start_idx = self.length if self.length else 0
                    self.length = start_idx + len(group)
                    for i, meta in enumerate(group.items()):
                        self._idx_to_name[start_idx + i] = meta[0] # image_name
                        
        print(f'{mode}dataset: Load dataset with {self.__len__()} images.')

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        assert self._idx_to_name is not None
        img_name = self._idx_to_name[index]
        video_name = img_name.split('_')[0]

        ds = self._hf[video_name][img_name]
        image = ds[()]
        
        label = torch.tensor(ds.attrs['class'])

        image = self.transform(image=image)['image']

        return image, label, img_name


class HDF5Dataset_Test(Dataset):
    def __init__(self, file_path, video_name):
        self.file_path = file_path
        self.video_name = video_name
        self.length = None
        self._idx_to_name = {}
        
        self.transform = test_transform

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname == video_name:
                    self.length = len(group)
                    for i, meta in enumerate(group.items()):
                        self._idx_to_name[i] = meta[0] # image_name

        print(f'{video_name}dataset: Load dataset with {self.__len__()} images.')

    def __len__(self):
        assert self.length is not None
        return self.length

    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        assert self._idx_to_name is not None
        img_name = self._idx_to_name[index]

        ds = self._hf[self.video_name][img_name]
        image = ds[()]
        label = torch.tensor(ds.attrs['class'])

        image = self.transform(image=image)['image']

        return image, label, img_name
    

