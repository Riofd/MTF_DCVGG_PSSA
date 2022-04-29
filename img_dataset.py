from PIL import Image
from torch.utils.data import Dataset
import os
import mat4py
import numpy as np


class DimDataset(Dataset):
    def __init__(self, root='E:/Users/zrf/转移文件夹/样本/知识', rotor_gaf=False, vol_gaf=False,
                 transform=None, target_transform=None):
        self.root = root
        self.rotor_root = self.root + '/rotor_image/' if not rotor_gaf else self.root + '/gaf_rotor/'
        self.vol_root = self.root + '/vol_image/' if not vol_gaf else self.root + '/gaf_vol/'
        self.transform = transform
        self.target_transform = target_transform
        self.rotor_image_files = os.listdir(self.rotor_root)
        self.rotor_image_files.sort(key=lambda x: int(x[:-4]))
        self.vol_image_files = os.listdir(self.vol_root)
        self.vol_image_files.sort(key=lambda x: int(x[:-4]))

    def __getitem__(self, index):
        rotor_img = Image.open(self.rotor_root + self.rotor_image_files[index])
        vol_img = Image.open(self.vol_root + self.vol_image_files[index])
        labels = mat4py.loadmat(self.root+'/label.mat')
        labels = np.array(labels['label'])
        label = labels[index]

        if self.transform is not None:
            rotor_img = self.transform(rotor_img)
            vol_img = self.transform(vol_img)
        return rotor_img, vol_img, label

    def __len__(self):
        return len(self.rotor_image_files)

