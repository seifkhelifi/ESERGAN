import torch
from tqdm import tqdm
import time
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
from PIL import Image
import cv2


class SRDataset(Dataset):
    def __init__(self, root_dir, lowres_transform, highres_transform, both_transform):
        super(SRDataset, self).__init__()
        
        self.root_dir = root_dir
        
        self.lowres_transform = lowres_transform
        self.highres_transform = highres_transform
        self.both_transform = both_transform
        
        self.list_of_files = os.listdir(os.path.join(root_dir))

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.root_dir, self.list_of_files[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.both_transform(image=image)['image']
        low_res_image = self.lowres_transform(image=image)['image']
        high_res_image = self.highres_transform(image=image)['image']
        
        return low_res_image, high_res_image

