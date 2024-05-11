import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

high_res = 1024
low_res = high_res // 4
num_channels = 3
lr = 1e-4
num_epochs = 5
batch_size = 4
lambda_gp = 10
num_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

highres_transform = A.Compose(
    [
        A.Resize(width=high_res, height=high_res),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0 , 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=high_res, height=high_res),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)