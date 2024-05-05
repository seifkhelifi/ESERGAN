import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth"
CHECKPOINT_DISC = "disc.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
BATCH_SIZE = 16
LAMBDA_GP = 10
NUM_WORKERS = 4
HIGH_RES = 1024
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

# This transformation is designed for high-resolution images. It normalizes the image 
# by subtracting the mean [0, 0, 0] and dividing by the standard deviation [1, 1, 1]. 
# Then, it converts the image to a PyTorch tensor.
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# This transformation is for low-resolution images. It resizes the image to a fixed 
# size specified by LOW_RES using bicubic interpolation. Then, it normalizes the image 
# and converts it to a PyTorch tensor.
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# This transformation is applied to both high and low-resolution images during training. 
# It randomly crops an area of size (HIGH_RES, HIGH_RES) from the image, then 
# randomly flips it horizontally with a probability of 0.5, and randomly rotates it 
# by 90 degrees with a probability of 0.5.
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

# This transformation is applied to test images. Similar to highres_transform, 
# it normalizes the image and converts it to a PyTorch tensor.
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)