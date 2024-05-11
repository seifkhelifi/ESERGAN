import torch.nn as nn
from torchvision.models import vgg19
import config


class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        vgg_input_features = self.vgg(output)
        vgg_target_features = self.vgg(target)
        loss = self.loss(vgg_input_features, vgg_target_features)

        return loss