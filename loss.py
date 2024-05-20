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
    
    
class ESRGANLoss(nn.Module):
    def __init__(self, l1_weight=1e-2, vgg_weight=1, device='cpu'):
        super().__init__()
        
        self.vgg_criterion = VGGLoss(device)
        self.l1_criterion = nn.L1Loss().to(device)
        self.l1_weight = l1_weight
        
    def forward(self, output, target):
        vgg_loss = self.vgg_criterion(output, target)
        l1_loss = self.l1_criterion(output, target)
        
        loss = vgg_loss + self.l1_weight * l1_loss
        return loss