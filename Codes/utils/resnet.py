import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.resnet = resnet18(pretrained=False, num_classes=num_classes)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x

