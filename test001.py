import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet50
from thop import profile

from models.lenet import LeNet


import time

import torch
import torchvision
import torchvision.transforms as transforms


from model import LeNet, VGG16
from models.resnet import ResNet


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ]
    )

    # model = LeNet()
    # model = VGG16()
    model = ResNet()
    input = torch.randn(1, 1, 28, 28)
    flops, params = profile(model, inputs=(input,))
    print("FLOPs:", flops)
    print("params", params)


if __name__ == '__main__':
    main()





