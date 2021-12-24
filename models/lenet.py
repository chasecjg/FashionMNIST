import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import resnet50
from thop import profile
#Lenet网络搭建
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(1, 28, 28 output(16, 24, 24)
        x = self.pool1(x)            # output(16, 12, 12)
        x = F.relu(self.conv2(x))    # output(32, 8, 8)
        x = self.pool2(x)            # output(32, 4, 4)
        x = x.view(-1, 32*4*4)       # output(32*4*4)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x




model = LeNet()
input = torch.randn(1, 1, 28, 28)
flops, params = profile(model, inputs=(input,))
print("FLOPs:", flops)
print("params", params)