import torch
import torchvision
from torch import nn, optim
from torch.nn import Linear
from torch.utils.data import DataLoader


class DepthModel(nn.Module):
    # __init__ 构造函数
    def __init__(self ):
        super(DepthModel, self).__init__()
        self.fc = nn.Linear(1, 10)

    def forward(self, x):
        return self.fc(x)
