import torch
from torch import nn
from torchvision.models import resnet

class ResNetCelebA(nn.Module):
    def __init__(self):
        super(ResNetCelebA, self).__init__()
        net = resnet.resnet18()
        self.base = torch.nn.Sequential(*list(net.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nf = 512
        self.fc = nn.Linear(self.nf, 40)
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        features = self.base(x)
        features = self.avg_pool(features)
        features = features.view(-1, self.nf)
        classes = self.sigm(self.fc(features))
        return classes
