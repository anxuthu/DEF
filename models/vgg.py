"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torchvision.transforms as transforms

__all__ = [
    'vgg16', 'vgg19',
    'vgg16bn', 'vgg19bn',
]


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(v)]
            layers += [nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg16(num_classes=10):
    return VGG(num_classes=num_classes, depth=16, batch_norm=False)
def vgg19(num_classes=10):
    return VGG(num_classes=num_classes, depth=19, batch_norm=False)

def vgg16bn(num_classes=10):
    return VGG(num_classes=num_classes, depth=16, batch_norm=True)
def vgg19bn(num_classes=10):
    return VGG(num_classes=num_classes, depth=19, batch_norm=True)
