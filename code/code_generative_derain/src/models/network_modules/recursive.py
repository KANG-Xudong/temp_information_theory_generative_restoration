'''
Implementation of Recursive Module defined in the paper:
Lightweight Pyramid Networks for Image Deraining
https://arxiv.org/pdf/1805.06173.pdf
'''

import torch.nn as nn


class RecursiveBlock(nn.Sequential):
    def __init__(self, num_features):
        super().__init__()
        layers = [
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, dilation=1, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RecursiveModule(nn.Module):
    def __init__(self, in_size, num_recursive, num_features=16):
        super().__init__()
        self.num_recursive = num_recursive
        self.start_conv = nn.Conv2d(in_size, num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True)
        self.recur_block = RecursiveBlock(num_features)

    def forward(self, x):
        out_0 = self.start_conv(x)
        out = out_0
        for _ in range(self.num_recursive):
            out = self.recur_block(out)
            out += out_0
        return out