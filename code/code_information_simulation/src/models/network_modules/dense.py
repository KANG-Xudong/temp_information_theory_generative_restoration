'''
Module of DenseNet Block defined in the paper:
Densely Connected Convolutional Networks
https://arxiv.org/pdf/1608.06993.pdf
'''

import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_size, out_size):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self, in_size, num_layers, growth_rate=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [DenseLayer(in_size + (i * growth_rate), growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1) # 1 = channel axis
        return x