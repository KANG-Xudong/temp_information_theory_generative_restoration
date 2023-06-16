'''
Module of Dilated Dense Block proposed in this study
'''

import torch
import torch.nn as nn


class DilatedLayer(nn.Sequential):
    def __init__(self, in_size, out_size, dilation, padding):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, dilation=dilation, padding=padding, bias=True),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        ]
        self.dilated_layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.dilated_layer(x)


class DilatedDenseLayer(nn.Sequential):
    def __init__(self, in_size, growth_rate):
        super().__init__()

        dilated_layers = []
        for d in [1, 3, 5]:
            dilated_layers.append(DilatedLayer(in_size, growth_rate, dilation=d, padding=d))
        self.dilated_layers = nn.ModuleList(dilated_layers)

    def forward(self, x):
        x1 = self.dilated_layers[0](x)
        x2 = self.dilated_layers[1](x)
        x3 = self.dilated_layers[2](x)
        x = torch.cat([x1, x2, x3], 1)
        return x


class DilatedDenseBlock(nn.Module):
    def __init__(self, in_size, num_layers, growth_rate=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [DilatedDenseLayer(in_size + (3 * i * growth_rate), growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1) # 1 = channel axis
        return x