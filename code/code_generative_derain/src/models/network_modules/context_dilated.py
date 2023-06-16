'''
Implementation of Contextualized Dilated Module defined in the paper:
Deep Joint Rain Detection and Removal from a Single Image
https://arxiv.org/pdf/1609.07769.pdf
'''

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


class ContextualizedDilatedBlock(nn.Sequential):
    def __init__(self, in_size):
        super().__init__()

        dilated_layers = []
        for d in [1, 2, 3]:
            current_dilated_path_layers = [
                DilatedLayer(in_size, in_size, dilation=d, padding=d) for _ in range(2)
            ]
            dilated_layers.append(nn.Sequential(*current_dilated_path_layers))
        self.dilated_layers = nn.ModuleList(dilated_layers)

    def forward(self, x):
        x1 = self.dilated_layers[0](x)
        x2 = self.dilated_layers[1](x)
        x3 = self.dilated_layers[2](x)
        x = x + x1 + x2 + x3
        return x


class ContextualizedDilatedModule(nn.Module):
    def __init__(self, in_size, num_layers):
        super().__init__()
        layers = [ContextualizedDilatedBlock(in_size) for _ in range(num_layers)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x