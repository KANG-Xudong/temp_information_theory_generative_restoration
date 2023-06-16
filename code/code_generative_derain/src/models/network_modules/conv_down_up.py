'''
Convolutional network layers for down-sampling using max-pooling and up-sampling using nearest-neighbor interpolations
'''

import torch.nn as nn


class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super(Down, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class Up(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

    def forward(self, x):
        return self.model(x)