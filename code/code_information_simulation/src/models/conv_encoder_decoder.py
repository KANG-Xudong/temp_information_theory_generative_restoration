'''
Convolutional encoder-decoder network (En/Decoder)
using max-pooling as the down-sampling and nearest-neighbor interpolations as the up-sampling processes.
'''


import torch.nn as nn
from .network_modules.conv_down_up import Down, Up

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 16)

        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u5 = self.up1(d4)
        u6 = self.up2(u5)
        u7 = self.up3(u6)
        u8 = self.up4(u7)

        return self.final(u8)