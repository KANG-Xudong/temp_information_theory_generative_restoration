'''
Implementation of Residual Module used in the paper:
Removing Rain from Single Images via a Deep Detail Network
https://openaccess.thecvf.com/content_cvpr_2017/papers/Fu_Removing_Rain_From_CVPR_2017_paper.pdf
'''

import torch.nn as nn


class BasicLayer(nn.Sequential):
    def __init__(self, in_size, out_size):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BasicBlock(nn.Sequential):
    def __init__(self, num_features):
        super().__init__()
        self.conv_1 = BasicLayer(num_features, num_features)
        self.conv_2 = BasicLayer(num_features, num_features)

    def forward(self, x):
        identity = x
        out = self.conv_1(x)
        out = self.conv_2(out)
        out += identity
        return out


class ResidualModule(nn.Module):
    def __init__(self, in_size, num_layers, num_features=16):
        super().__init__()
        self.start_conv = BasicLayer(in_size, num_features)
        self.res_layers = nn.Sequential(*[BasicBlock(num_features) for _ in range(num_layers)])

    def forward(self, x):
        x = self.start_conv(x)
        x = self.res_layers(x)
        return x