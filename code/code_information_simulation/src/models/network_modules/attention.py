'''
Implementation of Attentive Recurrent Module used in the paper:
Attentive Generative Adversarial Network for Raindrop Removal from A Single Image
https://arxiv.org/pdf/1711.10098.pdf
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Sequential):
    def __init__(self, num_features=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x
        out = self.layers(x)
        out += identity
        out = F.relu(out)
        return out


class AttentiveRecurrentModule(nn.Module):
    def __init__(self, in_size, num_residual=5, num_recurrent=5, num_features=32):
        super().__init__()
        self.num_recurrent = num_recurrent
        self.num_features = num_features
        self.start_conv = nn.Sequential(
            nn.Conv2d((in_size + 1), num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.res_layers = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_residual)])
        self.conv_i = nn.Sequential(
            nn.Conv2d((num_features * 2), num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d((num_features * 2), num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d((num_features * 2), num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d((num_features * 2), num_features, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(num_features, 1, kernel_size=3, stride=1, dilation=1, padding=1, bias=True),
        )

    def forward(self, input):
        batch_size, _, row, col = input.size()
        mask = torch.ones(batch_size, 1, row, col) / 2.
        h = torch.zeros(batch_size, self.num_features, row, col)
        c = torch.zeros(batch_size, self.num_features, row, col)
        mask, h, c = mask.to(input.device), h.to(input.device), c.to(input.device)
        for i in range(self.num_recurrent):
            x = torch.cat((input, mask), 1)
            x = self.start_conv(x)
            x = self.res_layers(x)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
        output = torch.cat((input, mask), 1)
        return output