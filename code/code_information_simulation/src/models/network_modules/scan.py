'''
Implementation of SCAN Module proposed in the paper:
Recurrent Squeeze-and-Excitation Context Aggregation Net for Single Image Deraining
https://openaccess.thecvf.com/content_ECCV_2018/papers/Xia_Li_Recurrent_Squeeze-and-Excitation_Context_ECCV_2018_paper.pdf
'''

from torch import nn


class SEBlock(nn.Module):
    def __init__(self, in_size, reduction=6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_size, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicLayer(nn.Module):
    def __init__(self, in_size, out_size, kernel, dilation):
        super().__init__()
        pad = int(dilation * (kernel - 1) / 2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel, stride=1, dilation=dilation, padding=pad, bias=True)
        self.se = SEBlock(out_size)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.se(x))
        return x


class SCANModule(nn.Module):
    def __init__(self, in_size, num_layers, kernel=3, num_features=24):
        super().__init__()
        layers = [BasicLayer(in_size, num_features, kernel, dilation=1)] + [
            BasicLayer(num_features, num_features, kernel, dilation=2 ** i) for i in range(num_layers - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x