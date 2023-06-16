'''
Implementation of ID-CGAN generator network defined in the paper:
Image De-raining Using a Conditional Generative Adversarial Network
https://arxiv.org/pdf/1701.05957.pdf
'''

import torch
import torch.nn as nn

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('relu', nn.LeakyReLU(0.2))
        self.add_module('norm', nn.BatchNorm2d(growth_rate))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

    '''
    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(x)
            x = torch.cat([x, out], 1)
        return x

    def forward(self, x):
        c = x.size()[1]
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], 1)
        x = x[:, c, :]
        return x
    '''


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('upsample', nn.Upsample(scale_factor=2))

    def forward(self, x):
        return super().forward(x)


class TransitionNoSample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))

    def forward(self, x):
        return super().forward(x)




##############################
#           U-NET
##############################


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.cblp = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )

        self.dense1 = DenseBlock(64, 48, 4)
        self.tdown1 = TransitionDown(256, 128)

        self.dense2 = DenseBlock(128, 64, 6)
        self.tdown2 = TransitionDown(512, 256)

        self.dense3 = DenseBlock(256, 96, 8)
        self.tno3 = TransitionNoSample(1024, 512)

        self.dense4 = DenseBlock(512, 32, 8)
        self.tno4 = TransitionNoSample(768, 128)

        self.dense5 = DenseBlock(128+256, 64, 6)
        self.tup5 = TransitionUp(768, 120) #(640, 120)

        self.dense6 = DenseBlock(120+128, 34, 4)
        self.tup6 = TransitionUp(384, 64)

        self.dense7 = DenseBlock(64+64, 16, 4)
        self.tup7 = TransitionUp(192, 64)

        self.dense8 = DenseBlock(64, 16, 4)
        self.tno8 = TransitionNoSample(128, 16) # (32, 16)

        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        down1 = self.cblp(x)
        down2 = self.tdown1(self.dense1(down1))
        down3 = self.tdown2(self.dense2(down2))
        down4 = self.tno3(self.dense3(down3))
        down5 = self.tno4(self.dense4(down4))
        up6 = self.tup5(self.dense5(torch.cat([down5, down3], 1)))
        up7 = self.tup6(self.dense6(torch.cat([up6, down2], 1)))
        up8 = self.tup7(self.dense7(torch.cat([up7, down1], 1)))
        up9 = self.tno8(self.dense8(up8))

        return self.final(up9)