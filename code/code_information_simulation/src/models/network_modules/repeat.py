'''
Module to simply repeat the channels of inputs for multiple times
'''

import torch
import torch.nn as nn


class RepeatModule(nn.Module):
    def __init__(self, in_size, num_repeat):
        super().__init__()
        self.num_repeat = num_repeat
        self.out_size = in_size * num_repeat


    def forward(self, x):
        x = torch.cat([x for _ in range(self.num_repeat)], 1) # 1 = channel axis
        return x