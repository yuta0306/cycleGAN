import torch
import torch.nn as nn

def build_conv_block(dim: int) -> nn.Sequential:
    conv_block = list()
    conv_block += [nn.ReflectionPad2d(1)]
    conv_block += [
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.InstanceNorm2d(dim),
        nn.ReLU(inplace=True)
    ]
    conv_block += [nn.ReflectionPad2d(1)]
    conv_block += [
        nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
        nn.InstanceNorm2d(dim)
    ]

    return nn.Sequential(*conv_block)