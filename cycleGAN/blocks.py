import torch
import torch.nn as nn

from .layers import build_conv_block

class ResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super(ResNetBlock).__init__()
        self.conv_block = build_conv_block(dim)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
