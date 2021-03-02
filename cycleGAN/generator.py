import torch
import torch.nn as nn

from .blocks import ResNetBlock

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc:int=3, output_nc:int=3, ngf:int=64,
                n_blocks:int=9) -> None:
        
        assert(n_blocks >= 0)
        super(ResNetGenerator).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        n_downsmapling = 2
        for i in range(n_downsmapling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]

        mult = 2 ** n_downsmapling
        for i in range(n_blocks):
            mult = 2 ** i
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsmapling):
            mult = 2 ** (n_downsmapling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf* mult / 2),
                                    kernel_size=3, stride=2, padding=1,
                                    output_padding=1, bias=True),
                nn.InstanceNorm1d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input_):
        return self.model(input_)