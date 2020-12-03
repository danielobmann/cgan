import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(self, inp_channels=32, out_channels=32, kernel_size=2, dropout=0, bn=False):
        super(ConvBlock, self).__init__()
        self.convblock = nn.ModuleList()
        self.convblock.append(nn.Conv2d(inp_channels, out_channels, 3, padding=1))
        self.convblock.append(nn.PReLU())
        if dropout > 0:
            self.convblock.append(nn.Dropout(dropout))
        if bn:
            self.convblock.append(nn.BatchNorm2d(out_channels))
        self.convblock.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.convblock.append(nn.PReLU())
        if dropout > 0:
            self.convblock.append(nn.Dropout(dropout))
        if bn:
            self.convblock.append(nn.BatchNorm2d(out_channels))
        self.convblock.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride=kernel_size))
        self.convblock.append(nn.PReLU())
        if dropout > 0:
            self.convblock.append(nn.Dropout(dropout))
        if bn:
            self.convblock.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        for layer in self.convblock:
            x = layer(x)
        return x


class PatchDiscriminator(torch.nn.Module):
    """
    PatchGAN discriminator implementation for better training of GANs.

    """
    def __init__(
        self,
        filter=64,
        steps=3
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = filter
        for l in range(steps):
            if l == 0:
                self.layers.append(ConvBlock(inp_channels=1, out_channels=channels))
            elif l == steps-1:
                self.layers.append(ConvBlock(inp_channels=channels, out_channels=1))
            else:
                self.layers.append(ConvBlock(inp_channels=channels, out_channels=channels*2))
                channels = channels*2
        self.layers.append(nn.Sigmoid())

    def forward(self, x, y):
        z = x - y
        for layer in self.layers:
            z = layer(z)
        return z

