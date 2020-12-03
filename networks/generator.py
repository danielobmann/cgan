import torch
import torch.nn as nn
import numpy as np


class ConvBlockGenerator(torch.nn.Module):
    def __init__(self, inp_channels=1, out_channels=32, kernel_size=3, bn=False):
        super(ConvBlockGenerator, self).__init__()
        self.convblock = nn.ModuleList()
        self.convblock.append(nn.Conv2d(inp_channels, out_channels, kernel_size, padding=1))
        self.convblock.append(nn.PReLU())
        if bn:
            self.convblock.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        for layer in self.convblock:
            x = layer(x)
        return x


class LatentUpsampling(torch.nn.Module):
    def __init__(self, latentdim, steps, filters=32):
        super(LatentUpsampling, self).__init__()
        self.latentdim = latentdim
        self.N = int(np.sqrt(latentdim))
        self.steps = steps
        self.layers = nn.ModuleList()
        channels = filters
        for step in range(steps):
            if step == 0:
                self.layers.append(ConvBlockGenerator(inp_channels=1, out_channels=channels))
                self.layers.append(nn.ConvTranspose2d(channels, 2 * channels, kernel_size=2, stride=2))
                self.layers.append(nn.PReLU())
                channels = 2 * channels
            elif step == steps-1:
                self.layers.append(ConvBlockGenerator(inp_channels=channels, out_channels=1))
                self.layers.append(nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2))
                self.layers.append(nn.PReLU())
            else:
                self.layers.append(ConvBlockGenerator(inp_channels=channels, out_channels=channels))
                self.layers.append(nn.ConvTranspose2d(channels, 2 * channels, kernel_size=2, stride=2))
                self.layers.append(nn.PReLU())
                channels = 2*channels

    def forward(self, z):
        z = torch.reshape(z, [z.size(0), 1, self.N, self.N])
        for layer in self.layers:
            z = layer(z)
        return z


class Generator(torch.nn.Module):
    def __init__(self, latentdim=16, steps=4, zsteps=4, filters=32):
        super(Generator, self).__init__()
        self.latentupsampling = LatentUpsampling(latentdim=latentdim, steps=zsteps, filters=filters)
        channels = filters
        self.combined = nn.ModuleList()
        for step in range(steps):
            if step == 0:
                self.combined.append(ConvBlockGenerator(inp_channels=2, out_channels=channels))
            elif step == steps - 1:
                self.combined.append(ConvBlockGenerator(inp_channels=channels, out_channels=1))
            else:
                self.combined.append(ConvBlockGenerator(inp_channels=channels, out_channels=2 * channels))
                channels = 2 * channels

    def forward(self, z, y):
        z = self.latentupsampling(z)
        x = torch.cat([z, y], 1)
        for layer in self.combined:
            x = layer(x)
        return x
