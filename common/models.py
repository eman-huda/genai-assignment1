"""Convolutional denoising autoencoder used in Task 1 (and as the specialist architecture in Task 2)."""
import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class DownBlock(nn.Module):
    """Halves spatial size (strided conv), changes channels."""

    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.net = nn.Sequential(*conv_bn_relu(cin, cout, 2), *conv_bn_relu(cout, cout), nn.Dropout2d(dropout))

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """Doubles spatial size (bilinear upsample + conv, avoids checkerboard artefacts)."""

    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.net = nn.Sequential(*conv_bn_relu(cin, cout), *conv_bn_relu(cout, cout), nn.Dropout2d(dropout))

    def forward(self, x):
        return self.net(self.up(x))


class ConvAutoencoder(nn.Module):
    """128x128 RGB -> latent (latent_channels x 8 x 8) -> 128x128 RGB.

    Encoder: 128 -> 64 -> 32 -> 16 -> 8 with channels c, 2c, 4c, 8c, 8c.
    Bottleneck: 1x1 conv to `latent_channels` (latent dim = latent_channels * 64).
    skip_channels > 0 adds ONE narrow skip at 16x16 (projected to skip_channels),
    used only for the skip-connection ablation. No full-resolution skips.
    """

    def __init__(self, base_channels=32, latent_channels=32, dropout=0.1, skip_channels=0):
        super().__init__()
        c, d = base_channels, dropout
        self.latent_channels = latent_channels
        self.skip_channels = skip_channels
        self.stem = nn.Sequential(*conv_bn_relu(3, c), *conv_bn_relu(c, c))   # 128
        self.down1 = DownBlock(c, 2 * c, d)                                   # 64
        self.down2 = DownBlock(2 * c, 4 * c, d)                               # 32
        self.down3 = DownBlock(4 * c, 8 * c, d)                               # 16
        self.down4 = DownBlock(8 * c, 8 * c, d)                               # 8
        self.to_latent = nn.Conv2d(8 * c, latent_channels, 1)                 # bottleneck z
        self.from_latent = nn.Sequential(nn.Conv2d(latent_channels, 8 * c, 1), nn.BatchNorm2d(8 * c),
                                         nn.ReLU(inplace=True))
        self.up1 = UpBlock(8 * c, 8 * c, d)                                   # 16
        if skip_channels > 0:
            self.skip_proj = nn.Sequential(nn.Conv2d(8 * c, skip_channels, 1), nn.ReLU(inplace=True))
        self.up2 = UpBlock(8 * c + skip_channels, 4 * c, d)                   # 32
        self.up3 = UpBlock(4 * c, 2 * c, d)                                   # 64
        self.up4 = UpBlock(2 * c, c, d)                                       # 128
        self.head = nn.Sequential(nn.Conv2d(c, 3, 3, padding=1), nn.Sigmoid())

    def encode(self, x):
        h16 = self.down3(self.down2(self.down1(self.stem(x))))
        z = self.to_latent(self.down4(h16))
        return z, h16

    def decode(self, z, h16=None):
        h = self.up1(self.from_latent(z))
        if self.skip_channels > 0:
            h = torch.cat([h, self.skip_proj(h16)], dim=1)
        return self.head(self.up4(self.up3(self.up2(h))))

    def forward(self, x):
        z, h16 = self.encode(x)
        return self.decode(z, h16)

    def latent_dim(self, img_size=128):
        return self.latent_channels * (img_size // 16) ** 2


def count_params(model):
    return sum(p.numel() for p in model.parameters())
