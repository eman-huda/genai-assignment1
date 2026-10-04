"""Style-conditioned pix2pix (Task 4): U-Net generator + PatchGAN discriminator.

The style condition is a learned nn.Embedding (3 styles). Its vector is expanded to a
spatial map and concatenated to the input of BOTH networks (the 'style-vector expansion'
used by FSGAN, the FS2K authors' model), and again at the generator bottleneck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def style_map(embed, style, h, w):
    e = embed(style)                                  # (N, E)
    return e[:, :, None, None].expand(-1, -1, h, w)


def _down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin, cout, dropout=0.0):
    layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class StyleUNetGenerator(nn.Module):
    """128x128 U-Net with 7 down / 7 up stages (pix2pix 'unet_128'), skip connections at every scale."""

    def __init__(self, base=64, emb_dim=8, dropout=0.5, n_styles=3, in_ch=3, out_ch=3):
        super().__init__()
        c, e = base, emb_dim
        self.embed = nn.Embedding(n_styles, e)
        self.d1 = _down(in_ch + e, c, norm=False)   # 64
        self.d2 = _down(c, 2 * c)                   # 32
        self.d3 = _down(2 * c, 4 * c)               # 16
        self.d4 = _down(4 * c, 8 * c)               # 8
        self.d5 = _down(8 * c, 8 * c)               # 4
        self.d6 = _down(8 * c, 8 * c)               # 2
        self.d7 = _down(8 * c, 8 * c, norm=False)   # 1 (no BatchNorm on a 1x1 map)
        self.u1 = _up(8 * c + e, 8 * c, dropout)    # 2   (style injected again at the bottleneck)
        self.u2 = _up(16 * c, 8 * c, dropout)       # 4
        self.u3 = _up(16 * c, 8 * c, dropout)       # 8
        self.u4 = _up(16 * c, 4 * c)                # 16
        self.u5 = _up(8 * c, 2 * c)                 # 32
        self.u6 = _up(4 * c, c)                     # 64
        self.out = nn.Sequential(nn.ConvTranspose2d(2 * c, out_ch, 4, 2, 1), nn.Tanh())   # 128

    def forward(self, x, style):
        h, w = x.shape[-2:]
        d1 = self.d1(torch.cat([x, style_map(self.embed, style, h, w)], 1))
        d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6)
        u = self.u1(torch.cat([d7, style_map(self.embed, style, d7.shape[-2], d7.shape[-1])], 1))
        u = self.u2(torch.cat([u, d6], 1))
        u = self.u3(torch.cat([u, d5], 1))
        u = self.u4(torch.cat([u, d4], 1))
        u = self.u5(torch.cat([u, d3], 1))
        u = self.u6(torch.cat([u, d2], 1))
        return self.out(torch.cat([u, d1], 1))


class StylePatchDiscriminator(nn.Module):
    """70x70 PatchGAN on (photo, sketch, style map). For 128x128 input it outputs a 14x14 logit map."""

    def __init__(self, base=64, emb_dim=8, n_styles=3, photo_ch=3, sketch_ch=3):
        super().__init__()
        c = base
        self.embed = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            nn.Conv2d(photo_ch + sketch_ch + emb_dim, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(2 * c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1, bias=False), nn.BatchNorm2d(4 * c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * c, 8 * c, 4, 1, 1, bias=False), nn.BatchNorm2d(8 * c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8 * c, 1, 4, 1, 1))

    def forward(self, photo, sketch, style):
        h, w = photo.shape[-2:]
        return self.net(torch.cat([photo, sketch, style_map(self.embed, style, h, w)], 1))


def init_weights(m):
    """pix2pix initialisation: N(0, 0.02) for convs, N(1, 0.02) for BatchNorm scale."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 1.0)


def build_models(cfg):
    G = StyleUNetGenerator(cfg["base_channels"], cfg["style_emb_dim"], cfg["dropout"])
    D = StylePatchDiscriminator(cfg["base_channels"], cfg["style_emb_dim"])
    G.apply(init_weights); D.apply(init_weights)
    return G, D


def d_losses(D, x, y, y_fake, s):
    """Discriminator BCE-with-logits, real and fake terms returned separately."""
    real, fake = D(x, y, s), D(x, y_fake.detach(), s)
    l_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
    l_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
    acc_real = (real > 0).float().mean()
    acc_fake = (fake < 0).float().mean()
    return l_real, l_fake, acc_real, acc_fake


def g_losses(D, x, y, y_fake, s):
    """Generator adversarial (non-saturating BCE) and L1 terms. L_G = adv + lambda_L1 * L1."""
    pred = D(x, y_fake, s)
    adv = F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred))
    return adv, F.l1_loss(y_fake, y)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
