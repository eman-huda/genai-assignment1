"""Task 2 corruption classifier (also the gating network initialisation in Task 3).

Input: RGB tensor in [0, 1], shape (N, 3, 128, 128). Output: 4 logits in CONDITIONS order
(clean, salt_pepper, gaussian_blur, occlusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_CONFIGS = {"small": (16, 32, 64, 128), "medium": (32, 64, 128, 256),
                   "wide": (48, 96, 192, 256), "deep_narrow": (32, 64, 96, 128)}


class CorruptionClassifier(nn.Module):
    """Four conv stages. The first stage runs at full resolution before any pooling, so fine
    detail (isolated noise pixels, the edge sharpness that blur removes) is still visible.
    Global average + max pooling, dropout, then a linear layer to 4 logits."""

    def __init__(self, channels=(32, 64, 128, 256), dropout=0.3, n_classes=4):
        super().__init__()
        layers, cin = [], 3
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(cin, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * cin, n_classes))

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([F.adaptive_avg_pool2d(f, 1), F.adaptive_max_pool2d(f, 1)], 1).flatten(1)
        return self.head(f)
