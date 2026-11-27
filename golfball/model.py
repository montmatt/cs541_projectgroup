"""Convolutional autoencoder used by train.py, sweep.py and evaluate.py.

Defaults match the saved checkpoints/ae_z*.pt (base=32, batch norm, ReLU), so they load as-is.
"""
import torch.nn as nn


class AE(nn.Module):
    """128x128 -> 4 stride-2 convs -> 8x8 -> linear bottleneck (z) -> mirror."""

    def __init__(self, z=128, base=32, norm="batch", act="relu"):
        super().__init__()
        A = (lambda: nn.ReLU()) if act == "relu" else (lambda: nn.LeakyReLU(0.2))
        N = (lambda c: nn.BatchNorm2d(c)) if norm == "batch" else (lambda c: nn.Identity())
        c = [1] + [base * 2**i for i in range(4)]  # channels per stage
        self.enc = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(c[i], c[i + 1], 4, 2, 1), N(c[i + 1]), A()) for i in range(4)],
            nn.Flatten(), nn.Linear(c[4] * 64, z))
        self.dec = nn.Sequential(
            nn.Linear(z, c[4] * 64), nn.Unflatten(1, (c[4], 8, 8)),
            *[nn.Sequential(nn.ConvTranspose2d(c[i + 1], c[i], 4, 2, 1), N(c[i]), A()) for i in (3, 2, 1)],
            nn.ConvTranspose2d(c[1], 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))
