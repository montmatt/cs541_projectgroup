"""Train the convolutional autoencoder on normal balls, then evaluate: normal test balls vs
the same balls with synthetic defects. Saves the best (lowest val loss) model to checkpoints/.

uv run scripts/train.py [--z 128] [--epochs 50] [--patience 5]
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import ROOT
from golf_dataset import Defective, golf_balls, loader


def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())


def up(i, o):
    return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())


class AE(nn.Module):
    def __init__(self, z=128):
        super().__init__()
        self.enc = nn.Sequential(down(1, 32), down(32, 64), down(64, 128), down(128, 256),  # 128 -> 8
                                 nn.Flatten(), nn.Linear(256 * 8 * 8, z))
        self.dec = nn.Sequential(nn.Linear(z, 256 * 8 * 8), nn.Unflatten(1, (256, 8, 8)),
                                 up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def score(x, y, m):
    """Per-image MSE over ball pixels only (normalized by ball area)."""
    return ((x - y) ** 2 * m).sum((1, 2, 3)) / m.sum((1, 2, 3))


@torch.no_grad()
def scores(model, ds, dev):
    """Per-image (mean, peak) error. mean = score() over the ball; peak = max of the 9x9-blurred
    error map, which catches small defects (thin scratches) that the mean dilutes."""
    model.eval()
    mean, peak = [], []
    for x, m in DataLoader(ds, 64):
        x, m = x.to(dev), m.to(dev)
        e = (model(x) - x) ** 2 * m
        mean.append((e.sum((1, 2, 3)) / m.sum((1, 2, 3))).cpu())
        peak.append(F.avg_pool2d(e, 9, 1, 4).amax((1, 2, 3)).cpu())
    return torch.cat(mean), torch.cat(peak)


def auroc(normal, defect):
    """P(a defective ball scores higher than a normal one); ties count half."""
    d = defect[:, None] - normal[None]
    return ((d > 0).float() + 0.5 * (d == 0).float()).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", type=int, default=128, help="bottleneck size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5, help="stop after this many epochs without val improvement")
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / "checkpoints" / f"ae_z{a.z}.pt"
    ckpt.parent.mkdir(exist_ok=True)

    model = AE(a.z).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    train, val = loader("train"), golf_balls("val")

    best, stale = float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = n = 0
        for x, m in train:
            x, m = x.to(dev), m.to(dev)
            loss = score(model(x), x, m).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(x), n + len(x)
        v = scores(model, val, dev)[0].mean().item()
        print(f"epoch {epoch:3d}  train {total / n:.5f}  val {v:.5f}", flush=True)
        if v < best:
            best, stale = v, 0
            torch.save(model.state_dict(), ckpt)
        elif (stale := stale + 1) >= a.patience:
            break

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    test = golf_balls("test")
    sv, s0, s1 = scores(model, val, dev), scores(model, test, dev), scores(model, Defective(test), dev)
    print(f"\nsaved {ckpt.relative_to(ROOT)}  (best val {best:.5f})")
    for i, name in enumerate(("mean", "peak")):
        thr = sv[i].quantile(0.95)  # val is all normal: targets a 5% false-alarm rate
        print(f"{name}: test AUROC {auroc(s0[i], s1[i]):.3f}  |  threshold {thr:.5f}: "
              f"defects caught {(s1[i] > thr).float().mean():.2f}, normals flagged {(s0[i] > thr).float().mean():.2f}")


if __name__ == "__main__":
    main()
