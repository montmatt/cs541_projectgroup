"""Train the convolutional autoencoder on normal balls, then evaluate: normal test balls vs
the same balls with synthetic defects. Saves the best (lowest val loss) model to checkpoints/.

uv run scripts/train.py [--z 128] [--epochs 50] [--patience 5]
"""
import argparse

import torch

from common import ROOT
from golf_dataset import golf_balls, loader
from model import AE
from evaluate import evaluate, get_device, score, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", type=int, default=128, help="bottleneck size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5, help="stop after this many epochs without val improvement")
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    torch.manual_seed(0)
    dev = get_device()
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
    print(f"\nsaved {ckpt.relative_to(ROOT)}  (best val {best:.5f})")
    evaluate(model, dev)


if __name__ == "__main__":
    main()
