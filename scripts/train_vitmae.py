"""Train Meta's MAE (models_mae.py, unchanged apart from 1-channel support) on normal golf balls.

Loss is MSE on masked patches, counted only inside the ball (background pixels ignored).
Saves the best-val model to checkpoints/ and a reconstruction grid per eval.

uv run scripts/train_vitmae.py --overfit     # sanity check: one batch, loss should -> ~0
uv run scripts/train_vitmae.py [--epochs 200] [--mask-ratio 0.75]
"""
import argparse
import math
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

from common import ROOT
from golf_dataset import golf_balls, loader
from models_mae import MaskedAutoencoderViT


def build_model():
    return MaskedAutoencoderViT(img_size=128, patch_size=8, in_chans=1,
                                embed_dim=512, depth=10, num_heads=8,
                                decoder_embed_dim=224, decoder_depth=2, decoder_num_heads=4)


def ball_loss(model, x, ball, mask_ratio):
    """Masked-patch MSE restricted to ball pixels. Returns (loss, pred, patch_mask)."""
    _, pred, mask = model(x, mask_ratio)            # Meta's loss ignored; recomputed below
    err = (pred - model.patchify(x)) ** 2           # [N, L, p*p]
    w = model.patchify(ball) * mask[..., None]      # inside ball AND in a masked patch
    return (err * w).sum() / w.sum().clamp(min=1), pred, mask


@contextmanager
def fixed_rng(dev):
    """Seed 0 inside the block, then restore the outer RNG, so eval uses the same masks every
    time without resetting training's shuffle/mask/augmentation randomness."""
    cpu = dev == "cpu"
    with torch.random.fork_rng(devices=[] if cpu else None, device_type="cuda" if cpu else dev):
        torch.manual_seed(0)
        yield


@torch.no_grad()
def evaluate(model, ds, dev, mask_ratio):
    model.eval()
    total = n = 0
    with fixed_rng(dev):  # same masks every eval so val loss is comparable across epochs
        for x, ball in torch.utils.data.DataLoader(ds, 64):
            x, ball = x.to(dev), ball.to(dev)
            total += ball_loss(model, x, ball, mask_ratio)[0].item() * len(x)
            n += len(x)
    return total / n


@torch.no_grad()
def save_grid(model, ds, dev, mask_ratio, path, k=8):
    """Rows: original / visible patches / reconstruction (visible patches + predicted ones)."""
    model.eval()
    x, ball = (torch.stack(t).to(dev) for t in zip(*(ds[i] for i in range(k))))
    with fixed_rng(dev):
        _, pred, mask = model(x, mask_ratio)
    m = model.unpatchify(mask[..., None].expand_as(pred))  # 1 = masked
    visible = x * (1 - m)
    recon = (visible + model.unpatchify(pred).clamp(0, 1) * m) * ball  # background is never trained
    rows = [torch.cat(list(r[:, 0]), dim=1) for r in (x, visible, recon)]
    Image.fromarray((torch.cat(rows).cpu().numpy() * 255).astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--warmup", type=int, default=10, help="warmup epochs")
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--patience", type=int, default=30, help="stop after this many evals without val improvement")
    p.add_argument("--overfit", action="store_true", help="train on one batch only (sanity check)")
    a = p.parse_args()

    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "checkpoints"
    out.mkdir(exist_ok=True)
    print(f"device {dev}")

    model = build_model().to(dev)
    print(f"params {sum(q.numel() for q in model.parameters() if q.requires_grad) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.95), weight_decay=0.05)

    if a.overfit:
        x, ball = next(iter(loader("train")))
        x, ball = x.to(dev), ball.to(dev)
        model.train()
        for step in range(1, 501):
            torch.manual_seed(0)  # fixed mask too, so it's a pure memorization test
            loss = ball_loss(model, x, ball, a.mask_ratio)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"step {step:4d}  loss {loss.item():.5f}", flush=True)
        return

    train, val = loader("train"), golf_balls("val")
    steps = a.epochs * len(train)
    warm = a.warmup * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))

    best, stale = float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = n = 0
        for x, ball in train:
            x, ball = x.to(dev), ball.to(dev)
            loss = ball_loss(model, x, ball, a.mask_ratio)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        v = evaluate(model, val, dev, a.mask_ratio)
        print(f"epoch {epoch:3d}  train {total / n:.5f}  val {v:.5f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if epoch % 10 == 0:
            save_grid(model, val, dev, a.mask_ratio, out / f"vitmae_recon_e{epoch:03d}.png")
        if v < best:
            best, stale = v, 0
            torch.save(model.state_dict(), out / "vitmae.pt")
        elif (stale := stale + 1) >= a.patience:
            print("early stop")
            break

    model.load_state_dict(torch.load(out / "vitmae.pt", map_location=dev))
    save_grid(model, val, dev, a.mask_ratio, out / "vitmae_recon_best.png")
    print(f"best val {best:.5f}  saved checkpoints/vitmae.pt")


if __name__ == "__main__":
    main()
