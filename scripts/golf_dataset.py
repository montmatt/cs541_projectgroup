"""Autoencoder dataset: cropped golf balls (all normal), circular-masked.

Each item is (img, mask), both [1,128,128]. img is zeroed outside the ball circle;
mask is 1 inside it. Compute the loss only where mask == 1.
Run directly for a self-check.
"""
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from common import ROOT, circle_mask, load_ndjson, open_image
CLEANED = ROOT / "data" / "cleaned_final"

NAMES = ["golf-ball-roboflow", "golf-ball-detection", "golf-ball"]  # all normal balls


class GolfBalls(Dataset):
    def __init__(self, name, split, augment=False):
        _, recs = load_ndjson(CLEANED / f"{name}.ndjson")
        imgs, masks = [], []
        for r in (r for r in recs if r["split"] == split):
            im = open_image(r, CLEANED).convert("L")
            imgs.append(torch.from_numpy(np.asarray(im, dtype="float32") / 255)[None])
            masks.append(torch.from_numpy(circle_mask(r["annotations"]["boxes"][0], im.width)).float()[None])
        self.imgs, self.masks, self.augment = torch.stack(imgs), torch.stack(masks), augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img, mask = self.imgs[i], self.masks[i]
        if self.augment:  # mild only: flips + brightness, nothing that mimics damage
            for d in (1, 2):
                if torch.rand(()) < 0.5:
                    img, mask = img.flip(d), mask.flip(d)
            img = (img * (0.9 + 0.2 * torch.rand(()))).clamp(0, 1)
        return img * mask, mask


def golf_balls(split, augment=False):
    return ConcatDataset([GolfBalls(n, split, augment) for n in NAMES])


def add_defect(img, mask, rng=random):
    """Paint 1-3 synthetic scratches/scuffs inside the ball (dark gouge or bright scuff).
    ponytail: stand-in for real damaged balls; swap in real defect photos when the team has them."""
    canvas = Image.new("L", img.shape[1:])
    d = ImageDraw.Draw(canvas)
    ys, xs = mask[0].nonzero(as_tuple=True)
    for _ in range(rng.randint(1, 3)):
        i, j = rng.randrange(len(xs)), rng.randrange(len(xs))
        x, y = xs[i].item(), ys[i].item()
        if rng.random() < 0.5:  # scratch; stays inside since the circle is convex
            d.line([(x, y), (xs[j].item(), ys[j].item())], fill=255, width=rng.randint(1, 3))
        else:  # scuff
            r = rng.randint(4, 12)
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    m = torch.from_numpy(np.asarray(canvas, dtype="float32") / 255)[None] * mask
    return img * (1 - m) + rng.choice([0.1, 0.9]) * m, mask


class Defective(Dataset):
    """Wraps a dataset of normal balls; item i gets a synthetic defect (same one every time)."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return add_defect(*self.ds[i], random.Random(i))


def loader(split, batch_size=64, **kw):
    train = split == "train"
    return DataLoader(golf_balls(split, augment=train), batch_size=batch_size, shuffle=train, **kw)


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        ds = golf_balls(s)
        img, mask = ds[0]
        assert img.shape == mask.shape == (1, 128, 128) and (img * (1 - mask)).sum() == 0
        bad, _ = Defective(ds)[0]
        assert (bad * (1 - mask)).sum() == 0 and (bad != img).any()
        print(s, len(ds), f"mask area {mask.mean():.2f}")
