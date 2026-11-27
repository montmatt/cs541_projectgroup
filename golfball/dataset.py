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

from golfball.paths import ROOT, circle_mask, load_ndjson, open_image
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


def add_defect(img, mask, rng=random, size=3):
    """Paint 3-6 heavy synthetic damage marks inside the ball: scratches, jagged cracks, gouges,
    noisy scuff patches and large broken-off chunks (dark or bright).
    ponytail: stand-in for real damaged balls; swap in real defect photos when the team has them."""
    shape, scale = img.shape[1:], size / 5  # size 0-10 scales every mark; 5 = the original (heavy) look
    sc = lambda v: max(1, round(v * scale))
    ys, xs = mask[0].nonzero(as_tuple=True)
    pt = lambda: (xs[(k := rng.randrange(len(xs)))].item(), ys[k].item())  # random point inside the ball
    for _ in range(rng.randint(3, 6)):
        canvas = Image.new("L", shape)
        d = ImageDraw.Draw(canvas)
        kind = rng.choice(["scratch", "crack", "gouge", "scuff", "chunk"])
        if kind == "scratch":  # stays inside since the circle is convex
            d.line([pt(), pt()], fill=255, width=sc(rng.randint(3, 6)))
        elif kind == "crack":  # jagged random walk
            x, y = pt()
            for _ in range(rng.randint(6, 10)):
                nx, ny = x + sc(rng.randint(-14, 14)), y + sc(rng.randint(-14, 14))
                d.line([(x, y), (nx, ny)], fill=255, width=sc(rng.randint(2, 4)))
                x, y = nx, ny
        else:
            x, y = pt()
            r = sc({"gouge": rng.randint(8, 20), "scuff": rng.randint(12, 26), "chunk": rng.randint(20, 36)}[kind])
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        m = torch.from_numpy(np.asarray(canvas, dtype="float32") / 255)[None] * mask
        if kind == "scuff":  # grey, grainy patch
            fill = (0.5 + 0.25 * torch.randn(img.shape, generator=None)).clamp(0, 1)
        else:
            fill = rng.choice([0.05, 0.95]) if kind != "chunk" else 0.0  # broken-off chunk is a hole
        img = img * (1 - m) + fill * m
    return img, mask


class Defective(Dataset):
    """Wraps a dataset of normal balls; item i gets a synthetic defect (same one every time)."""
    def __init__(self, ds, size=3):  # size 0-10: defect scale (5 = original look)
        self.ds, self.size = ds, size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return add_defect(*self.ds[i], random.Random(i), self.size)


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
