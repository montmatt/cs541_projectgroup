"""Autoencoder dataset: cropped golf balls (all normal), circular-masked.

Each item is (img, mask), both [1,128,128]. img is zeroed outside the ball circle;
mask is 1 inside it. Compute the loss only where mask == 1.
Run directly for a self-check.
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from common import CROPPED, circle_mask, load_ndjson, open_image

NAMES = ["golf-ball-roboflow", "golf-ball-detection", "golf-ball"]  # all normal balls


class GolfBalls(Dataset):
    def __init__(self, name, split, augment=False):
        _, recs = load_ndjson(CROPPED / f"{name}.ndjson")
        imgs, masks = [], []
        for r in (r for r in recs if r["split"] == split):
            im = open_image(r, CROPPED).convert("L")
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


def loader(split, batch_size=64, **kw):
    train = split == "train"
    return DataLoader(golf_balls(split, augment=train), batch_size=batch_size, shuffle=train, **kw)


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        ds = golf_balls(s)
        img, mask = ds[0]
        assert img.shape == mask.shape == (1, 128, 128) and (img * (1 - mask)).sum() == 0
        print(s, len(ds), f"mask area {mask.mean():.2f}")
