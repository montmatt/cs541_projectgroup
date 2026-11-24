"""Save a grid of normal test balls (top row) above the same balls with synthetic defects (bottom row).

uv run scripts/preview_defects.py [--n 8] [--out defects_preview.png]
"""
import argparse

import numpy as np
from PIL import Image

from golf_dataset import Defective, golf_balls

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=8)
p.add_argument("--out", default="defects_preview.png")
a = p.parse_args()

test = golf_balls("test")
bad = Defective(test)
idx = np.linspace(0, len(test) - 1, a.n).astype(int).tolist()  # spread across the split
row = lambda ds: np.hstack([(ds[i][0][0].numpy() * 255).astype("uint8") for i in idx])
Image.fromarray(np.vstack([row(test), row(bad)])).save(a.out)
print(f"saved {a.out}")
