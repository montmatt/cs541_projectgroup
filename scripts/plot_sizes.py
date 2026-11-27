"""Plot test AUROC vs synthetic defect size for the baseline and the sweep's best model.

uv run scripts/plot_sizes.py [--sizes 1 2 3 4 5] [--out docs/figures/auroc_vs_size.png]
Models: checkpoints/ae_z32.pt (configs/baseline.json) and checkpoints/best.pt (configs/best.json). Edit MODELS to add more.
"""
import argparse
import json

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from golfball.dataset import Defective, golf_balls
from golfball.engine import auroc, get_device, scores
from golfball.model import AE
from golfball.paths import ROOT

MODELS = {  # name -> (checkpoint, config json or None for AE defaults)
    "baseline (z=32)": ("checkpoints/ae_z32.pt", "configs/baseline.json"),
    "sweep best": ("checkpoints/best.pt", "configs/best.json"),
}

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
p.add_argument("--out", default=str(ROOT / "docs" / "figures" / "auroc_vs_size.png"))
a = p.parse_args()

dev = get_device()
test = golf_balls("test")
bad = {s: Defective(test, s) for s in a.sizes}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for name, (ckpt, cfg) in MODELS.items():
    c = json.load(open(ROOT / cfg)) if cfg else {}
    model = AE(c.get("z", 128), c.get("base", 32), c.get("norm", "batch"), c.get("act", "relu")).to(dev)
    model.load_state_dict(torch.load(ROOT / ckpt, map_location=dev))
    s0 = scores(model, test, dev)
    res = [[auroc(s0[i], scores(model, bad[s], dev)[i]) for s in a.sizes] for i in (0, 1)]
    for ax, r in zip(axes, res):
        ax.plot(a.sizes, r, marker="o", label=name)
    print(f"{name:18} " + "  ".join(f"s{s}: mean {res[0][j]:.3f} peak {res[1][j]:.3f}" for j, s in enumerate(a.sizes)))
for ax, t in zip(axes, ("anomaly score: mean error", "anomaly score: peak error")):
    ax.set(title=t, xlabel="defect size", xticks=a.sizes)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("test AUROC (higher = better detection)")
axes[0].legend()
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(f"saved {a.out}")
