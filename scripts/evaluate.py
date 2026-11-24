"""Evaluate an autoencoder on normal test balls vs the same balls with synthetic defects.

uv run scripts/evaluate.py checkpoints/ae_z128.pt [--z 128] [--base 32] [--norm batch] [--act relu]
uv run scripts/evaluate.py checkpoints/best.pt --config configs/best.json   # model flags from a config
The model flags must match how the checkpoint was trained (sweep runs: see the run's config).
"""
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from golf_dataset import Defective, golf_balls
from model import AE


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


def evaluate(model, dev):
    """Print test AUROC and detection rates; thresholds come from the (all-normal) val split."""
    val, test = golf_balls("val"), golf_balls("test")
    sv, s0, s1 = scores(model, val, dev), scores(model, test, dev), scores(model, Defective(test), dev)
    for i, name in enumerate(("mean", "peak")):
        thr = sv[i].quantile(0.95)  # targets a 5% false-alarm rate
        print(f"{name}: test AUROC {auroc(s0[i], s1[i]):.3f}  |  threshold {thr:.5f}: "
              f"defects caught {(s1[i] > thr).float().mean():.2f}, normals flagged {(s0[i] > thr).float().mean():.2f}")


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--z", type=int, default=128)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--norm", default="batch", choices=["batch", "none"])
    p.add_argument("--act", default="relu", choices=["relu", "leaky"])
    p.add_argument("--config", help="JSON with z/base/norm/act (e.g. configs/best.json); overrides those flags")
    a = p.parse_args()
    if a.config:
        c = json.load(open(a.config))
        a.z, a.base, a.norm, a.act = c["z"], c["base"], c["norm"], c["act"]
    dev = get_device()
    model = AE(a.z, a.base, a.norm, a.act).to(dev)
    model.load_state_dict(torch.load(a.checkpoint, map_location=dev))
    evaluate(model, dev)
