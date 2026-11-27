"""Evaluate a saved autoencoder on normal test balls vs the same balls with synthetic defects.

uv run scripts/evaluate.py checkpoints/ae_z128.pt [--z 128] [--base 32] [--norm batch] [--act relu]
uv run scripts/evaluate.py checkpoints/best.pt --config configs/best.json   # model flags from a config
The model flags must match how the checkpoint was trained (sweep runs: see the run's config).
"""
import argparse
import json

import torch

from golfball.engine import evaluate, get_device
from golfball.model import AE

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
