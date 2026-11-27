"""Train the autoencoder on normal balls, then evaluate (normal test balls vs synthetic defects).
Best (lowest val loss) weights go to checkpoints/ae_z<z>.pt (or --out).

uv run scripts/train.py [--z 128] [--epochs 50] [--patience 5]
uv run scripts/train.py --config configs/best.json     # retrain a saved config (best.json = sweep recipe; baseline.json = defaults)
Other flags: --base --norm --act --loss --lr --weight-decay --batch-size --no-augment --cosine
"""
import argparse
import json

import torch

from golfball.engine import evaluate, fit, get_device
from golfball.model import AE
from golfball.paths import ROOT

p = argparse.ArgumentParser()
p.add_argument("--config", help="JSON with z/base/norm/act/loss/lr/weight_decay/batch_size/augment/epochs (+ optional cosine/patience)")
p.add_argument("--z", type=int, default=128, help="bottleneck size")
p.add_argument("--base", type=int, default=32)
p.add_argument("--norm", default="batch", choices=["batch", "none"])
p.add_argument("--act", default="relu", choices=["relu", "leaky"])
p.add_argument("--loss", default="mse", choices=["mse", "l1"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=0.0)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--no-augment", action="store_true")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--patience", type=int, default=5, help="early-stop after this many epochs without val improvement (0 = off)")
p.add_argument("--cosine", action="store_true", help="cosine lr schedule")
p.add_argument("--size", type=int, default=3, choices=range(11), help="synthetic defect size 0-10 for the final evaluation")
p.add_argument("--out", help="checkpoint path (default checkpoints/ae_z<z>.pt)")
a = p.parse_args()
if a.config:  # cosine/patience default to the sweep recipe (cosine, no early stop)
    c = json.load(open(a.config))
    a.z, a.base, a.norm, a.act, a.loss = c["z"], c["base"], c["norm"], c["act"], c["loss"]
    a.lr, a.weight_decay, a.batch_size, a.epochs = c["lr"], c["weight_decay"], c["batch_size"], c["epochs"]
    a.no_augment, a.cosine, a.patience = not c["augment"], c.get("cosine", True), c.get("patience", 0)

torch.manual_seed(0)
dev = get_device()
ckpt = ROOT / (a.out or f"checkpoints/ae_z{a.z}.pt")
model = AE(a.z, a.base, a.norm, a.act).to(dev)
best = fit(model, dev, epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay, loss=a.loss,
           batch_size=a.batch_size, augment=not a.no_augment, cosine=a.cosine, patience=a.patience, ckpt=ckpt,
           log=lambda d: print(f"epoch {d['epoch']:3d}  train {d['train_loss']:.5f}  val {d['val_loss']:.5f}", flush=True))
print(f"\nsaved {ckpt.relative_to(ROOT)}  (best val {best:.5f})")
evaluate(model, dev, a.size)
