"""Autoencoder hyperparameter sweep, logged to Weights & Biases.

Objective: masked reconstruction loss on the (all-normal) val split -> minimize `val_loss`.
Always MSE-on-mask for val so runs with different train losses stay comparable.

  uv run scripts/sweep.py                 # create a bayes sweep + run 30 trials
  uv run scripts/sweep.py -n 50           # more trials
  uv run scripts/sweep.py --sweep <id>    # join an existing sweep (e.g. from a teammate's machine)
  uv run scripts/sweep.py --smoke         # 1 quick epoch, no wandb upload
"""
import argparse
import gc
import os

import torch
import wandb
from torch.utils.data import DataLoader

from golf_dataset import golf_balls
from model import AE

ENTITY = "neelawsare-worcester-polytechnic-institute"
PROJECT = "golf-ball-defect-detection"
CKPT = "checkpoints/sweep"

SWEEP = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "early_terminate": {"type": "hyperband", "min_iter": 5},
    "parameters": {
        "latent": {"values": [16, 32, 64, 128, 256]},
        "base": {"values": [16, 32, 64]},  # channels of first conv; doubles each stage
        "norm": {"values": ["batch", "none"]},
        "act": {"values": ["relu", "leaky"]},
        "loss": {"values": ["mse", "l1"]},
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-3},
        "batch_size": {"values": [32, 64, 128]},
        "augment": {"values": [True, False]},
        "epochs": {"value": 40},
    },
}


def masked(err, mask):
    return (err * mask).sum() / mask.sum()


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    tot = n = 0
    for x, m in dl:
        x, m = x.to(dev), m.to(dev)
        tot += masked((model(x) - x) ** 2, m).item() * len(x)
        n += len(x)
    return tot / n


_cache = {}


def data(split, augment):  # sweep runs share a process: load images once
    k = (split, augment)
    if k not in _cache:
        _cache[k] = golf_balls(split, augment)
    return _cache[k]


def train(smoke=False):
    try:
        _train(smoke)
    except Exception as e:
        if type(e) is Exception and not e.args:
            # wandb's agent kills a run's thread with a bare Exception when the sweep stops it
            # (Hyperband early termination): a normal end, not a failure.
            print("run stopped by sweep (early termination)", flush=True)
            wandb.finish()
            return
        print(f"TRAIN FAILED: {type(e).__name__}: {e}", flush=True)
        raise


def _train(smoke=False):
    cfg = {k: v.get("value", v.get("values", [None])[0]) for k, v in SWEEP["parameters"].items()}
    run = wandb.init(config={**cfg, "epochs": 1, "lr": 1e-3, "weight_decay": 1e-5} if smoke else None)
    c = run.config
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AE(c.latent, c.base, c.norm, c.act).to(dev)
    opt = torch.optim.AdamW(model.parameters(), c.lr, weight_decay=c.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.epochs)
    dist = (lambda a, b: (a - b) ** 2) if c.loss == "mse" else (lambda a, b: (a - b).abs())
    tr = DataLoader(data("train", c.augment), c.batch_size, shuffle=True, drop_last=True)
    va = DataLoader(data("val", False), 128)
    run.summary["params"] = sum(p.numel() for p in model.parameters())
    best = float("inf")
    for ep in range(c.epochs):
        model.train()
        tot = n = 0
        for x, m in tr:
            x, m = x.to(dev), m.to(dev)
            loss = masked(dist(model(x), x), m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        val = evaluate(model, va, dev)
        run.log({"epoch": ep, "train_loss": tot / n, "val_loss": val, "lr": sched.get_last_lr()[0]})
        if val < best:
            best = val
            os.makedirs(CKPT, exist_ok=True)
            torch.save(model.state_dict(), f"{CKPT}/{run.id}.pt")
    run.summary["best_val_loss"] = best
    run.finish()
    del model, opt, sched, tr, va  # sweep trials share one process: release GPU memory between runs
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30, help="number of trials this agent runs")
    ap.add_argument("--sweep", help="existing sweep id to join")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        os.environ["WANDB_MODE"] = "offline"
        train(smoke=True)
    else:
        sid = a.sweep or wandb.sweep(SWEEP, entity=ENTITY, project=PROJECT)
        wandb.agent(sid, train, entity=ENTITY, project=PROJECT, count=a.n)
