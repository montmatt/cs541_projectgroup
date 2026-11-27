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
from pathlib import Path

import torch
import wandb

from golfball.engine import fit, get_device
from golfball.model import AE

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
    dev = get_device()
    model = AE(c.latent, c.base, c.norm, c.act).to(dev)
    run.summary["params"] = sum(p.numel() for p in model.parameters())
    run.summary["best_val_loss"] = fit(
        model, dev, epochs=c.epochs, lr=c.lr, weight_decay=c.weight_decay, loss=c.loss, batch_size=c.batch_size,
        augment=c.augment, cosine=True, ckpt=Path(CKPT) / f"{run.id}.pt", log=run.log)
    run.finish()
    del model  # sweep trials share one process: release GPU memory between runs
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
