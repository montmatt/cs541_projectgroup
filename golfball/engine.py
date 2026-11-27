"""Training + evaluation shared by scripts/train.py, evaluate.py and sweep.py."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from golfball.dataset import Defective, golf_balls


def get_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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


_cache = {}


def data(split, augment=False):  # sweep trials share a process: load images once
    if (split, augment) not in _cache:
        _cache[split, augment] = golf_balls(split, augment)
    return _cache[split, augment]


def fit(model, dev, *, epochs=50, lr=1e-3, weight_decay=0.0, loss="mse", batch_size=64, augment=True,
        cosine=False, patience=None, ckpt=None, log=print):
    """Masked-loss training with the best (lowest val MSE) weights saved to `ckpt` and reloaded at the end.
    log(dict) is called every epoch (train_loss, val_loss, lr). Returns the best val loss."""
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs) if cosine else None
    dist = (lambda a, b: (a - b) ** 2) if loss == "mse" else (lambda a, b: (a - b).abs())
    train = DataLoader(data("train", augment), batch_size, shuffle=True, drop_last=True)
    val = data("val")
    best, stale = float("inf"), 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, m in train:
            x, m = x.to(dev), m.to(dev)
            l = (dist(model(x), x) * m).sum() / m.sum()
            opt.zero_grad()
            l.backward()
            opt.step()
            tot, n = tot + l.item() * len(x), n + len(x)
        v = scores(model, val, dev)[0].mean().item()  # always MSE so runs with different train losses compare
        log({"epoch": ep, "train_loss": tot / n, "val_loss": v, "lr": opt.param_groups[0]["lr"]})
        if sched:
            sched.step()
        if v < best:
            best, stale = v, 0
            if ckpt:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
        elif patience and (stale := stale + 1) >= patience:
            break
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    return best
