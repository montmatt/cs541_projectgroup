"""Visual previews.

uv run scripts/preview.py image data/raw/<file>.ndjson [index|name-substring] [--save out.png]
    Draw an image's boxes; random image if no selector. Prefers the local file, falls back to the URL.
uv run scripts/preview.py defects [--n 8] [--out outputs/defects_preview.png]
    Grid of normal test balls (top row) over the same balls with synthetic defects (bottom row).
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from golfball.paths import ROOT, load_ndjson, open_image


def draw_boxes(img, rec, class_names):
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size  # use the real size; boxes are normalized
    thickness = max(2, round(min(w, h) / 200))
    for cls, xc, yc, bw, bh in rec["annotations"]["boxes"]:
        x0, y0 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x1, y1 = (xc + bw / 2) * w, (yc + bh / 2) * h
        color = ["#00e676", "#ff1744", "#2979ff", "#ffea00"][int(cls) % 4]
        d.rectangle([x0, y0, x1, y1], outline=color, width=thickness)
        d.text((x0 + 3, max(0, y0 - 12)), class_names.get(str(int(cls)), str(int(cls))), fill=color)
    return img


def image(a):
    header, images = load_ndjson(a.ndjson)
    if a.selector is None:
        rec = random.choice(images)
    elif a.selector.isdigit():
        rec = images[int(a.selector)]
    else:
        matches = [r for r in images if a.selector in r["file"]]
        if not matches:
            raise SystemExit(f"No file name contains {a.selector!r}")
        rec = matches[0]
    img = draw_boxes(open_image(rec, a.ndjson.parent), rec, header.get("class_names", {}))
    print(f"{rec['file']}  split={rec['split']}  size={img.size}  boxes={len(rec['annotations']['boxes'])}")
    img.save(a.save) if a.save else img.show()


def defects(a):
    from golfball.dataset import Defective, golf_balls  # imports torch; keep out of `image`
    test = golf_balls("test")
    bad = Defective(test)
    idx = np.linspace(0, len(test) - 1, a.n).astype(int).tolist()  # spread across the split
    row = lambda ds: np.hstack([(ds[i][0][0].numpy() * 255).astype("uint8") for i in idx])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([row(test), row(bad)])).save(out)
    print(f"saved {out}")


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("image")
s.add_argument("ndjson", type=Path)
s.add_argument("selector", nargs="?", help="image index or part of a file name (default: random)")
s.add_argument("--save", type=Path, help="save to this path instead of opening a viewer")
s.set_defaults(fn=image)
s = sub.add_parser("defects")
s.add_argument("--n", type=int, default=8)
s.add_argument("--out", default=str(ROOT / "outputs" / "defects_preview.png"))
s.set_defaults(fn=defects)
a = p.parse_args()
a.fn(a)
