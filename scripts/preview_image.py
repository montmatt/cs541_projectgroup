"""Preview an image from any of the project's NDJSON files with its boxes drawn.

Usage:
    uv run scripts/preview_image.py data/raw/golf-ball-roboflow.ndjson            # random image
    uv run scripts/preview_image.py data/raw/golf-ball-roboflow.ndjson 5          # 6th image (0-based index)
    uv run scripts/preview_image.py data/raw/golf-ball.ndjson LINE_ALBUM          # first file name containing text
    uv run scripts/preview_image.py data/raw/golf-ball.ndjson 3 --save out.png    # save instead of opening a viewer
"""
import argparse
import random
from pathlib import Path

from PIL import ImageDraw

from common import load_ndjson, open_image


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


def main():
    ap = argparse.ArgumentParser(description="Preview an image from an NDJSON dataset file.")
    ap.add_argument("ndjson", type=Path)
    ap.add_argument("selector", nargs="?", help="image index or part of a file name (default: random)")
    ap.add_argument("--save", type=Path, help="save to this path instead of opening a viewer")
    args = ap.parse_args()

    header, images = load_ndjson(args.ndjson)
    if args.selector is None:
        rec = random.choice(images)
    elif args.selector.isdigit():
        rec = images[int(args.selector)]
    else:
        matches = [r for r in images if args.selector in r["file"]]
        if not matches:
            raise SystemExit(f"No file name contains {args.selector!r}")
        rec = matches[0]

    img = draw_boxes(open_image(rec, args.ndjson.parent), rec, header.get("class_names", {}))
    print(f"{rec['file']}  split={rec['split']}  size={img.size}  boxes={len(rec['annotations']['boxes'])}")
    if args.save:
        img.save(args.save)
    else:
        img.show()


if __name__ == "__main__":
    main()