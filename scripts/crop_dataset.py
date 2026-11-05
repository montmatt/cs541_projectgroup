"""Crop every bounding box (one crop per ball) to a square around the box, resized to
OUT_SIZE x OUT_SIZE. Skips boxes whose shorter side is under MIN_SIZE px (in the
original image) and images with no boxes.

Writes data/cropped/<name>/<split>/*.jpg and data/cropped/<name>.ndjson (same format;
the box is the ball's position inside the square crop, so preview_image.py still works).
Run: uv run scripts/crop_dataset.py [name ...]
"""
import json
import sys
from pathlib import Path

from PIL import Image

from common import CROPPED, RAW, load_ndjson, open_image

MIN_SIZE = 32  # px, shorter side of the original box; smaller balls have almost no texture
OUT_SIZE = 128  # px, every output image is OUT_SIZE x OUT_SIZE
SOURCES = ["golf-ball-roboflow", "golf-ball-detection", "golf-ball"]


def main():
    for name in sys.argv[1:] or SOURCES:
        header, images = load_ndjson(RAW / f"{name}.ndjson")
        header = {**header, "name": header["name"] + " (cropped)"}
        out_dir = CROPPED / name
        kept = skipped = failed = 0
        with open(CROPPED / f"{name}.ndjson", "w", encoding="utf-8") as out:
            out.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
            for rec in images:
                boxes = rec.get("annotations", {}).get("boxes", [])
                if not boxes:
                    continue
                try:
                    img = open_image(rec, RAW).convert("L")  # grayscale
                except SystemExit as e:
                    failed += 1
                    print(e)
                    continue
                w, h = img.size
                for i, (cls, xc, yc, bw, bh) in enumerate(boxes):
                    x0, y0 = (xc - bw / 2) * w, (yc - bh / 2) * h
                    bpx, bpy = bw * w, bh * h
                    if min(bpx, bpy) < MIN_SIZE:
                        skipped += 1
                        continue
                    # Square window around the box center, shifted to stay inside the image
                    # (capped at the image's short side, which can clip a huge ball).
                    side = min(max(bpx, bpy), w, h)
                    sx = min(max(xc * w - side / 2, 0), w - side)
                    sy = min(max(yc * h - side / 2, 0), h - side)
                    crop = img.crop((round(sx), round(sy), round(sx + side), round(sy + side)))
                    crop = crop.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                    # This ball's box inside the square crop, normalized and clipped to 0-1.
                    # Other balls that happen to fall in the window are not labeled.
                    bx0, by0 = max((x0 - sx) / side, 0), max((y0 - sy) / side, 0)
                    bx1, by1 = min((x0 + bpx - sx) / side, 1), min((y0 + bpy - sy) / side, 1)
                    box = [cls, round((bx0 + bx1) / 2, 5), round((by0 + by1) / 2, 5),
                           round(bx1 - bx0, 5), round(by1 - by0, 5)]
                    rel = Path(name) / rec["split"] / f"{Path(rec['file']).stem}_{i}.jpg"
                    (CROPPED / rel).parent.mkdir(parents=True, exist_ok=True)
                    crop.save(CROPPED / rel, quality=95)
                    out.write(json.dumps({"type": "image", "file": rel.as_posix(),
                                          "width": crop.width, "height": crop.height,
                                          "split": rec["split"],
                                          "annotations": {"boxes": [box]}},
                                         ensure_ascii=False, separators=(",", ":")) + "\n")
                    kept += 1
        print(f"{name}: kept {kept} crops, skipped {skipped} balls under {MIN_SIZE}px, failed {failed} images")


if __name__ == "__main__":
    main()
