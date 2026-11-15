"""Convert Roboflow COCO exports to the Ultralytics NDJSON format used by
golf-ball-detection.ndjson and golf-ball.ndjson.

Target format (one JSON object per line):
    line 1:  {"type":"dataset","task":"detect","name":...,"class_names":{"0":...},...}
    line 2+: {"type":"image","file":...,"width":W,"height":H,"split":"train|val|test",
            "annotations":{"boxes":[[cls, x_center, y_center, w, h], ...]}}   # normalized 0-1

Run: uv run scripts/coco_to_ndjson.py
"""
import json
from collections import defaultdict
from pathlib import Path

from common import RAW

ORIGINAL = RAW / "golf-ball-roboflow-original"
OUTPUT = RAW / "golf-ball-roboflow.ndjson"

# (split, directory)
SPLITS = [("train", "train"), ("val", "valid"), ("test", "test")]

HEADER = {
    "type": "dataset",
    "task": "detect",
    "name": "Golf Ball (Roboflow)",
    "description": "Golf ball detection dataset converted from Roboflow COCO export.",
    "url": "https://universe.roboflow.com/young-jin-park-6vhep/golf-ball",
    "class_names": {"0": "Golf-ball"},
    "version": "latest",
}

def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def convert_boxes(anns, width, height):
    boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        boxes.append([
            0,
            round((x0 + w / 2) / width, 5),
            round((y0 + h / 2) / height, 5),
            round(w / width, 5),
            round(h / height, 5),
        ])
    return boxes

def main():
    n_img = n_box = 0
    missing = []
    with open(OUTPUT, "w", encoding="utf-8") as out:
        out.write(dumps(HEADER) + "\n")
        for split, folder in SPLITS:
            split_dir = ORIGINAL / folder
            coco = json.loads((split_dir / "_annotations.coco.json").read_text())
            by_image = defaultdict(list)
            for a in coco["annotations"]:
                by_image[a["image_id"]].append(a)
            for img in coco["images"]:
                image_path = split_dir / img["file_name"]
                if not image_path.exists():
                    missing.append(image_path)
                    continue
                boxes = convert_boxes(by_image[img["id"]], img["width"], img["height"])
                out.write(dumps({
                    "type": "image",
                    # Relative path to the JSON's folder
                    "file": image_path.relative_to(RAW).as_posix(),
                    "width": img["width"],
                    "height": img["height"],
                    "split": split,
                    "annotations": {"boxes": boxes},
                }) + "\n")
                n_img += 1
                n_box += len(boxes)

    print(f"Wrote {OUTPUT}: {n_img} images, {n_box} boxes")
    if missing:
        raise SystemExit(f"{len(missing)} images listed in COCO but not on disk, e.g. {missing[0]}")

if __name__ == "__main__":
    main()