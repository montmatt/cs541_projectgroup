"""Shared NDJSON + sharpness utilities for the cleaning pipeline (scripts/clean.py).

The input directory should have <dataset>.ndjson at its root and images at paths
relative to that directory, e.g. golf-ball/train/image_0.jpg.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_datasets(root: Path):
    datasets = []
    for ndjson_path in sorted(root.glob('*.ndjson')):
        with ndjson_path.open(encoding='utf-8') as file:
            records = [json.loads(line) for line in file if line.strip()]
        if not records:
            raise ValueError(f'Empty NDJSON: {ndjson_path}')
        header, images = records[0], [r for r in records[1:] if r.get('type') == 'image']
        for record in images:
            path = root / record['file']
            if not path.is_file():
                raise FileNotFoundError(f'Missing image: {path}')
        datasets.append((ndjson_path.name, header, images))
    if not datasets:
        raise FileNotFoundError(f'No *.ndjson files in {root}')
    return datasets


def all_images(datasets):
    return [(name, rec) for name, _header, images in datasets for rec in images]


def save_subset(datasets, source: Path, output: Path, removed: set):
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError('Output must differ from input; original files are preserved.')
    if source in output.parents:
        raise ValueError('Output cannot be inside input (avoids accidental recursive data).')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}. Choose a new output path.')
    output.mkdir(parents=True)
    kept = 0
    for name, header, images in datasets:
        dest_ndjson = output / name
        with dest_ndjson.open('w', encoding='utf-8') as file:
            file.write(json.dumps(header, ensure_ascii=False) + '\n')
            for rec in images:
                key = (name, rec['file'])
                if key in removed:
                    continue
                file.write(json.dumps(rec, ensure_ascii=False) + '\n')
                dest = output / rec['file']
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / rec['file'], dest)
                kept += 1
    return kept


def parse_dirs(args):
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    return source, output


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sharpness(path, margin=5):
    """Variance of Laplacian on the eroded non-black ball interior (avoids the artificial mask edge).
    JPEG can leave nonzero background pixels, so intensity is thresholded at 12 before erosion.
    Returns None if fewer than 100 interior pixels remain."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not decode image: {path}')
    k = 2 * margin + 1
    inner = cv2.erode(np.uint8(img > 12) * 255, np.ones((k, k), dtype=np.uint8)) > 0
    if inner.sum() < 100:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F)[inner].var())
