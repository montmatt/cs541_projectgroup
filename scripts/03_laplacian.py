"""Detect blurry golf-ball images using mask-aware variance of Laplacian.

Scores ALL images. With --threshold, remove images with blur score BELOW threshold.
Without --threshold, copies all images and writes a CSV of scores for inspection.
Crops are grayscale and circular-masked; eroding the non-black ball region avoids
measuring the strong artificial circular boundary.

Examples:
 uv run python scripts/03_laplacian.py --input data/cleaned_phash --output data/cleaned_final
 uv run python scripts/03_laplacian.py --input data/cleaned_phash --output data/cleaned_final --threshold 40

Requires: opencv-python, numpy (uv add opencv-python numpy)
"""
import argparse
import csv

import cv2
import numpy as np

from cleaning_common import read_datasets, all_images, save_subset, parse_dirs


def blur_score(path, margin):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not decode image: {path}')
    # The round mask background is exactly black in image generation, but JPEG
    # compression can introduce a few dark pixels near the edge.
    ball = np.uint8(img > 12) * 255
    k = 2 * margin + 1
    inner = cv2.erode(ball, np.ones((k, k), dtype=np.uint8), iterations=1) > 0
    lap = cv2.Laplacian(img, cv2.CV_64F)
    if inner.sum() < 100:
        return None
    return float(lap[inner].var())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/cleaned_phash')
    parser.add_argument('--output', default='data/cleaned_final')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Remove images with score below this threshold; omit to score only')
    parser.add_argument('--margin', type=int, default=5, help='Mask erosion radius in pixels')
    args = parser.parse_args()
    if args.margin < 0:
        parser.error('--margin must be >= 0')
    source, output = parse_dirs(args)
    datasets = read_datasets(source)
    rows, removed = [], set()
    for dataset_name, rec in all_images(datasets):
        score = blur_score(source / rec['file'], args.margin)
        is_removed = args.threshold is not None and score is not None and score < args.threshold
        if is_removed:
            removed.add((dataset_name, rec['file']))
        rows.append({'dataset': dataset_name, 'file': rec['file'], 'split': rec.get('split'),
                     'blur_score': '' if score is None else f'{score:.5f}',
                     'decision': 'removed' if is_removed else 'kept'})
    kept = save_subset(datasets, source, output, removed)
    with (output / 'laplacian_scores.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'file', 'split', 'blur_score', 'decision'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: float(r['blur_score']) if r['blur_score'] else float('inf')))
    print(f'Laplacian: kept {kept}; removed {len(removed)} blurry images. Output: {output}')
    if args.threshold is None:
        print('No blur threshold set; inspect laplacian_scores.csv to choose one.')


if __name__ == '__main__':
    main()
