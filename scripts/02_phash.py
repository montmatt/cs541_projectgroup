"""Automatically remove near-duplicate golf-ball crops using 64-bit pHash.

No prompts, previews, or human inspection. Images are assigned greedily to the
FIRST group anchor whose pHash lies within --threshold bits. This avoids
connected-component chaining: each member is within threshold of its anchor.

For each group, keep the image with the HIGHEST mask-aware variance-of-Laplacian
sharpness score. Ties keep the first. Drop the other members. Groups with
only one image are kept unchanged.

The sharpness score is computed within the eroded non-black ball region, not
across the artificial black mask edge. This is a heuristic, not a ground-truth
measure of image quality or duplicate identity.

Produces the filtered images and NDJSONs under --output, and an audit JSON
containing {"threshold": ..., "groups": [{"members": [...], ...}, ...]} in
<output>_review/near_duplicate_groups.json. The original input is untouched.

Run from project root:
    uv run python scripts/02_phash.py --input data/cleaned_sha \\
        --output data/cleaned_phash --threshold 2
"""
import argparse
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from cleaning_common import all_images, parse_dirs, read_datasets, save_subset, write_json


def build_groups(entries, threshold):
    """Group each image against the original anchors, never via a similarity chain."""
    groups = []
    anchors = []
    for entry in entries:
        match = next((i for i, anchor in enumerate(anchors)
                      if (entry['hash_int'] ^ anchor).bit_count() <= threshold), None)
        if match is None:
            anchors.append(entry['hash_int'])
            groups.append([entry])
        else:
            groups[match].append(entry)
    return groups


def sharpness_score(path, margin):
    """Variance of Laplacian on an eroded, non-black ball interior.

    Matches the scoring strategy in 03_laplacian.py. JPEG compression can
    introduce nonzero background pixels, so threshold intensity at 12 before
    erosion. Return None if there are fewer than 100 valid interior pixels.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not decode image: {path}')
    ball = ((img > 12).astype(np.uint8) * 255)
    k = 2 * margin + 1
    inner = cv2.erode(ball, np.ones((k, k), dtype=np.uint8), iterations=1) > 0
    if int(inner.sum()) < 100:
        return None
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap[inner].var())


def pick_sharpest(group):
    """Return first maximum-scoring member; None scores rank below valid ones."""
    winner = max(range(len(group)), key=lambda i: (
        group[i]['sharpness'] if group[i]['sharpness'] is not None else float('-inf')
    ))
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/cleaned_sha',
                        help='Input dataset directory containing NDJSONs and images')
    parser.add_argument('--output', default='data/cleaned_phash',
                        help='New output directory for automatically cleaned dataset')
    parser.add_argument('--threshold', type=int, default=2,
                        help='Maximum 64-bit pHash Hamming distance to group anchor (default: 2)')
    parser.add_argument('--margin', type=int, default=5,
                        help='Mask erosion radius in pixels for sharpness (default: 5)')
    args = parser.parse_args()
    if args.margin < 0:
        parser.error('--margin must be >= 0')
    if not 0 <= args.threshold <= 64:
        parser.error('--threshold must be an integer from 0 through 64')

    source, output = parse_dirs(args)
    if output.exists():
        parser.error(f'Output already exists: {output}. Use a different --output path.')
    datasets = read_datasets(source)
    entries = []
    for dataset_name, rec in all_images(datasets):
        with Image.open(source / rec['file']) as img:
            phash = imagehash.phash(img)
        entries.append({'dataset': dataset_name, 'file': rec['file'],
                        'split': rec.get('split'), 'phash': str(phash),
                        'hash_int': int(str(phash), 16),
                        'sharpness': sharpness_score(source / rec['file'], args.margin)})

    groups = build_groups(entries, args.threshold)
    removed = set()
    report_groups = []
    for group in groups:
        if len(group) == 1:
            continue
        chosen = pick_sharpest(group)
        anchor = group[0]['hash_int']
        members = []
        for i, entry in enumerate(group):
            distance = (entry['hash_int'] ^ anchor).bit_count()
            keep = i == chosen
            if not keep:
                removed.add((entry['dataset'], entry['file']))
            members.append({'dataset': entry['dataset'], 'file': entry['file'],
                            'split': entry['split'], 'phash': entry['phash'],
                            'distance_to_anchor': distance,
                            'sharpness_score': entry['sharpness'], 'kept': keep})
        report_groups.append({'id': f'group_{len(report_groups) + 1:04d}',
                              'anchor_file': group[0]['file'],
                              'selected_index': chosen,
                              'selected_file': group[chosen]['file'],
                              'selected_sharpness_score': group[chosen]['sharpness'],
                              'members': members})

    kept = save_subset(datasets, source, output, removed)
    report_dir = output.parent / f'{output.name}_review'
    report_path = report_dir / 'near_duplicate_groups.json'
    write_json(report_path, {'threshold': args.threshold,
                             'method': 'greedy_anchor_keep_sharpest',
                             'sharpness_margin': args.margin,
                             'total_input_images': len(entries),
                             'total_kept_images': kept,
                             'total_removed_images': len(removed),
                             'groups': report_groups})
    print(f'pHash automatic filtering: {len(report_groups)} near-duplicate groups, '
          f'removed {len(removed)}, kept {kept} / {len(entries)} images.')
    print(f'Cleaned dataset: {output}')
    print(f'Group dictionary: {report_path}')


if __name__ == '__main__':
    main()
