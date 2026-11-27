"""Cleaning pipeline for the cropped dataset: sha -> phash -> laplacian. Each step writes a new
full copy (output must not exist); inputs are never modified.

uv run scripts/clean.py all                      # data/cropped -> cleaned_sha -> cleaned_phash -> cleaned_final
uv run scripts/clean.py sha [--input --output]
uv run scripts/clean.py phash [--threshold 1] [--margin 5] [--input --output]
uv run scripts/clean.py laplacian [--threshold 40] [--input --output]   # no threshold: only writes laplacian_scores.csv

phash: greedy anchor grouping (no chaining), keeps the sharpest image per group, writes an audit JSON to
<output>_review/near_duplicate_groups.json. See data/DATA_CLEANING.md for results and limitations.
"""
import argparse
import csv
import hashlib
from pathlib import Path

import imagehash
from PIL import Image

from golfball.cleaning import all_images, parse_dirs, read_datasets, save_subset, sharpness, write_json

PHASH_THRESHOLD, BLUR_THRESHOLD = 1, 40  # defaults used by `all`; blur threshold is provisional


def sha(source, output):
    datasets = read_datasets(source)
    seen, removed, duplicates = {}, set(), []
    for name, rec in all_images(datasets):
        digest = hashlib.sha256((source / rec['file']).read_bytes()).hexdigest()
        if digest in seen:
            removed.add((name, rec['file']))
            duplicates.append({'removed': rec['file'], 'dataset': name,
                               'identical_to': seen[digest][1], 'sha256': digest})
        else:
            seen[digest] = (name, rec['file'])
    kept = save_subset(datasets, source, output, removed)
    write_json(output / 'sha256_duplicates.json', {'duplicates': duplicates})
    print(f'SHA-256: kept {kept}; removed {len(removed)} exact duplicates. Output: {output}')


def phash(source, output, threshold=PHASH_THRESHOLD, margin=5):
    if not 0 <= threshold <= 64:
        raise SystemExit('--threshold must be an integer from 0 through 64')
    datasets = read_datasets(source)
    entries = []
    for name, rec in all_images(datasets):
        with Image.open(source / rec['file']) as img:
            h = imagehash.phash(img)
        entries.append({'dataset': name, 'file': rec['file'], 'split': rec.get('split'), 'phash': str(h),
                        'hash_int': int(str(h), 16), 'sharpness': sharpness(source / rec['file'], margin)})
    groups, anchors = [], []  # greedy: join the FIRST anchor within threshold, never chain
    for e in entries:
        i = next((i for i, a in enumerate(anchors) if (e['hash_int'] ^ a).bit_count() <= threshold), None)
        if i is None:
            anchors.append(e['hash_int'])
            groups.append([e])
        else:
            groups[i].append(e)
    removed, report = set(), []
    for g in groups:
        if len(g) == 1:
            continue
        best = max(range(len(g)), key=lambda i: float('-inf') if g[i]['sharpness'] is None else g[i]['sharpness'])
        members = []
        for i, e in enumerate(g):
            if i != best:
                removed.add((e['dataset'], e['file']))
            members.append({'dataset': e['dataset'], 'file': e['file'], 'split': e['split'], 'phash': e['phash'],
                            'distance_to_anchor': (e['hash_int'] ^ g[0]['hash_int']).bit_count(),
                            'sharpness_score': e['sharpness'], 'kept': i == best})
        report.append({'id': f'group_{len(report) + 1:04d}', 'anchor_file': g[0]['file'], 'selected_index': best,
                       'selected_file': g[best]['file'], 'selected_sharpness_score': g[best]['sharpness'],
                       'members': members})
    kept = save_subset(datasets, source, output, removed)
    path = output.parent / f'{output.name}_review' / 'near_duplicate_groups.json'
    write_json(path, {'threshold': threshold, 'method': 'greedy_anchor_keep_sharpest', 'sharpness_margin': margin,
                      'total_input_images': len(entries), 'total_kept_images': kept,
                      'total_removed_images': len(removed), 'groups': report})
    print(f'pHash: {len(report)} near-duplicate groups, removed {len(removed)}, kept {kept} / {len(entries)}. '
          f'Output: {output}  audit: {path}')


def laplacian(source, output, threshold=None, margin=5):
    datasets = read_datasets(source)
    rows, removed = [], set()
    for name, rec in all_images(datasets):
        score = sharpness(source / rec['file'], margin)
        drop = threshold is not None and score is not None and score < threshold
        if drop:
            removed.add((name, rec['file']))
        rows.append({'dataset': name, 'file': rec['file'], 'split': rec.get('split'),
                     'blur_score': '' if score is None else f'{score:.5f}',
                     'decision': 'removed' if drop else 'kept'})
    kept = save_subset(datasets, source, output, removed)
    with (output / 'laplacian_scores.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'file', 'split', 'blur_score', 'decision'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: float(r['blur_score']) if r['blur_score'] else float('inf')))
    print(f'Laplacian: kept {kept}; removed {len(removed)} blurry images. Output: {output}')
    if threshold is None:
        print('No --threshold set; inspect laplacian_scores.csv to choose one.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', required=True)
    for step, inp, out in (('sha', 'cropped', 'cleaned_sha'), ('phash', 'cleaned_sha', 'cleaned_phash'),
                           ('laplacian', 'cleaned_phash', 'cleaned_final'), ('all', 'cropped', None)):
        s = sub.add_parser(step)
        s.add_argument('--input', default=f'data/{inp}')
        if out:
            s.add_argument('--output', default=f'data/{out}')
        if step in ('phash', 'all'):
            s.add_argument('--phash-threshold' if step == 'all' else '--threshold', dest='phash_t', type=int,
                           default=PHASH_THRESHOLD)
        if step in ('laplacian', 'all'):
            s.add_argument('--blur-threshold' if step == 'all' else '--threshold', dest='blur_t', type=float,
                           default=None if step == 'laplacian' else BLUR_THRESHOLD)
        if step != 'sha':
            s.add_argument('--margin', type=int, default=5, help='mask erosion radius (px) for sharpness')
    a = p.parse_args()
    if a.step == 'all':
        src = Path(a.input).expanduser().resolve()
        d = src.parent  # cleaned_* dirs go next to the input
        sha(src, d / 'cleaned_sha')
        phash(d / 'cleaned_sha', d / 'cleaned_phash', a.phash_t, a.margin)
        laplacian(d / 'cleaned_phash', d / 'cleaned_final', a.blur_t, a.margin)
        return
    src, out = parse_dirs(a)
    if a.step == 'sha':
        sha(src, out)
    elif a.step == 'phash':
        phash(src, out, a.phash_t, a.margin)
    else:
        laplacian(src, out, a.blur_t, a.margin)


if __name__ == '__main__':
    main()
