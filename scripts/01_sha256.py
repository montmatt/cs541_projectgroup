"""Remove byte-identical golf-ball images, preserving NDJSON metadata and folders.

Example: uv run python scripts/01_sha256.py --input data/cropped --output data/cleaned_sha
"""
import argparse
import hashlib
from pathlib import Path
from cleaning_common import read_datasets, all_images, save_subset, parse_dirs, write_json


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/cropped')
    parser.add_argument('--output', default='data/cleaned_sha')
    args = parser.parse_args()
    source, output = parse_dirs(args)
    datasets = read_datasets(source)
    seen = {}
    removed = set()
    duplicates = []
    for name, record in all_images(datasets):
        key = (name, record['file'])
        digest = sha256(source / record['file'])
        if digest in seen:
            removed.add(key)
            duplicates.append({'removed': record['file'], 'dataset': name,
                               'identical_to': seen[digest][1], 'sha256': digest})
        else:
            seen[digest] = key
    kept = save_subset(datasets, source, output, removed)
    write_json(output / 'sha256_duplicates.json', {'duplicates': duplicates})
    print(f'SHA-256: kept {kept}; removed {len(removed)} exact duplicates. Output: {output}')


if __name__ == '__main__':
    main()
