# Golf Ball Cleaning Pipeline

Full write-up (thresholds, results, limitations): [`data/DATA_CLEANING.md`](../data/DATA_CLEANING.md).

Dependencies (`ImageHash`, `opencv-python`, ...) are already in `pyproject.toml`; run `uv sync`. From the repository root:

```bash
uv run python scripts/01_sha256.py --input data/cropped --output data/cleaned_sha
uv run python scripts/02_phash.py --input data/cleaned_sha --output data/cleaned_phash --threshold 1
uv run python scripts/03_laplacian.py --input data/cleaned_phash --output data/cleaned_final --threshold 40
```

- `01_sha256.py` removes byte-identical files.
- `02_phash.py` is fully automatic (no prompts or review). It greedily groups images whose 64-bit pHash is within `--threshold` bits of a group's first image and keeps the sharpest one per group. The audit JSON goes to `<output>_review/near_duplicate_groups.json`. Threshold 1 is used for `cleaned_final` (2 removed too many images).
- `03_laplacian.py` scores blur (variance of Laplacian inside the ball). Without `--threshold` it only copies everything and writes `laplacian_scores.csv` for inspection; with `--threshold 40` it drops images scoring below 40. 40 is a provisional heuristic, not a validated cutoff.

**Notes:**
- The scripts preserve image-relative paths, dataset NDJSON headers, annotations and original splits, and never edit input images.
- Run only on **cropped** images; `MIN_SIZE=32` was already applied by `crop_dataset.py`.
- pHash is intentionally conservative: similar masked golf balls are not necessarily duplicate photos.
- `scripts/golf_dataset.py` reads from `data/cleaned_final` (its `CLEANED` constant); change that to load a different output.
- If an output directory already exists, the scripts stop instead of mixing new results with an old run.
