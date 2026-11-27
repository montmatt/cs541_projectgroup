# Golf Ball Cleaning Pipeline

Full write-up (thresholds, results, limitations): [`data/DATA_CLEANING.md`](../data/DATA_CLEANING.md).

Dependencies (`ImageHash`, `opencv-python`, ...) are already in `pyproject.toml`; run `uv sync`. From the repository root:

```bash
uv run scripts/clean.py all     # cropped -> cleaned_sha -> cleaned_phash -> cleaned_final (threshold 1, 40)
```

Or step by step (all of `--input`/`--output` are optional, shown are the defaults' equivalents):

```bash
uv run scripts/clean.py sha       --input data/cropped       --output data/cleaned_sha
uv run scripts/clean.py phash     --input data/cleaned_sha   --output data/cleaned_phash --threshold 1
uv run scripts/clean.py laplacian --input data/cleaned_phash --output data/cleaned_final --threshold 40
```

- `sha` removes byte-identical files.
- `phash` is fully automatic (no prompts or review). It greedily groups images whose 64-bit pHash is within `--threshold` bits of a group's first image and keeps the sharpest one per group. The audit JSON goes to `<output>_review/near_duplicate_groups.json`. Threshold 1 is used for `cleaned_final` (2 removed too many images).
- `laplacian` scores blur (variance of Laplacian inside the ball). Without `--threshold` it only copies everything and writes `laplacian_scores.csv` for inspection; with `--threshold 40` it drops images scoring below 40. 40 is a provisional heuristic, not a validated cutoff.

Shared helpers (NDJSON reading, subset copying, the sharpness score) are in `golfball/cleaning.py`.

**Notes:**
- The scripts preserve image-relative paths, dataset NDJSON headers, annotations and original splits, and never edit input images.
- Run only on **cropped** images; `MIN_SIZE=32` was already applied by `scripts/data/crop_dataset.py`.
- pHash is intentionally conservative: similar masked golf balls are not necessarily duplicate photos.
- `golfball/dataset.py` reads from `data/cleaned_final` (its `CLEANED` constant); change that to load a different output.
- If an output directory already exists, the scripts stop instead of mixing new results with an old run.
