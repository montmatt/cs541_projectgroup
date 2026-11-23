# Golf Ball Cleaning Pipeline

Install dependencies once:

```bash
uv add pillow ImageHash numpy opencv-python
```

Then run from the repository root:

```bash
uv run python scripts/01_sha256.py --input data/cropped --output data/cleaned_sha
uv run python scripts/02_phash.py --input data/cleaned_sha --output data/cleaned_phash --threshold 4 --review
uv run python scripts/03_laplacian.py --input data/cleaned_phash --output data/cleaned_final
```

The final command reports blur scores without deleting anything. Review `data/cleaned_final/laplacian_scores.csv`, then run it again with **a different output directory** and your chosen threshold, for example:

```bash
uv run python scripts/03_laplacian.py --input data/cleaned_phash --output data/cleaned_final_threshold40 --threshold 40
```

`02_phash.py` generates `data/cleaned_phash_review/near_duplicate_groups.json` with this structure:

```json
{
  "threshold": 4,
  "groups": [
    {
      "id": "group_0001",
      "members": [
        {"dataset": "...ndjson", "file": "...jpg", "split": "train", "phash": "..."},
        {"dataset": "...ndjson", "file": "...jpg", "split": "train", "phash": "..."}
      ]
    }
  ]
}
```

`groups` is a **list of near-duplicate subsets inside a dictionary**. Each group has a corresponding contact sheet in `data/cleaned_phash_review/previews/`. When `--review` is provided, enter the indices of images you want to *keep* in each group (e.g. `0,2`). Press Enter to keep all. To reuse the saved decisions instead of being prompted, run without `--review`, with `--decisions data/cleaned_phash_review/decisions.json` and a different `--output` folder.

**Notes:**
- These scripts preserve image-relative paths, dataset NDJSON headers, annotations and original splits, and do not edit input images.
- Run only on **cropped** images; your teammate's `MIN_SIZE=32` has already been applied.
- pHash is intentionally conservative: similar masked golf balls are not necessarily duplicate photos. pHash connected-component groups may contain chains of similar images.
- Laplacian threshold `40` above is merely an example, not a validated threshold. The score is computed in the interior of the ball to avoid its artificial mask boundary.
- The current `golf_dataset.py` needs its `CROPPED` references made configurable to load the desired output directory (both the NDJSON file and image paths).
- If output already exists, scripts stop instead of accidentally mixing new results with an old run.
