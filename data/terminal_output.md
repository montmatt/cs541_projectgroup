```bash
cs541_project % uv run python scripts/02_phash.py \
    --input data/cleaned_sha \
    --output data/cleaned_phash \
    --threshold 2
pHash automatic filtering: 1295 near-duplicate groups, removed 1695, kept 2578 / 4273 images.
Cleaned dataset: /Users/<user>/GitHub/cs541_project/data/cleaned_phash
Group dictionary: /Users/<user>/GitHub/cs541_project/data/cleaned_phash_review/near_duplicate_groups.json
```

```bash
matthewmontero@autoreg-6828698 cs541_project % uv run python scripts/03_laplacian.py \
    --input data/cleaned_phash \
    --output data/cleaned_final \
    --threshold 40
Laplacian: kept 2250; removed 328 blurry images. Output: /Users/<user>/GitHub/cs541_project/data/cleaned_final
```