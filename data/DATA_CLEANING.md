# Golf Ball Dataset Cleaning

This document describes the preprocessing and data-cleaning pipeline used to prepare the golf ball image dataset for training a masked autoencoder (MAE) for anomaly detection. It summarizes the filtering methods, selected heuristics, and final dataset statistics.

## 1. Overview

The original cropped dataset contained **4,320 golf ball images** from three sources:

- `golf-ball-roboflow`
- `golf-ball-detection`
- `golf-ball`

The goal of cleaning was to reduce redundant images and remove low-quality examples while retaining enough variation in normal golf ball appearances for model training.

The cleaning pipeline consists of three sequential stages:

1. SHA-256 hashing for exact duplicate removal.
2. Perceptual hashing (pHash) for near-duplicate removal.
3. Laplacian variance for blur detection and filtering.

The final dataset contains **2,731 images**, retaining approximately **63.2%** of the original cropped dataset.

## 2. Initial Preprocessing

Before deduplication, the existing cropping pipeline (`scripts/data/crop_dataset.py`) prepares individual golf ball images.

The preprocessing steps include:

- Excluding golf balls whose original bounding boxes have a width or height below 32 pixels.
- Cropping each detected golf ball from its source image.
- Converting the crop to grayscale.
- Resizing the image to 128 × 128 pixels.
- Applying a circular mask to remove background pixels outside the golf ball.
- Saving the processed images as JPEGs with quality 95.

The resulting images and updated NDJSON annotations are stored in `data/cropped/`.

The 32-pixel minimum-size filter is applied during cropping, not during subsequent training.

## 3. Cleaning Pipeline

### Stage 1: Exact Duplicate Removal (SHA-256)

**Script:** `scripts/clean.py sha`

SHA-256 generates a cryptographic hash for each image file. Images with identical file contents produce identical hashes.

When duplicates are found, one image is retained and the remaining identical files are removed.

**Heuristic:** Exact SHA-256 hash equality. No numerical similarity threshold is required.

**Input:** `data/cropped/`  
**Output:** `data/cleaned_sha/`

This step removes redundant files without comparing their visual appearance.

### Stage 2: Near-Duplicate Removal (pHash)

**Script:** `scripts/clean.py phash`

Perceptual hashing generates a compact representation of an image's visual structure. Unlike SHA-256, pHash can identify images that are visually similar even when their underlying file contents differ.

Similarity is measured using the Hamming distance between two perceptual hashes.

**Selected heuristic:**

- Hash length: 64 bits.
- Hamming distance threshold: **1**.
- Images with a Hamming distance ≤ 1 are eligible to be grouped as near-duplicates.
- Within each detected group, the image with the highest interior-ball Laplacian variance is retained as the sharpest representative.

The initial pHash threshold was `2`, but it was reduced to `1` after observing substantial image removal.

A threshold of `1` is more conservative and is intended to preserve more useful variation in golf ball texture, rotation, and illumination.

**Input:** `data/cleaned_sha/`  
**Output:** `data/cleaned_phash/`

This threshold is a heuristic rather than a guarantee of visual identity. Distinct images can have similar perceptual hashes.

### Stage 3: Blur Filtering (Laplacian Variance)

**Script:** `scripts/clean.py laplacian`

Laplacian variance is used as a measure of image sharpness.

The Laplacian operator responds to changes in pixel intensity, such as edges and fine surface details. Images with lower Laplacian variance generally contain less high-frequency detail and may be blurry.

Sharpness is evaluated using the interior of the golf ball to reduce interference from the black background and circular mask boundary.

**Selected heuristic:**

- Laplacian variance threshold: **40**.
- Images with scores below the threshold are removed.
- Images meeting or exceeding the threshold are retained.

The threshold of `40` is a provisional quality-control heuristic, not an experimentally validated cutoff. Some naturally low-texture images may also receive low scores.

**Input:** `data/cleaned_phash/`  
**Output:** `data/cleaned_final/`

The script also produces `laplacian_scores.csv` for recording image sharpness scores.

## 4. Running the Pipeline

Run the cleaning stages sequentially from the project root:

```bash
uv run scripts/clean.py all
```

This runs the three stages below in order with the thresholds used for `cleaned_final` (pHash `1`, Laplacian `40`). Each stage can also be run alone:

```bash
uv run scripts/clean.py sha       --input data/cropped       --output data/cleaned_sha
uv run scripts/clean.py phash     --input data/cleaned_sha   --output data/cleaned_phash --threshold 1
uv run scripts/clean.py laplacian --input data/cleaned_phash --output data/cleaned_final --threshold 40
```

The shared `golfball/cleaning.py` module provides utilities for loading image records, processing directories, saving filtered datasets, generating reports, and the mask-aware sharpness score used by both the pHash and Laplacian stages.

The pipeline is automated and does not require manual review of near-duplicate image groups.

**Note:** The commands assume the output directories do not already exist. Existing output directories may need to be moved or removed before rerunning the scripts.

## 5. Final Dataset Results

After cleaning with a pHash threshold of `1` and Laplacian threshold of `40`, the dataset loader reported:

| Dataset split | Images | Percentage |
|---|---:|---:|
| Training | 2,088 | 76.5% |
| Validation | 402 | 14.7% |
| Testing | 241 | 8.8% |
| **Total** | **2,731** | **100%** |

### Overall Retention

| Metric | Images |
|---|---:|
| Original cropped dataset | 4,320 |
| Final cleaned dataset | 2,731 |
| Images removed | 1,589 |
| Retention rate | 63.2% |
| Removal rate | 36.8% |

The original train/validation/test split labels were preserved rather than rebalancing the dataset after filtering.

### Comparison of pHash Thresholds

Two pHash thresholds were evaluated during preprocessing:

| Split | Threshold 2 | Threshold 1 | Additional images retained |
|---|---:|---:|---:|
| Train | 1,702 | 2,088 | +386 |
| Validation | 341 | 402 | +61 |
| Test | 207 | 241 | +34 |
| **Total** | **2,250** | **2,731** | **+481** |

Reducing the pHash threshold from `2` to `1` preserved **481 additional images**, increasing the final dataset size by approximately **21.4%** relative to the more aggressively filtered dataset.

The threshold of `1` was selected as a conservative compromise: it removes extremely similar images while reducing the risk of discarding useful examples of normal golf ball appearances.

## 6. Dataset Loading and Verification

**Module:** `golfball/dataset.py`

The dataset loader is configured to read the final cleaned dataset from:

```text
data/cleaned_final/
```

To verify the dataset:

```bash
uv run -m golfball.dataset
```

The verification run produced:

```text
train 2088 mask area 0.63
val 402 mask area 0.56
test 241 mask area 0.46
```

This confirms that:

- All three dataset splits can be loaded.
- Images and masks have the expected shape `(1, 128, 128)`.
- Image pixels outside the circular mask are zero.
- The final dataset contains 2,731 images.

The reported mask-area values describe the proportion of pixels inside the mask for the first image of each split. They are not dataset-wide averages or model performance metrics.

Running this script does not modify the dataset or train the MAE.

## 7. Limitations and Considerations

**Perceptual similarity:** Golf balls naturally share similar shapes and surface textures. pHash may group distinct images that contain useful visual differences. The selected threshold of `1` reduces, but does not eliminate, this risk.

**Blur threshold:** Laplacian variance is sensitive to image content as well as blur. The threshold of `40` should be treated as provisional until evaluated against model performance or independently labeled image quality.

**Dataset size:** The pipeline removes 36.8% of the original cropped images. This reduces redundancy but may also decrease the diversity of normal examples available for training.

**Data leakage:** Deduplication alone does not guarantee independence between training, validation, and test sets. Images originating from the same video or recording session may remain highly correlated across splits. Ideally, splitting should be performed at the source-video or recording-session level.

**Model performance:** Successful dataset loading does not demonstrate that cleaning improves anomaly detection. That requires comparing model performance using appropriately separated evaluation data.

## 8. Final Configuration

| Component | Selected setting |
|---|---|
| Minimum original bounding-box dimension | 32 pixels |
| Image resolution | 128 × 128 |
| Color format | Grayscale |
| Background processing | Circular mask |
| Exact deduplication | SHA-256 |
| Near-duplicate detection | 64-bit pHash |
| pHash Hamming distance threshold | **1** |
| Near-duplicate representative | Highest interior-ball Laplacian variance |
| Blur detection | Laplacian variance |
| Blur threshold | **40** |
| Final dataset size | **2,731 images** |
| Final dataset directory | `data/cleaned_final/` |

## 9. Conclusion

The cleaning pipeline reduced the original 4,320 cropped golf ball images to **2,731 images** using automated exact deduplication, conservative perceptual deduplication, and blur filtering.

The final configuration prioritizes removing highly redundant and potentially low-quality examples while preserving more variation than the initial pHash threshold of `2`.

The resulting dataset is compatible with the project's PyTorch dataset loader and has passed its basic loading and masking checks. Further experiments are needed to determine whether the cleaning pipeline improves the MAE's anomaly-detection performance.
