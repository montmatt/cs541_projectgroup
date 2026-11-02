# CS541 Final Project Repository

## Group Members

- Neel Awsare
- Daniel Gwon
- Matthew Montero
- Onkar Sama

## Paper

- [Link to Paper Proposal](https://docs.google.com/document/d/1iLjUq3U7Ygea7iWgvsFbWedjgt0L40MWl2J0o4DOJvw/edit?usp=sharing)
- [Link to paper (not available yet)]()

## Data

- [`data/golf ball.v1i.coco`](<data/golf ball.v1i.coco>)
    - Original Source: [golf ball Computer Vision Model](https://universe.roboflow.com/young-jin-park-6vhep/golf-ball-5sqna)
    - No images with defects.
    - Contain images in directory.
    - Imported as individual datasets of train/validation/test splits (modify later to combine all into one dataset)
- [`data/golf-ball-detection.ndjson`](data/golf-ball-detection.ndjson)
    - Original Source: [Golf Ball Detection](https://platform.ultralytics.com/xu-wen/datasets/golf-ball-detection)
    - No images with defects.
    - References out of directory images.
    - Imported as a complete dataset with train/validation/test splits in the JSON.
- [`data/golf-ball.ndjson`](data/golf-ball.ndjson)
    - Original Source: [golf ball](https://platform.ultralytics.com/shark-heron/datasets/golf-ball)
    - Contains images with both defects and no defects.
    - References out of directory images.
    - Imported as a complete dataset with train/validation/test splits in the JSON.

## Setup

This project uses uv. 

- Sync dependencies with `uv sync`
- Add packages with `uv add <package>` (not pip)
- Run scripts with `uv run <script>.py`