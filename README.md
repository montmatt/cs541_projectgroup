# CS541 Final Project Repository

## Group Members

- Neel Awsare
- Daniel Gwon
- Matthew Montero
- Onkar Sama

## Paper

- [Link to Paper Proposal](https://docs.google.com/document/d/1iLjUq3U7Ygea7iWgvsFbWedjgt0L40MWl2J0o4DOJvw/edit?usp=sharing)
- [Link to paper (not available yet)]()

## [Data](data)

- [`data/raw`](data/raw) = imported sources
- [`scripts`](scripts) = data tooling, [`docs`](docs) = assignment PDF and LaTeX paper example


- [`data/raw/golf-ball-roboflow.ndjson`](data/raw/golf-ball-roboflow.ndjson) = 2188 images
    - Source: [golf ball Computer Vision Model | Roboflow](https://universe.roboflow.com/young-jin-park-6vhep/golf-ball-5sqna)
    - No images with defects.
    - Contains images in directory.
    - Imported as individual datasets of train/validation/test splits.
        - Original Import: [`data/raw/golf-ball-roboflow-original`](data/raw/golf-ball-roboflow-original)
    - Reformated as one dataset with train/validation/test splits in JSON
        - **IMPORTANT**: DO NOT DELETE ORIGINAL IMAGES. `data/raw/golf-ball-roboflow.ndjson` points to original imports.
- [`data/raw/golf-ball-detection.ndjson`](data/raw/golf-ball-detection.ndjson) = 989 images
    - Source: [Golf Ball Detection | Ultralytics](https://platform.ultralytics.com/xu-wen/datasets/golf-ball-detection)
    - No images with defects.
    - References out of directory images.
    - Imported as one dataset with train/validation/test splits in JSON.
- [`data/raw/golf-ball.ndjson`](data/raw/golf-ball.ndjson) = 60 images
    - Source: [golf ball | Ultralytics](https://platform.ultralytics.com/shark-heron/datasets/golf-ball)
    - Contains images with both defects and no defects.
    - References out of directory images.
    - Imported as one dataset with train/validation/test splits in JSON.

## Setup

This project uses uv.

- Sync dependencies with `uv sync`
- Add packages with `uv add <package>` (not pip)
- Run scripts with `uv run scripts/<script>.py`