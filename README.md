# Dental image detection and segmentation experiments

An early computer-vision exploration using Ultralytics YOLO on a small set of dental images. The original [Colab notebook](teeth.ipynb) contains object detection, segmentation, and keypoint experiments; this repository also includes historical detection and segmentation checkpoints and their image/annotation files. A small Python command below makes the two included checkpoints easier to inspect locally.

> **Not for clinical use.** These experiments have not been independently validated for diagnosis or treatment. Data provenance, consent, reuse rights and patient-level separation have not been established in this repository.

## Run a checkpoint on one image

With Python 3.10 and the dependencies installed:

```bash
python -m pip install -r requirements.txt
python scripts/predict.py --model segment --image images/test/ff90ceb2-6607778.png --output prediction.png
```

Use `--model detect` for `models/best.pt`, or `--model segment` for `SegModels/best.pt`. The command is intended to run on CPU and write an annotated image to a **new** output path; it refuses to overwrite an existing file. End-to-end inference has not yet been verified: the available local Python 3.10 environment has incompatible Torch and torchvision versions. A clean environment with compatible versions is needed. Load PyTorch checkpoints only when you trust their source.

To run the input-safety tests, install `pytest` and use `python -m pytest -q`. These tests do not measure model quality.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`teeth.ipynb`](teeth.ipynb) | Original notebook and saved historical training output. |
| [`scripts/predict.py`](scripts/predict.py) | Local single-image detection or segmentation visualization. |
| [`models/best.pt`](models/best.pt), [`SegModels/best.pt`](SegModels/best.pt) | Historical detection and segmentation checkpoints. |
| [`images/`](images/) and [`labels/`](labels/) | Historical image and YOLO-label splits. |
| [`keypoint/`](keypoint/), [`keypoint.yaml`](keypoint.yaml) | Historical keypoint files and configuration; the notebook references a separate pose checkpoint not included here. |

## Evaluation and reproducibility limits

- The original [`data.yaml`](data.yaml) points training, validation, and test to the same `images/test` directory. It also uses an old Google Drive path. Keep it as historical evidence; do not use it to claim held-out performance.
- Among the checked-in detection/segmentation images, exact image bytes repeat across splits: one train/test pair and two train/validation pairs. The repository contains 18 train, 3 validation, and 3 test images, so these are not independent held-out splits.
- The saved notebook metrics are historical logs, not a fresh evaluation. A valid assessment requires verified data permissions, patient-level separation, a clean holdout and a documented evaluation protocol.
- The original notebook needs Colab paths and, for its pose example, an external checkpoint. The local command is an inference demonstration; it does not reproduce training or establish accuracy.
