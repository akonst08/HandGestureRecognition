# HandGestureRecognition

An advanced/customized hand gesture recognition pipeline based on MediaPipe landmarks, trained for 18 gesture classes.

## Overview
This project extends a MediaPipe-based baseline with a custom 18-class setup, retraining workflow, and project-specific preprocessing/evaluation scripts.

## Advancements in This Version
- Expanded to **18 static hand gesture classes**
- Custom preprocessing and augmentation flow (`scripts/prepareData.py`)
- Updated training notebook (`notebooks/keypoint_classification-correct.ipynb`)
- Inference and frame-level logging pipeline (`app.py`)
- Evaluation utilities with confusion matrix and metrics (`scripts/video.py`)

## Project Structure
- `app.py` - Main real-time inference script
- `model/` - Inference models and classifier modules
- `utils/` - Utility helpers (FPS calculation)
- `notebooks/` - Training notebook(s)
- `scripts/` - Data prep and evaluation scripts
- `assets/` - Demo media for showcase

## Live Detection Demo
- [Watch demo video](assets/demo.mp4)

## Gesture Classes (18)
Call, Dislike, Fist, Four, Like, Mute, Ok, One, Palm, Peace Inverted, Peace, Rock, Stop Inverted, Stop, Three, Three2, Two up Inverted, Two Up

## Requirements
- Python 3.10+
- pip

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Inference
```bash
python app.py
```

## Training Steps
1. Prepare data:
```bash
python scripts/prepareData.py
```
2. Open and run all cells in:
- `notebooks/keypoint_classification-correct.ipynb`

## Evaluation
```bash
python scripts/video.py
```

## License and Attribution
This repository includes work derived from an Apache-2.0 licensed MediaPipe hand-gesture baseline and contains substantial modifications/extensions for this project.
See `LICENSE` and `NOTICE`.
