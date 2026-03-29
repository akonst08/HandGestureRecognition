# HandGestureRecognition

An advanced/customized hand gesture recognition pipeline based on MediaPipe landmarks, trained for 18 gesture classes.

## Overview
This project extends a MediaPipe-based baseline with a custom 18-class setup, retraining workflow, and project-specific preprocessing/evaluation scripts.

## Advancements in This Version
- Expanded to **18 static hand gesture classes**
- Custom preprocessing and augmentation flow (`prepareData.py`)
- Updated training notebook (`keypoint_classification-correct.ipynb`)
- Inference and frame-level logging pipeline (`app.py`)
- Evaluation utilities with confusion matrix and metrics (`video.py`)

## Live Detection Demo
- [Watch demo video](assets/live_detection_demo.mp4)

## Gesture Classes (18)
Call, Dislike, Fist, Four, Like, Mute, Ok, One, Palm, Peace Inverted, Peace, Rock, Stop Inverted, Stop, Three, Three2, Two up Inverted, Two Up

## Requirements
- Python 3.10+
- pip

Install dependencies:
```bash
pip install -r requirements.txt
```
