# HandGestureRecognition

Hand gesture recognition project using MediaPipe hand landmarks and custom classifiers, extended for an 18-class gesture recognition task.

## Project Overview
This repository presents an advanced/customized version of a MediaPipe-based baseline:
- Expanded and customized gesture classes (18 total)
- Custom data preparation and augmentation scripts
- Retraining notebooks for keypoint and point-history classifiers
- Video/webcam inference pipeline
- Evaluation scripts with confusion matrix and classification metrics

## Gestures (18)
Call, Dislike, Fist, Four, Like, Mute, Ok, One, Palm, Peace Inverted, Peace, Rock, Stop Inverted, Stop, Three, Three2, Two up Inverted, Two Up

## Main Files
- `app.py`: Main inference script (webcam/video)
- `prepareData.py`: Data preprocessing and augmentation
- `video.py`: Evaluation metrics and confusion matrix
- `groundtruth.py`: Ground-truth CSV generation helper
- `keypoint_classification-correct.ipynb`: Keypoint model training notebook
- `point_history_classification.ipynb`: Point-history model training notebook

## How To Run
```bash
python app.py
```
