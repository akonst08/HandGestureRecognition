#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.keras',
        num_threads=1,
    ):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def __call__(
        self,
        landmark_list,
    ):
        x = np.array([landmark_list], dtype=np.float32)
        y = self.model.predict(x, verbose=0)
        return int(np.argmax(np.squeeze(y)))
