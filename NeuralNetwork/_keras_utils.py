"""Shared Keras helpers used by the public CMIP6 neural-network scripts."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model


class FeaturewiseGaussianNoise(Layer):
    """Add Gaussian noise to each feature with separate standard deviations."""

    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        if isinstance(stddev, np.ndarray):
            self.stddev = stddev.tolist()
        elif isinstance(stddev, (list, tuple)):
            self.stddev = list(stddev)
        else:
            self.stddev = [float(stddev)]
        self.stddev_tensor = None

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("FeaturewiseGaussianNoise expects 2D input (batch_size, features)")

        n_features = input_shape[1]
        if len(self.stddev) == 1:
            self.stddev = self.stddev * n_features
        elif len(self.stddev) != n_features:
            raise ValueError(
                f"Length of stddev ({len(self.stddev)}) must match number of features ({n_features})"
            )

        self.stddev_tensor = tf.constant(self.stddev, dtype=K.floatx())

    def call(self, inputs, training=None):
        def noised():
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev_tensor,
                dtype=inputs.dtype,
            )
            return inputs + noise

        return K.in_train_phase(noised, inputs, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


NOISE_LAYER_CUSTOM_OBJECTS = {"FeaturewiseGaussianNoise": FeaturewiseGaussianNoise}


def load_model_with_noise_support(model_path, **kwargs):
    """Load a Keras model while supporting the shared noise layer."""

    extra_custom_objects = kwargs.pop("custom_objects", None)
    custom_objects = dict(NOISE_LAYER_CUSTOM_OBJECTS)
    if extra_custom_objects:
        custom_objects.update(extra_custom_objects)
    return load_model(model_path, custom_objects=custom_objects, **kwargs)
