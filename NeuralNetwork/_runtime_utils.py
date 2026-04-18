"""Shared runtime helpers for CMIP6 neural-network scripts."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class CovariateConfig:
    """Normalized covariate configuration derived from a comma-separated string."""

    covariate_names: str
    input_var: str
    noise_size: int


def configure_tensorflow_runtime(seed: int = 0) -> None:
    """Apply the shared TensorFlow runtime configuration used across scripts."""

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="keras.engine.training_v1",
    )
    tf.keras.utils.set_random_seed(seed)
    tf.config.list_physical_devices("GPU")
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()


def prepare_covariate_config(
    covariate_names: str,
    amoc26_input: bool = False,
    amoc56_input: bool = False,
) -> CovariateConfig:
    """Build normalized covariate metadata used in multiple experiment scripts."""

    covariate_list = [name.strip() for name in covariate_names.split(",") if name.strip()]
    noise_match = re.search(r"whiteNoise(\d+)", ",".join(covariate_list))
    noise_size = int(noise_match.group(1)) if noise_match else 0

    input_parts = list(covariate_list)
    if amoc26_input:
        input_parts.append("AMOC26")
    if amoc56_input:
        input_parts.append("AMOC56")

    return CovariateConfig(
        covariate_names=",".join(covariate_list),
        input_var="+".join(input_parts),
        noise_size=noise_size,
    )
