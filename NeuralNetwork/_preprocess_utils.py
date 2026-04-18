"""Shared preprocessing helpers for CMIP6 data-preparation scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_low_pass_filter(data, cutoff_freq, order=5, sampling_rate=1, padding_length=None):
    """Apply a Butterworth low-pass filter along the first axis (time)."""

    sos = butter(order, cutoff_freq, btype="low", output="sos", analog=False, fs=sampling_rate)

    if padding_length is None:
        padding_length = 0

    ndim = data.ndim

    if ndim == 1:
        filtered_data = np.empty_like(data)
        padded = np.pad(data, (padding_length, padding_length), mode="reflect") if padding_length else data
        filtered = sosfiltfilt(sos, padded)
        filtered_data[:] = filtered[padding_length:-padding_length] if padding_length else filtered
    elif ndim == 2:
        _, ny = data.shape
        filtered_data = np.empty_like(data)

        for j in range(ny):
            ts = data[:, j]
            if np.any(np.isnan(ts)):
                filtered_data[:, j] = ts
                continue

            padded = np.pad(ts, (padding_length, padding_length), mode="reflect") if padding_length else ts
            filtered = sosfiltfilt(sos, padded)
            filtered_data[:, j] = filtered[padding_length:-padding_length] if padding_length else filtered
    elif ndim == 3:
        _, ny, nx = data.shape
        filtered_data = np.empty_like(data)

        for j in range(ny):
            for i in range(nx):
                ts = data[:, j, i]
                if np.any(np.isnan(ts)):
                    filtered_data[:, j, i] = ts
                    continue

                padded = np.pad(ts, (padding_length, padding_length), mode="reflect") if padding_length else ts
                filtered = sosfiltfilt(sos, padded)
                filtered_data[:, j, i] = filtered[padding_length:-padding_length] if padding_length else filtered
    else:
        raise ValueError("Input data must be 1D, 2D, or 3D with time as the first dimension.")

    return filtered_data


def plot_random_clean_series(temp, temp_lpf, max_attempts=100):
    """Plot a random non-NaN time series from a 2D or 3D array."""

    if temp.ndim == 3:
        _, nlat, nlon = temp.shape
        for _ in range(max_attempts):
            i = np.random.randint(0, nlat)
            j = np.random.randint(0, nlon)
            raw_series = temp[:, i, j]
            if not np.any(np.isnan(raw_series)):
                filtered_series = temp_lpf[:, i, j]
                label = f"Time series at random grid point (lat {i}, lon {j})"
                break
        else:
            raise ValueError("No valid 3D grid point found without NaNs.")
    elif temp.ndim == 2:
        _, npos = temp.shape
        for _ in range(max_attempts):
            idx = np.random.randint(0, npos)
            raw_series = temp[:, idx]
            if not np.any(np.isnan(raw_series)):
                filtered_series = temp_lpf[:, idx]
                label = f"Time series at random position {idx}"
                break
        else:
            raise ValueError("No valid 2D position found without NaNs.")
    else:
        raise ValueError("Input `temp` must be either 2D or 3D.")

    plt.figure(figsize=(10, 4))
    plt.plot(raw_series, label="Original", linestyle="--", color="gray")
    plt.plot(filtered_series, label="Low-pass filtered", color="blue")
    plt.xlabel("Time (months)")
    plt.ylabel("Value")
    plt.title(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
