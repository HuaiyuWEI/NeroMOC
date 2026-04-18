"""Shared filesystem helpers for the public CMIP6 Python scripts."""

from __future__ import annotations

import os
from typing import TypeVar

PathT = TypeVar("PathT")


def require_existing_directory(path: PathT, label: str) -> PathT:
    """Return `path` if it exists as a directory, otherwise raise a clear error."""

    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def require_existing_file(path: PathT, label: str) -> PathT:
    """Return `path` if it exists as a file, otherwise raise a clear error."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path


def ensure_directory(path: PathT) -> PathT:
    """Create `path` if needed and return it unchanged."""

    os.makedirs(path, exist_ok=True)
    return path
