"""Utility helpers for TrashNet

Changes in this version
-----------------------
* 2025-07-11 – `stratified_split()` added.  Splits the dataset into train /
  validation / test **with identical class proportions** using one call to
  `sklearn.model_selection.train_test_split` under the hood.
* 2025-07-11 – `IMG_SIZE` reverted to `(256, 256)` so MobileNet-like backbones
  can be used directly without huge GPU memory.
* The rest of the preprocessing pipeline from the previous skeleton is kept
  intact (small shuffle / prefetch windows, limited parallelism, explicit
  float → [0,1] scaling).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (524, 524)
NUM_PAR_CALLS = 4     # parallel JPEG decodes
SHUFFLE_BUF = 200     # lower = less RAM, still random
PREFETCH_BUFS = 2

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_paths_and_labels(csv_path: str | os.PathLike = "data.csv") -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Read the project CSV and return file paths, integer labels, encoder.

    The CSV is expected to have at least two columns:
      * ``filePath`` – absolute or relative path to the image file
      * ``category`` – categorical class label
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    return df["filePath"].values, df["label"].values, le


# ---------------------------------------------------------------------------
# Dataset splitting with stratification
# ---------------------------------------------------------------------------

def stratified_split(
    paths: Sequence[str] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return **train / val / test** splits with identical class proportions.

    Parameters
    ----------
    paths, labels
        1-D arrays with the same length.
    test_size, val_size
        Fractions of the *original* dataset to carve out for test and
        validation. They can be ints (absolute counts) or floats.  The
        function first peels off the *test* portion, then splits the
        remaining data into train and validation, all with
        ``stratify=labels``.
    random_state
        Guarantees reproducibility.

    Returns
    -------
    tuple of six ``np.ndarray``:
        ``train_paths, val_paths, test_paths, train_labels, val_labels, test_labels``
    """
    paths = np.asarray(paths)
    labels = np.asarray(labels)

    # 1) Split off the test set
    tr_paths, te_paths, tr_labels, te_labels = train_test_split(
        paths,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # 2) Now split what remains into train / validation
    val_frac = val_size / (1.0 - test_size)  # make it relative to the leftover
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        tr_paths,
        tr_labels,
        test_size=val_frac,
        random_state=random_state,
        stratify=tr_labels,
    )

    return tr_paths, val_paths, te_paths, tr_labels, val_labels, te_labels


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------

def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read JPEG → resize → scale to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_dataset(
    paths: Sequence[str] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    batch_size: int = 4,
    *,
    shuffle: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(SHUFFLE_BUF, len(paths)), seed=42)

    return (
        ds.map(_load_and_preprocess, num_parallel_calls=NUM_PAR_CALLS)
        .batch(batch_size)
        .prefetch(PREFETCH_BUFS)
    )
