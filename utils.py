import tensorflow as tf
import os
import cv2
from typing import Tuple

import numpy as np
from typing import Optional
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def getDatum(file_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Reads JPEG, decodes, resizes to img_size², normalizes to [0,1].
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizes to [0,1]
    return img, label


def loadData(
    data_path: str = "data.csv",
    batch_size: int = 32,
    shuffle: bool = True
):
    """
    Loads data from a CSV file, encodes category labels, and returns a TensorFlow dataset and label encoder.

    Args:
        data_path (str): Path to the CSV file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        Tuple[tf.data.Dataset, LabelEncoder]: Batched TensorFlow dataset and fitted label encoder.
    """
    df = pd.read_csv(data_path)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])
    paths, labels = df['filePath'].values, df['label'].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(getDatum, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, le


def summarizeData(ds: tf.data.Dataset, le: LabelEncoder) -> None:
    """
    Prints a summary of the dataset.

    Args:
        ds: TensorFlow dataset of (image, label) pairs.
        le: Fitted LabelEncoder.
    """
    print("Dataset Summary:")
    num_samples = ds.cardinality().numpy()
    print(f"Number of samples: {num_samples}")
    print(f"Number of unique categories: {len(le.classes_)}")
    if num_samples > 0:
        print(f"Sample data: {next(iter(ds))}")
    else:
        print("Sample data: Dataset is empty.")

def make_dataset(paths, labels, batch_size=8, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: getDatum(p, l),
                num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def load_paths_and_labels(csv_path="data.csv"):
    """
    Returns (file_paths: np.ndarray[str], labels: np.ndarray[int], LabelEncoder).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    df = pd.read_csv(csv_path)
    for col in ("filePath","category"):
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    return df["filePath"].values, df["label"].values, le