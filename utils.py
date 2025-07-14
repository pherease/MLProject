from __future__ import annotations

import os
import collections
import pathlib
from pathlib import Path
from typing import Tuple, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    return df["filePath"].to_numpy(), df["label"].to_numpy(), le


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
    img = tf.divide(tf.cast(img, tf.float32), 255.0)
    return img, label


def make_dataset(
    paths: Sequence[str] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    batch_size: int = 4,
    autotune: bool = False,
    *,
    shuffle: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(SHUFFLE_BUF, len(paths)), seed=42)

    if autotune:
        return (
            ds.map(_load_and_preprocess, num_parallel_calls=NUM_PAR_CALLS)
            .batch(batch_size)
            .prefetch(PREFETCH_BUFS))
    else:
        return (
            ds
            .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

def show_image_of_batch(ds: tf.data.Dataset, le: LabelEncoder) -> None:
    imgs, lbls = next(iter(ds))
    # → cast originals to float32 so Matplotlib is happy
    imgs32 = tf.cast(imgs, tf.float32)

    # 3) plot originals vs augmented  ────────────────────────────────
    batch_size = int(imgs32.shape[0]) # type: ignore
    fig, axes = plt.subplots(batch_size, figsize=(batch_size * 2.5, 5)) # type: ignore

    for i in range(batch_size):
        # originals
        axes[i].imshow(np.clip(imgs32[i].numpy(), 0, 1)) # type: ignore
        axes[i].set_title(le.inverse_transform([lbls[i].numpy()])[0])
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig("batch.png")
    fig.show()
    print("Saved → batch.png")

def show_dataset_class_distribution(name, arr):
    c = collections.Counter(arr)
    print(f"{name} {dict(sorted(c.items()))}  (total: {sum(c.values())})")

class ConfusionMatrixSaver(tf.keras.callbacks.Callback):
    """Compute CM rarely & on *sub-sample* of val set to save time."""
    def __init__(self, val_ds, label_names,
                 every=5,          # only every N epochs
                 out_dir="cm_plots"):
        super().__init__()
        self.val_ds = val_ds
        self.label_names = label_names
        self.every = every
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, _):
        if (epoch + 1) % self.every:       # quick check ⇒ skip
            return

        y_true, y_pred = [], []
        for x, y in self.val_ds:
            y_pred.extend(np.argmax(self.model(x, training=False), axis=1))
            y_true.extend(y.numpy())

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=self.label_names).plot(
            xticks_rotation=60, cmap="Blues", ax=ax, values_format="d")
        ax.set_title(f"Epoch {epoch+1}")
        fig.savefig(self.out_dir / f"cm_epoch_{epoch+1:03d}.png")
        plt.close(fig)

def checkpoint_cb():
    return tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/weights_epoch_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=1
)

# 2) Log all losses & metrics to CSV:
def csv_logger_cb():
    return tf.keras.callbacks.CSVLogger(
    filename="training_log.csv",
    separator=",",
    append=True
)
