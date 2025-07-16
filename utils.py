from __future__ import annotations

import os
import collections
import pathlib
from pathlib import Path
from typing import Tuple, Sequence
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
IMG_SIZE: Tuple[int, int] = (256, 256)
NUM_PAR_CALLS = 32     # parallel JPEG decodes
SHUFFLE_BUF = 5000
PREFETCH_BUFS = 32

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_paths_and_labels(csv_path, le = None) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
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


def encode(csv, le):
    df = pd.read_csv(csv)
    return df["filePath"].to_numpy(), le.transform(df["category"])

# ---------------------------------------------------------------------------
# Dataset splitting with stratification
# ---------------------------------------------------------------------------

def stratified_split(
    paths: Sequence[str] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
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
    img = tf.divide(tf.cast(img, tf.float32), 255.0)
    return img, label


def make_dataset(
    paths: Sequence[str] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    batch_size: int,
    autotune: bool = False,
    *,
    shuffle: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(SHUFFLE_BUF, len(paths)), seed=424)

    if autotune:
        return ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    else:
        return ds.map(_load_and_preprocess, num_parallel_calls=NUM_PAR_CALLS).batch(batch_size).prefetch(PREFETCH_BUFS)

def resize(img_size = IMG_SIZE):
    return layers.Resizing(img_size, img_size)


def make_augment():
    return tf.keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomFlip("horizontal"),
        # layers.RandomContrast(0.10),
        layers.Lambda(lambda t: tf.clip_by_value(t, 0., 1.))
    ], name="augment")

def show_image_of_batch(ds: tf.data.Dataset, le: LabelEncoder) -> None:
    imgs, lbls = next(iter(ds))
    # → cast originals to float32 so Matplotlib is happy
    imgs32 = tf.cast(imgs, tf.float32)

    # 3) plot originals vs augmented  ────────────────────────────────
    batch_size = int(imgs32.shape[0]) # type: ignore
    fig, axes = plt.subplots(2, int(batch_size/2), figsize=(batch_size * 2.5, 5)) # type: ignore
    axes = axes.flatten()
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
        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(cm, display_labels=self.label_names).plot(
            xticks_rotation=60, cmap="Blues", ax=ax, values_format="d")
        ax.set_title(f"Epoch {epoch+1}")
        fig.savefig(self.out_dir / f"cm_epoch_{epoch+1:03d}.png")
        plt.close(fig)

class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get the LR: if it’s a schedule, evaluate it at the current step
        lr = self.model.optimizer.learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        # force it into Python float
        lr = float(tf.keras.backend.get_value(lr))
        # add to logs so progbar/TensorBoard see it
        logs["lr"] = lr

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


class ALRCLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        num_stddev=3,
        decay=0.999,
        mu1_start=25.0,
        mu2_start=30**2,
        eps=1e-8,
        name="alrc_loss"
    ):
        # Tell Keras we want per-sample losses (no reduction here)
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

        self.num_stddev = num_stddev
        self.decay      = decay
        self.eps        = eps

        # Plain tf.Variables instead of add_weight()
        # They’re non‑trainable by default.
        self.mu  = tf.Variable(mu1_start, dtype=tf.float32, trainable=False, name="mu1")
        self.mu2 = tf.Variable(mu2_start, dtype=tf.float32, trainable=False, name="mu2")

    def call(self, y_true, y_pred):
        # 1) raw per-sample loss (change to your base loss as needed)
        raw = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred
        )  # → shape=(batch,)

        # 2) dynamic threshold: μ + num_stddev * sqrt(μ2 - μ^2 + eps)
        sigma     = tf.sqrt(self.mu2 - self.mu**2 + self.eps)
        threshold = self.mu + self.num_stddev * sigma

        # 3) clip each element of raw loss to [0, threshold]
        clipped = tf.minimum(raw, threshold)

        # 4) update the running averages with the *clipped* losses
        mean_l   = tf.reduce_mean(clipped)
        mean_l2  = tf.reduce_mean(clipped * clipped)

        # Imperative assigns will run in eager or inside the graph
        self.mu .assign(self.decay * self.mu  + (1 - self.decay) * mean_l)
        self.mu2.assign(self.decay * self.mu2 + (1 - self.decay) * mean_l2)

        return clipped

class CleanTuner(kt.tuners.RandomSearch):
    def run_trial(self, trial, *args, **kw):
        tf.keras.backend.clear_session(); gc.collect()
        super().run_trial(trial, *args, **kw)
