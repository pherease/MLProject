#!/usr/bin/env python3
# check.py  – diagnostic script for TrashNet project

# -----------------------------------------------
import os, random, collections, math, numpy as np, pandas as pd
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
import tensorflow as tf
import matplotlib, matplotlib.pyplot as plt
import utils
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras import layers, mixed_precision


# ───── reproducibility & logging ──────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ──────────────── Settings ──────────────────────────────────
mixed_precision.set_global_policy('mixed_float16')
matplotlib.use("Agg")
tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
DATA_CSV = "data.csv"
# ───── Hyperparametes ─────────────────────────────────────────────────
LEARN_RATE = 1e-3
BATCH    = 2
EPOCH = 60
IMG_SIZE = 128
autotune = True
# ───── ConfusionMatrixSaver ─────────────────────────────────────────────────
class LiteConfMatrixSaver(tf.keras.callbacks.Callback):
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
# ───── load paths / labels & stratified split ────────────────────
paths, labels, le = utils.load_paths_and_labels(DATA_CSV)

tr_p, va_p, te_p, tr_l, va_l, te_l = utils.stratified_split(
    paths, labels, test_size=0.20, val_size=0.20, random_state=SEED
)


def show(name, arr):
    c = collections.Counter(arr)
    print(f"{name} {dict(sorted(c.items()))}  (total: {sum(c.values())})")

show("TRAIN", tr_l); show("VAL  ", va_l); show("TEST ", te_l)

def make_augment(img_size = IMG_SIZE):
    return tf.keras.Sequential([
        layers.Resizing(img_size, img_size, name="resize"),
        layers.RandomRotation(0.10),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.Lambda(lambda t: tf.clip_by_value(t, 0., 1.))
    ], name="augment")
augment = make_augment(IMG_SIZE)
# ───── tf.data pipelines identical to main.py ────────────────────
train_ds = utils.make_dataset(tr_p, tr_l, BATCH, shuffle=True, autotune = autotune)
train_ds = train_ds.map(lambda img, lbl: (augment(img), lbl), num_parallel_calls=tf.data.AUTOTUNE)
val_ds   = utils.make_dataset(va_p, va_l, BATCH, shuffle=False, autotune = autotune)
val_ds = val_ds.map(lambda img, lbl: (tf.image.resize(img, [IMG_SIZE, IMG_SIZE]), lbl), num_parallel_calls=tf.data.AUTOTUNE)

# ───── sanity-plot one val batch with augmentations ────────────────

# 1) grab one batch  ─────────────────────────────────────────────
imgs, lbls = next(iter(val_ds))

# → cast originals to float32 so Matplotlib is happy
imgs32 = tf.cast(imgs, tf.float32)

# 2) augmentation preview (already cast to float32 below)                   # fresh instance
aug_imgs = augment(imgs32, training=True)    # ← float16

# ↓ cast so both rows are float32
aug_imgs = tf.cast(aug_imgs, tf.float32)
tf.debugging.check_numerics(aug_imgs, "Augmented images contain NaNs")

# 3) plot originals vs augmented  ────────────────────────────────
batch_size = imgs32.shape[0]
fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2.5, 5))

for i in range(batch_size):
    # originals
    axes[0, i].imshow(np.clip(imgs32[i].numpy(), 0, 1))
    axes[0, i].set_title(le.inverse_transform([lbls[i].numpy()])[0])
    axes[0, i].axis("off")

    # augmented
    axes[1, i].imshow(np.clip(aug_imgs[i].numpy(), 0, 1))
    axes[1, i].set_title("augmented")
    axes[1, i].axis("off")

fig.tight_layout()
fig.savefig("val_batch_with_aug.png")
print("Saved → val_batch_with_aug.png")

# ───── class weights to regularize differently sized categories ───────────
cw_vals = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(tr_l),
    y = tr_l
)
class_w = dict(enumerate(cw_vals))

def build_model(num_classes: int = 9):
    model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')])

    return model

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)



model = build_model(num_classes=len(le.classes_))
model.compile(optimizer = tf.keras.optimizers.Adam(LEARN_RATE),
              loss = loss, 
              metrics=["accuracy"])
model.summary()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = EPOCH,
    class_weight=class_w,
    callbacks=[LiteConfMatrixSaver(val_ds, le.classes_, every=3)]
)
model.save("model_trained.keras")

