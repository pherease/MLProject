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
LEARN_RATE = 1e-4
BATCH    = 4
# ───── ConfusionMatrixSaver ─────────────────────────────────────────────────
class ConfMatrixSaver(tf.keras.callbacks.Callback):
    """Save a confusion-matrix PNG after each epoch."""
    def __init__(self, val_ds, label_names, out_dir="cm_plots"):
        super().__init__()
        self.val_ds = val_ds
        self.label_names = label_names
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # 1) collect preds vs true
        preds, true = [], []
        for x, y in self.val_ds:
            y_hat = np.argmax(self.model.predict(x, verbose=0), axis=1)
            preds.extend(y_hat); true.extend(y.numpy())

        # 2) build CM
        cm = confusion_matrix(true, preds)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(
            cm, display_labels=self.label_names
        ).plot(xticks_rotation=60,
               cmap="Blues",
               values_format="d",
               ax=ax)
        ax.set_title(f"Epoch {epoch+1}")
        fig.tight_layout()

        # 3) save
        fname = self.out_dir / f"epoch-{epoch+1:02d}.png"
        fig.savefig(fname)
        plt.close(fig)
        print(f"[ConfMatrixSaver] saved → {fname}")
# ───── load paths / labels & stratified split ────────────────────
paths, labels, le = utils.load_paths_and_labels(DATA_CSV)
df = pd.DataFrame(dict(path=paths, label=labels))

tr_p, va_p, te_p, tr_l, va_l, te_l = utils.stratified_split(
    paths, labels, test_size=0.20, val_size=0.20, random_state=SEED
)


def show(name, arr):
    c = collections.Counter(arr)
    print(f"{name} {dict(sorted(c.items()))}  (total: {sum(c.values())})")

show("TRAIN", tr_l); show("VAL  ", va_l); show("TEST ", te_l)

# ───── tf.data pipelines identical to main.py ────────────────────
train_ds = utils.make_dataset(tr_p, tr_l, BATCH, shuffle=True)
val_ds   = utils.make_dataset(va_p, va_l, BATCH, shuffle=False)

# ───── sanity-plot one val batch with augmentations ────────────────

def make_augment():
    return tf.keras.Sequential([
        layers.RandomRotation(0.10),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.Lambda(lambda t: tf.clip_by_value(t, 0., 1.))
    ], name="augment")

# 1) grab one batch  ─────────────────────────────────────────────
imgs, lbls = next(iter(val_ds))

# → cast originals to float32 so Matplotlib is happy
imgs32 = tf.cast(imgs, tf.float32)

# 2) augmentation preview (already cast to float32 below)
aug_preview = make_augment()                     # fresh instance
aug_imgs = aug_preview(imgs32, training=True)    # ← float16

# ↓ cast so both rows are float32
aug_imgs = tf.cast(aug_imgs, tf.float32)

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

def dw_sep_block(x, out_ch):
    """Depth-wise separable conv → BN → ReLU → 1×1 conv → BN → ReLU."""
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(out_ch, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def projection_shortcut(x, out_ch):
    """1×1 projection to match channel depth before residual add."""
    x = layers.Conv2D(out_ch, 1, use_bias=False)(x)
    return layers.BatchNormalization()(x)

def build_model(num_classes=9):
    # ---------- input & basic aug ----------
    inputs = layers.Input(shape=(524, 524, 3))
    x = layers.Resizing(256, 256)(inputs)

    # (you already defined make_augment elsewhere)
    x = make_augment()(x)          # ← fresh augmentation

    # ---------- stem ----------
    x = layers.Conv2D(32, 3, strides=2, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ---------- Block 1 (32 → 64 channels) ----------
    y = dw_sep_block(x, 64)
    s = projection_shortcut(x, 64)    # match channels
    x = layers.Add()([s, y])

    # ---------- Block 2 (64 → 128 channels) ----------
    y = dw_sep_block(x, 128)
    s = projection_shortcut(x, 128)
    x = layers.Add()([s, y])

    # ---------- Block 3 (128 → 128 channels) ----------
    y = dw_sep_block(x, 128)
    # same depth → identity shortcut
    x = layers.Add()([x, y])

    # ---------- Block 4 (128 → 256 channels) ----------
    y = dw_sep_block(x, 256)
    s = projection_shortcut(x, 256)
    x = layers.Add()([s, y])

    # ---------- classifier head ----------
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes,
                           activation="softmax",
                           dtype="float32")(x)

    return tf.keras.Model(inputs, outputs)


cm_saver = ConfMatrixSaver(val_ds, le.classes_, out_dir="cm_plots")

loss = tf.keras.losses.SparseCategoricalCrossentropy(
           from_logits=False)

lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
               initial_learning_rate=LEARN_RATE,
               first_decay_steps=len(train_ds)*30)

model = build_model(num_classes=len(le.classes_))
model.compile(optimizer = tf.keras.optimizers.Adam(lr_sched),
              loss = loss, 
              metrics=["accuracy"])
model.summary()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_w,
    callbacks=[cm_saver],
)
model.save("model_trained.keras")