#!/usr/bin/env python3
# check.py  – diagnostic script for TrashNet project

# -----------------------------------------------
import os, random, numpy as np
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async" 
import tensorflow as tf
import matplotlib, matplotlib.pyplot as plt
import utils

from sklearn.utils import class_weight
from tensorflow.keras import layers, mixed_precision
from utils import ConfusionMatrixSaver


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
BATCH    = 2
EPOCH = 60
IMG_SIZE = 128
autotune = True

# ───── load paths / labels & stratified split ────────────────────
paths, labels, le = utils.load_paths_and_labels(DATA_CSV)

tr_p, va_p, te_p, tr_l, va_l, te_l = utils.stratified_split(
    paths, labels, test_size=0.20, val_size=0.20, random_state=SEED
)

utils.show_dataset_class_distribution("TRAIN", tr_l)
utils.show_dataset_class_distribution("VAL  ", va_l)
utils.show_dataset_class_distribution("TEST ", te_l)

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
utils.show_image_of_batch(val_ds, le)

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
    callbacks=[ConfusionMatrixSaver(val_ds, le.classes_, every=3)]
)
model.save("model_trained.keras")

