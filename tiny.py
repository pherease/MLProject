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
from utils import ConfusionMatrixSaver, csv_logger_cb, checkpoint_cb, make_augment


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

TRAIN_CSV = "./split_data/train.csv"
TEST_CSV = "./split_data/test.csv"
VAL_CSV = "./split_data/val.csv"
# ───── Hyperparametes ─────────────────────────────────────────────────
LEARN_RATE = 1e-4
BATCH    = 2
EPOCH = 60
IMG_SIZE = 256
autotune = True

# ───── load paths / labels & stratified split ────────────────────
train_paths, train_labels, train_label_encoder = utils.load_paths_and_labels(TRAIN_CSV)
val_paths, val_labels, val_label_encoder = utils.load_paths_and_labels(VAL_CSV)
test_paths, test_labels, test_label_encoder = utils.load_paths_and_labels(TEST_CSV)

utils.show_dataset_class_distribution("TRAIN", train_labels)
utils.show_dataset_class_distribution("VAL  ", val_labels)
utils.show_dataset_class_distribution("TEST ", test_labels)

# ───── tf.data pipelines identical to main.py ────────────────────
augment_layer = make_augment()
train_ds = utils.make_dataset(train_paths, train_labels, BATCH, shuffle=True, autotune = autotune)
train_ds = train_ds.map(
        lambda imgs, labs: (augment_layer(imgs, training=True), labs), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = utils.make_dataset(val_paths, val_labels, BATCH, shuffle=False, autotune = autotune)
test_ds = utils.make_dataset(test_paths, test_labels, BATCH, shuffle=False, autotune = autotune)

# ───── sanity-plot one val batch with augmentations ────────────────
utils.show_image_of_batch(train_ds, train_label_encoder)

# ───── class weights to regularize differently sized categories ───────────
cw_vals = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train_labels),
    y = train_labels
)
class_w = dict(enumerate(cw_vals))

def build_model(num_classes: int = 9):
    model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')])

    return model

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model = build_model(num_classes=len(train_label_encoder.classes_))
model.compile(optimizer = tf.keras.optimizers.Adam(LEARN_RATE),
              loss = loss, 
              metrics=["accuracy"])
model.summary()


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = EPOCH,
    class_weight=class_w,
    callbacks=[ConfusionMatrixSaver(val_ds, train_label_encoder.classes_, every=3), checkpoint_cb(), csv_logger_cb()]
)
model.save("model_trained.keras")

