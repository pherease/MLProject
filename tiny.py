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
from utils import ConfusionMatrixSaver, csv_logger_cb, checkpoint_cb, make_augment, ALRCLoss, LrPrinter


# ───── reproducibility & logging ──────────────────────────────────
SEED = 424
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
LEARN_RATE = 1e-2
BATCH    = 16
EPOCH = 50
IMG_SIZE = 256
autotune = True

# ───── load paths / labels & stratified split ────────────────────
train_paths, train_labels, le = utils.load_paths_and_labels(TRAIN_CSV)

val_paths, val_labels = utils.encode(VAL_CSV, le)
test_paths, test_labels = utils.encode(TEST_CSV, le)

utils.show_dataset_class_distribution("TRAIN", train_labels)
utils.show_dataset_class_distribution("VAL  ", val_labels)
utils.show_dataset_class_distribution("TEST ", test_labels)

# ───── tf.data pipelines identical to main.py ────────────────────
augment_layer = make_augment()
train_ds = utils.make_dataset(train_paths, train_labels, BATCH, shuffle=True, autotune = autotune)
train_ds = train_ds.map(
        lambda imgs, labs: (augment_layer(imgs, training=True), labs), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = utils.make_dataset(val_paths, val_labels, BATCH, shuffle=True, autotune = autotune)
test_ds = utils.make_dataset(test_paths, test_labels, BATCH, shuffle=False, autotune = autotune)

# ───── sanity-plot one val batch with augmentations ────────────────
utils.show_image_of_batch(train_ds, le)
# ───── class weights to regularize differently sized categories ───────────
cw_vals = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train_labels),
    y = train_labels
)
class_w = dict(enumerate(cw_vals))

def build_model():
    filters = 128
    model = tf.keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(filters, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//4, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters, (2,2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),

        # layers.Dense(dense1, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.25),
        # layers.Dense(dense2, activation="relu"),

        layers.Dense(len(le.classes_), activation='softmax', dtype='float32')
    ])
    return model


def sparse_focal_loss(gamma: float = 1.0, alpha: float = 0.25):
    """Sparse-label focal loss that works with int32 *or* int64 y_true."""
    def loss_fn(y_true, y_pred):
        # ensure both tensors share the same integer dtype
        y_true = tf.cast(y_true, tf.int64)                      # <─ ①

        # build range in the *same* dtype as y_true
        batch_idxs = tf.range(tf.shape(y_pred)[0], dtype=y_true.dtype)  # <─ ②
        indices = tf.stack([batch_idxs, y_true], axis=1)

        # gather p_t (prob of true class) and compute loss
        p_t = tf.gather_nd(y_pred, indices)
        alpha_factor = tf.where(tf.equal(y_true, 0), 1.0 - alpha, alpha)
        focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
        loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
        return tf.reduce_mean(loss)
    return loss_fn

lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", 
                                             factor = 0.25, 
                                             patience = 3, 
                                             min_lr = 1e-6,
                                             min_delta = 0.01)

ALRCLoss_fn = ALRCLoss(
    num_stddev = 2,
    decay = 0.999,
    mu1_start = 0.85,
    mu2_start = 0.90**2
)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

model = build_model()
model.compile(optimizer = tf.keras.optimizers.Adam(LEARN_RATE),
              loss = ALRCLoss_fn, 
              metrics=["accuracy"])
model.summary()

lr_printer = LrPrinter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = EPOCH,
    callbacks=[ConfusionMatrixSaver(val_ds, le.classes_, every=3),  
               lr_printer,
               csv_logger_cb(),
               lr_cb
               ]
)
model.save("model_trained.keras")

