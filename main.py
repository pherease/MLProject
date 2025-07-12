import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils
import os
import cv2

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, callbacks
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')
BATCH = 4
EPOCHS = 20

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)



all_paths, all_labels, label_encoder = utils.load_paths_and_labels("data.csv")

# 2) split into train+val and test (50%)
paths_trainval, paths_test, labs_trainval, labs_test = train_test_split(
    all_paths, all_labels, test_size=0.5, random_state=42, stratify=all_labels
)

# 3) split trainval into train (80% of trainval) and val (20%)
paths_train, paths_val, labs_train, labs_val = train_test_split(
    paths_trainval, labs_trainval, test_size=0.2,
    random_state=42, stratify=labs_trainval
)

# 4) build tf.data.Datasets
train_ds = utils.make_dataset(paths_train, labs_train, BATCH, shuffle=False)
val_ds   = utils.make_dataset(paths_val,   labs_val,   BATCH, shuffle=False)
test_ds  = utils.make_dataset(paths_test,  labs_test,  BATCH, shuffle=False)

# ——————————————— Model definition ———————————————

def build_model(num_classes):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(None, None, 3)),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(3),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(3),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(8, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

model = build_model(num_classes=len(label_encoder.classes_))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False         
)
model.summary()

# ——————————————— Training ———————————————

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 20,
)

# ——————————————— Save & Evaluate ———————————————

model.save("model_trained.keras")
print("Test accuracy:", model.evaluate(test_ds)[1])