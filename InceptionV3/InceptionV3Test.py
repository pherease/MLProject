import os, random, numpy as np
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from tensorflow import keras
from keras import layers, mixed_precision
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.utils import class_weight
from datetime import datetime
import utils  # your local utils.py

# Reproducibility
SEED = 424
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision for performance
mixed_precision.set_global_policy('mixed_float16')

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit(False)

# Paths & parameters
TRAIN_CSV = "./split_data/train.csv"
VAL_CSV   = "./split_data/val.csv"
TEST_CSV  = "./split_data/test.csv"
BATCH     = 8
IMG_SIZE  = 256  # InceptionV3 will resize accordingly
autotune  = True
EPOCHS    = 50

# Load filepaths & labels
train_paths, train_labels, le = utils.load_paths_and_labels(TRAIN_CSV)
val_paths, val_labels   = utils.encode(VAL_CSV, le)
test_paths, test_labels = utils.encode(TEST_CSV, le)

# Show class distribution
utils.show_dataset_class_distribution("TRAIN", train_labels)
utils.show_dataset_class_distribution("VAL  ", val_labels)
utils.show_dataset_class_distribution("TEST ", test_labels)

# Data augmentation & pipelines
augment_layer = utils.make_augment()

train_ds = utils.make_dataset(train_paths, train_labels, BATCH, shuffle=True, autotune=autotune)
train_ds = train_ds.map(
    lambda imgs, labs: (augment_layer(imgs, training=True), labs),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = utils.make_dataset(val_paths, val_labels, BATCH, shuffle=True, autotune=autotune)
test_ds = utils.make_dataset(test_paths, test_labels, BATCH, shuffle=False, autotune=autotune)

# utils.show_image_of_batch(train_ds, le)
# TEST: dont show images in this script or save batch.png


# Callbacks
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.25, patience=3, min_lr=1e-6, min_delta=0.005
)

# Define the model building function
def build_model(fine_tune_at, dropout_rate, learning_rate):
    base_model = InceptionV3(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    # Freeze all layers initially
    base_model.trainable = False
    # Unfreeze last `fine_tune_at` layers for fine-tuning
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(len(le.classes_), activation='softmax', dtype='float32')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Instantiate and train model
dropout_rate = 0.2
fine_tune_at = 200
learning_rate = 1e-4
model = build_model(fine_tune_at, dropout_rate, learning_rate)

print(model.summary())

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=lr_cb
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# Save final model
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'inceptionV3_test.h5'))