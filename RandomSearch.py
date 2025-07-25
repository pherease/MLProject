import os, random, numpy as np
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import matplotlib.pyplot as plt
import utils  # your local utils.py

from sklearn.utils import class_weight
from tensorboard import program
from keras_tuner import RandomSearch
from datetime import datetime



SEED = 424
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

mixed_precision.set_global_policy('mixed_float16')

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

tf.config.optimizer.set_jit(False)




TRAIN_CSV = "./split_data/train.csv"
VAL_CSV = "./split_data/val.csv"
TEST_CSV = "./split_data/test.csv"
BATCH = 8
IMG_SIZE = 256
autotune = True


train_paths, train_labels, le = utils.load_paths_and_labels(TRAIN_CSV)
val_paths, val_labels = utils.encode(VAL_CSV, le)
test_paths, test_labels = utils.encode(TEST_CSV, le)

utils.show_dataset_class_distribution("TRAIN", train_labels)
utils.show_dataset_class_distribution("VAL  ", val_labels)
utils.show_dataset_class_distribution("TEST ", test_labels)



augment_layer = utils.make_augment()

train_ds = utils.make_dataset(train_paths, train_labels, BATCH, shuffle=True, autotune=autotune)
train_ds = train_ds.map(
    lambda imgs, labs: (augment_layer(imgs, training=True), labs),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = utils.make_dataset(val_paths, val_labels, BATCH, shuffle=True, autotune=autotune)
test_ds = utils.make_dataset(test_paths, test_labels, BATCH, shuffle=False, autotune=autotune)

utils.show_image_of_batch(train_ds, le)

def build_model(hp):
    # LEARN_RATE_INIT = hp.Choice("LEARN_RATE_INIT", values = [1e-2, 1e-3, 1e-4], default = 1e-3)
    LEARN_RATE_INIT = 1e-3

    filters = 64

    # dense2 = hp.Choice("dense2", values = [16, 32, 64], default = 16)

    kernel_size = 3

    # dense1 = hp.Choice("dense1", values = [0, 32], default = 0)
    dense1 = 32

    # dropout1 = 0.0

    # dropout2 = 0.0

    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (kernel_size, kernel_size), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//4, (kernel_size, kernel_size), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters//2, (kernel_size, kernel_size), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),
        # layers.BatchNormalization() if use_lastbatchnorm else layers.Identity(),
        # layers.Dropout(dropout1),

        # layers.Dense(dense1, activation='relu') if dense1 != 0 else layers.Identity(),
        # layers.Dropout(dropout2),

        layers.Dense(len(le.classes_), activation='softmax', dtype='float32')
    ])

    def sparse_focal_loss(gamma=1.0, alpha=0.25):
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int64)
            batch_idxs = tf.range(tf.shape(y_pred)[0], dtype=y_true.dtype)
            indices = tf.stack([batch_idxs, y_true], axis=1)
            p_t = tf.gather_nd(y_pred, indices)
            alpha_factor = tf.where(tf.equal(y_true, 0), 1.0 - alpha, alpha)
            focal_weight = alpha_factor * tf.pow(1.0 - p_t, gamma)
            loss = -focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
            return tf.reduce_mean(loss)
        return loss_fn

    alpha = hp.Choice("alpha", values = [0.25, 0.50, 1.00], default = 1.00)
    gamma = hp.Choice("gamma", values = [0.5, 1.0, 2.0, 5.0, 20.0], default = 2.0)
    model.compile(optimizer=keras.optimizers.Adam(LEARN_RATE_INIT),
                  loss=sparse_focal_loss(gamma = gamma, alpha = alpha),
                  metrics=["accuracy"])
    return model


lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", 
                                             factor = 0.25, 
                                             patience = 5, 
                                             min_lr = 1e-6,
                                             min_delta = 0.0)

dyn_cb = utils.DynamicMinDelta(reduce_cb = lr_cb, ratio=0.01)



tuner = RandomSearch(
    build_model,
    objective = "val_accuracy",
    max_trials = 6,
    executions_per_trial = 1,
    directory = "tuner_logs",
    project_name = "trashnet_tune",
    seed = SEED
)


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)


tuner.search(
    train_ds,
    validation_data = val_ds,
    epochs = 50,
    callbacks=[tensorboard_cb,
               dyn_cb,
               lr_cb
               ]
)

best_hp = tuner.get_best_hyperparameters()[0]
print("Best hyperparameters:", best_hp.values)

model = tuner.hypermodel.build(best_hp)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs = 100,
    callbacks=[
        tensorboard_cb,
        utils.ConfusionMatrixSaver(val_ds, le.classes_, every=3),
        utils.checkpoint_cb(),
        utils.csv_logger_cb(),
        dyn_cb,
        lr_cb        
    ]
)
