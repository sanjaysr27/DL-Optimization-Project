# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:11:23 2026

@author: PS1-2026
"""
"""
VGG16 — SpatialDropout2D After Every Conv Layer + Configurable Dense Dropout
=============================================================================

Architecture changes vs. base model:
--------------------------------------
  • SpatialDropout2D(CONV_DROPOUT_RATE) inserted after every Conv2D→BN→LeakyReLU
    unit (13 insertions, named sdrop_0 … sdrop_12).
  • Dropout(DENSE_DROPOUT_RATE) replaces Dropout(0.5) in both dense units.
  • All other layers and their names are unchanged.

Workflow:
  1. Load     base model from SAVED_BASE_MODEL_PATH
  2. Verify   all expected layer names are present
  3. Build    new VGG16 with dropout inserted
  4. Transfer Conv2D, BN, Dense weights by layer name
  5. Evaluate before fine-tuning  (dropout OFF at eval → comparable to base)
  6. Fine-tune with fresh CosineDecay schedule (dropout ON during training)
  7. Evaluate after fine-tuning
  8. Save     to DROPOUT_MODEL_PATH
  9. Prediction sanity check on one test batch
"""

import os
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\flexible_model_base_vgg16.keras"
DROPOUT_MODEL_PATH    = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\vgg16_SpatialDropout30.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# Dropout rate for ALL SpatialDropout2D layers (one after each of the 13 conv layers)
CONV_DROPOUT_RATE  = 0.30

# Dropout rate for both Dense layers in the classifier head
DENSE_DROPOUT_RATE = 0.30

# Fine-tuning hyper-parameters
FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 5
FINETUNE_INIT_LR  = 1e-5

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()

# Verify all weight-bearing layer names exist in the base model before proceeding.
# Crashes immediately with a clear message rather than silently skipping weights.
expected_names = (
    [f"conv_{i}" for i in range(13)] +   # 13 Conv2D layers
    [f"bn_{i}"   for i in range(13)] +   # 13 BatchNorm layers
    ["dense_1", "dense_2", "output"]      # classifier head
    # Note: drop_1, drop_2 exist in the base model but carry no weights —
    #       they do NOT need to appear here.
)

missing = [n for n in expected_names
           if not any(l.name == n for l in base_model.layers)]

if missing:
    raise ValueError(
        f"Base model is missing expected layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to the correct file."
    )
print("Layer-name check passed — all expected layers present.")

# ==============================================================================
# 3.  DATASET  (pipeline order matches base model script exactly)
# ==============================================================================
random.seed(SEED)

cats_dir = os.path.join(DATASET_DIR, "Cat")
dogs_dir = os.path.join(DATASET_DIR, "Dog")

cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

selected_files  = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0 = Cat, 1 = Dog

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)

dataset_size = len(selected_files)          # 8000
train_size   = int(0.65 * dataset_size)     # 5200
val_size     = int(0.15 * dataset_size)     # 1200
test_size    = dataset_size - train_size - val_size   # 1600


def load_image(path, label):
    # Resizing is done here so all downstream ops see (224, 224, 3) tensors.
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle BEFORE splitting — same order as base model
full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

# Batch FIRST, then preprocess/augment — same order as base model
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])


def preprocess_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def train_preprocess(image, label):
    image, label = preprocess_img(image, label)
    image = data_augmentation(image, training=True)
    return image, label


train_dataset = (train_dataset
                 .shuffle(train_size // 5)
                 .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset  = (val_dataset
                .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))



# ==============================================================================
# 4.  BUILD NEW VGG16
#     SpatialDropout2D(CONV_DROPOUT_RATE) is placed after EVERY individual
#     Conv2D → BatchNorm → LeakyReLU sequence (13 times total, named sdrop_0
#     through sdrop_12).  Dropout(DENSE_DROPOUT_RATE) follows each Dense unit.
#
#     All weight-bearing layer names (conv_N, bn_N, dense_N, output) are kept
#     identical to the base model so weights transfer cleanly by name lookup.
#     The new sdrop_N layers carry no weights — they are invisible to the
#     weight-transfer loop.
# ==============================================================================
def VGG16_With_Dropout():
    inp = layers.Input(shape=(224, 224, 3), name="input_layer")

    # ── Block 1 — 2 × Conv(64) ────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_0")(inp)
    x = layers.BatchNormalization(name="bn_0")(x)
    x = layers.LeakyReLU(name="leaky_0")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_0")(x)   # NEW

    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.LeakyReLU(name="leaky_1")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_1")(x)   # NEW

    x = layers.MaxPooling2D((2, 2), name="pool_0")(x)

    # ── Block 2 — 2 × Conv(128) ───────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.LeakyReLU(name="leaky_2")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_2")(x)   # NEW

    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.LeakyReLU(name="leaky_3")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_3")(x)   # NEW

    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)

    # ── Block 3 — 3 × Conv(256) ───────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.LeakyReLU(name="leaky_4")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_4")(x)   # NEW

    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_5")(x)
    x = layers.BatchNormalization(name="bn_5")(x)
    x = layers.LeakyReLU(name="leaky_5")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_5")(x)   # NEW

    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_6")(x)
    x = layers.BatchNormalization(name="bn_6")(x)
    x = layers.LeakyReLU(name="leaky_6")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_6")(x)   # NEW

    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)

    # ── Block 4 — 3 × Conv(512) ───────────────────────────────────────────────
    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_7")(x)
    x = layers.BatchNormalization(name="bn_7")(x)
    x = layers.LeakyReLU(name="leaky_7")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_7")(x)   # NEW

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_8")(x)
    x = layers.BatchNormalization(name="bn_8")(x)
    x = layers.LeakyReLU(name="leaky_8")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_8")(x)   # NEW

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_9")(x)
    x = layers.BatchNormalization(name="bn_9")(x)
    x = layers.LeakyReLU(name="leaky_9")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_9")(x)   # NEW

    x = layers.MaxPooling2D((2, 2), name="pool_3")(x)

    # ── Block 5 — 3 × Conv(512) ───────────────────────────────────────────────
    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_10")(x)
    x = layers.BatchNormalization(name="bn_10")(x)
    x = layers.LeakyReLU(name="leaky_10")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_10")(x)  # NEW

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_11")(x)
    x = layers.BatchNormalization(name="bn_11")(x)
    x = layers.LeakyReLU(name="leaky_11")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_11")(x)  # NEW

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_12")(x)
    x = layers.BatchNormalization(name="bn_12")(x)
    x = layers.LeakyReLU(name="leaky_12")(x)
    x = layers.SpatialDropout2D(CONV_DROPOUT_RATE, name="sdrop_12")(x)  # NEW

    x = layers.MaxPooling2D((2, 2), name="pool_4")(x)

    # ── Classifier Head ───────────────────────────────────────────────────────
    x   = layers.GlobalAveragePooling2D(name="gap")(x)
    x   = layers.Dense(4096, name="dense_1")(x)
    x   = layers.LeakyReLU(name="leaky_d1")(x)
    x   = layers.Dropout(DENSE_DROPOUT_RATE, name="drop_1")(x)
    x   = layers.Dense(4096, name="dense_2")(x)
    x   = layers.LeakyReLU(name="leaky_d2")(x)
    x   = layers.Dropout(DENSE_DROPOUT_RATE, name="drop_2")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


new_model = VGG16_With_Dropout()
new_model.summary()

# ==============================================================================
# 5.  TRANSFER WEIGHTS FROM BASE MODEL → NEW MODEL
# ==============================================================================
base_weights = {
    layer.name: layer.get_weights()
    for layer in base_model.layers
    if len(layer.get_weights()) > 0      # SpatialDropout2D / MaxPool / LeakyReLU
}                                         # have no weights and are excluded here

transferred = []
skipped     = []

for layer in new_model.layers:
    if len(layer.get_weights()) == 0:
        continue   # sdrop_N, pool_N, leaky_N — no weights to transfer
    if layer.name in base_weights:
        layer.set_weights(base_weights[layer.name])
        transferred.append(layer.name)
    else:
        skipped.append(layer.name)

print(f"\nWeight-bearing layers in base model : {len(base_weights)}")
print(f"Transferred to new model            : {len(transferred)}")
for name in transferred:
    print(f"   {name}")

if skipped:
    print(f"\n  No base weights found for: {skipped}")
    print("   These layers keep random initialisation.")
else:
    print("\nAll weight-bearing layers matched successfully.")

# ==============================================================================
# 6.  EVALUATE BEFORE FINE-TUNING
#     Compile with a fixed scalar LR for this eval — the LR value is irrelevant
#     during evaluate() but we need a compiled model.  A fresh CosineDecay
#     schedule is built separately for fine-tuning below so that no decay
#     steps are consumed here.
# ==============================================================================
new_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_INIT_LR),   # fixed scalar — eval only
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

loss_before, acc_before = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (before fine-tuning) : {loss_before:.4f}")
print(f"Test accuracy (before fine-tuning) : {acc_before * 100:.2f}%")
print("Expected: close to base model accuracy (~93.25%).")
print("SpatialDropout2D and Dropout are both DISABLED at eval time,")
print("so inference is identical to the original base model.")

# ==============================================================================
# 7.  FINE-TUNE WITH DROPOUT ACTIVE
#     Build a fresh CosineDecay schedule here so decay starts from step 0
#     (not partially consumed by the evaluate() call above).
# ==============================================================================
steps_per_epoch_ft = train_size // BATCH_SIZE
total_ft_steps     = steps_per_epoch_ft * FINETUNE_EPOCHS

lr_schedule_ft = CosineDecay(
    initial_learning_rate=FINETUNE_INIT_LR,
    decay_steps=total_ft_steps,
    alpha=0.01   # floor: FINETUNE_INIT_LR * 0.01 at end of schedule
)

# Recompile with the cosine-decaying LR for training
new_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_ft),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

print(f"NOTE: CONV_DROPOUT_RATE = {CONV_DROPOUT_RATE}.")
tensorboard_dropout = TensorBoard(log_dir='logs\\VGG16_SpatialDropout30')

es_dropout = EarlyStopping(
    monitor='val_loss',
    patience=FINETUNE_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history_ft = new_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_dropout, tensorboard_dropout],
    verbose=1
)

# ==============================================================================
# 8.  FINAL EVALUATION
# ==============================================================================
loss_after, acc_after = new_model.evaluate(test_dataset, verbose=1)
print(f"\nTest loss     (after fine-tuning)  : {loss_after:.4f}")
print(f"Test accuracy (after fine-tuning)  : {acc_after * 100:.2f}%")
print(f"Best val accuracy during fine-tune : "
      f"{max(history_ft.history['val_accuracy']) * 100:.2f}%")

# ==============================================================================
# 9.  SUMMARY
# ==============================================================================
base_params = sum(
    tf.keras.backend.count_params(w) for w in base_model.trainable_weights
)
new_params = sum(
    tf.keras.backend.count_params(w) for w in new_model.trainable_weights
)

print(f"\nBase model parameters  : {base_params:>12,}")
print(f"New  model parameters  : {new_params:>12,}")
print(f"Difference             : {new_params - base_params:>+12,}  (should be 0)")
print(f"New  model test accuracy (before fine-tune) : {acc_before * 100:.2f}%")
print(f"New  model test accuracy (after  fine-tune) : {acc_after  * 100:.2f}%")
print("\nDropout configuration:")
print(f"  Conv layers  — SpatialDropout2D rate : {CONV_DROPOUT_RATE}"
      f"  (sdrop_0 … sdrop_12, one after each of the 13 conv layers)")
print(f"  Dense layers — Dropout          rate : {DENSE_DROPOUT_RATE}"
      f"  (drop_1, drop_2)")

# ==============================================================================
# 10.  SAVE
# ==============================================================================
new_model.save(DROPOUT_MODEL_PATH)
print(f"\nModel saved to: {DROPOUT_MODEL_PATH}")

