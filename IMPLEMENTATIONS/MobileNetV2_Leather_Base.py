# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:34:55 2026

@author: PS1-2026

MobileNetV2 base model training — leather defect / non-defect.

Two-phase training:
  Phase 1: MobileNetV2 backbone frozen (weights=None, trainable=False),
           only the head is trained. Saved to SAVED_BASE_MODEL_PATH.
  Phase 2: A fresh, architecturally identical model is built, its weights
           are loaded from the phase-1 save, the backbone is unfrozen
           (trainable=True), and the whole network is fine-tuned end to
           end. Saved to final_SAVED_BASE_MODEL_PATH — this is the file
           the rest of the pipeline (channel pruning / dropout / layer
           pruning) actually loads.

FIXES APPLIED vs. the original draft
-------------------------------------
1. Both training phases built a TensorBoard callback object
   (`tensorboard1`) but never actually passed it into `model.fit()`'s
   `callbacks=[...]` list — only `es` (EarlyStopping) was included. That
   meant no TensorBoard logs were ever written despite the setup code
   being there. Both `model.fit()` calls now include `tensorboard1`.
2. A leftover comment above the phase-2 backbone said "Load VGG16 base
   pre-trained on ImageNet" — this project builds MobileNetV2, not VGG16,
   and it isn't pretrained (weights=None). Comment corrected.

NOTE for the pipeline driver script:
  The variable names `SAVED_BASE_MODEL_PATH` (phase-1 save) and
  `final_SAVED_BASE_MODEL_PATH` (phase-2 / final save) must keep these
  exact names — the driver script rewrites both of them automatically
  before running this file, so whatever you set below is only used when
  running this script standalone.
"""

import tensorflow as tf
import os
import random
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==========================================
# DATASET PATH
# ==========================================

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train_frozen.keras"

# ==========================================
# LOAD DATASET
# ==========================================
random.seed(SEED)

# Paths to each class
nonDef_dir = os.path.join(dataset_dir, "0_DefectFree")
def_dir = os.path.join(dataset_dir, "1_Defective")

# Get image filenames
nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

# Randomly select 3992 from each class
nonDef_files = random.sample(nonDef_files, 3992)
def_files = random.sample(def_files, 3992)

# Combine and shuffle
selected_files = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992   # 0=non-defective, 1=defective

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)

selected_files, selected_labels = zip(*combined)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)               # 7984
train_size = int(0.65 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(
    dataset_size,
    seed=SEED,
    reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

print("Train:", train_size)
print("Validation:", val_size)
print("Test:", test_size)

train_dataset = (
    train_dataset
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================================
# PHASE 1 — MOBILENETV2 (NO PRETRAINING, FROZEN BACKBONE)
# ==========================================

base_model = MobileNetV2(
    weights=None,                  # Train from scratch
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.load_weights(r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
base_model.trainable = False

# ==========================================
# MODEL
# ==========================================

inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inp)

x = base_model(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation='relu'
)(x)

x = layers.Dropout(0.5)(x)

out = layers.Dense(
    1,
    activation='sigmoid'
)(x)

model = Model(inp, out)

# ==========================================
# SUMMARY
# ==========================================

model.summary()

print(
    "Trainable Parameters:",
    sum(tf.keras.backend.count_params(w)
        for w in model.trainable_weights)
)

print(
    "Non-Trainable Parameters:",
    sum(tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights)
)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# TENSORBOARD
# ==========================================

tensorboardName = 'logs\\leather_base_applications_MobileNetV2'   #change model
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)

# ==========================================
# EARLY STOPPING
# ==========================================

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ==========================================
# TRAIN
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[es, tensorboard1]
)

# ==========================================
# EVALUATE
# ==========================================

loss, accuracy = model.evaluate(test_dataset)

print("\nTest Loss:", loss)

print("Test Accuracy:", accuracy)

print(
    "\nBest Validation Accuracy:",
    max(history.history['val_accuracy'])
)

model.save(SAVED_BASE_MODEL_PATH)
print(f"\n>>> Base model saved to: {SAVED_BASE_MODEL_PATH}")

# ==========================================
# PHASE 2 — FULL FINE-TUNE (UNFREEZE BACKBONE)
# ==========================================
# Build a fresh, architecturally identical MobileNetV2 model, then load
# the phase-1 weights into it before unfreezing and continuing training.

model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
model.trainable = True

model.summary()


model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboardName = 'logs\\leather_base_applications_MobileNetV2_full_train'   #change model
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)


es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[es, tensorboard1]
)

# ==============================================================================
# TESTING PHASE (Evaluation on Unseen Data)
# ==============================================================================
print("\n>>> Evaluating model on the completely unseen Testing Dataset...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


final_SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_MobileNetV2_Full_Train.keras"
model.save(final_SAVED_BASE_MODEL_PATH)
print(f"\n>>> Base model saved to: {final_SAVED_BASE_MODEL_PATH}")

import json
metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)}
with open("last_run_metrics.json", "w") as f:
    json.dump(metrics, f)