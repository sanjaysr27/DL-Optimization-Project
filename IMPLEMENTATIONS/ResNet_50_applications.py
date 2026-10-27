# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 2026

@author: PS1-2026

ResNet50 (tf.keras.applications, ImageNet-pretrained backbone)
Transfer learning on the CSIR Leather defect dataset.

Mirrors the structure of the MobileNetV2 "applications" base script:
    Phase 1: freeze ResNet50 backbone, train new classifier head
    Phase 2: unfreeze everything, fine-tune end-to-end
    Save both the head-only model and the fully fine-tuned model
"""

import tensorflow as tf
import os
import random
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==========================================
# DATASET PATH
# ==========================================

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42

# Path to the local ImageNet-pretrained ResNet50 weights (no top),
# same style as the MobileNetV2 script's local .h5 file.

SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50.keras"


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

# Guard against sampling more images than exist in either class
N_PER_CLASS = 3992
assert len(nonDef_files) >= N_PER_CLASS, (
    f"Only {len(nonDef_files)} defect-free images found, need {N_PER_CLASS}"
)
assert len(def_files) >= N_PER_CLASS, (
    f"Only {len(def_files)} defective images found, need {N_PER_CLASS}"
)

# Randomly select N_PER_CLASS from each class
nonDef_files = random.sample(nonDef_files, N_PER_CLASS)
def_files = random.sample(def_files, N_PER_CLASS)

# Combine and shuffle
selected_files = nonDef_files + def_files
selected_labels = [0] * N_PER_CLASS + [1] * N_PER_CLASS   # 0=non-defective, 1=defective

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)

selected_files, selected_labels = zip(*combined)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_bmp(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet50.preprocess_input(image)
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
], name="data_augmentation")

# ==========================================
# RESNET50 (ImageNet backbone, head frozen first)
# ==========================================

base_model = ResNet50(
    weights=None,                 # loaded from local file below
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.load_weights(r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\TRAINING\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
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

tensorboardName = 'logs\\leather_base_applications_ResNet50'
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
# TRAIN (head only, backbone frozen)
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

# ==============================================================================
# FULL FINE-TUNE PHASE
# Reload the exact model just saved (guarantees architecture match) and
# unfreeze everything for end-to-end fine-tuning.
# ==============================================================================

model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
model.trainable = True

model.compile(
    optimizer=Adam(
        learning_rate=1e-5   # lower LR for fine-tuning a pretrained backbone
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboardName = 'logs\\leather_base_applications_ResNet50_full_train'
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

FINAL_SAVED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_ResNet50_Full_Train.keras"
model.save(FINAL_SAVED_MODEL_PATH)
print(f"\n>>> Fully fine-tuned model saved to: {FINAL_SAVED_MODEL_PATH}")