# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:08:23 2026

@author: PS1-2026
"""

# ==============================================================================
# 0. IMPORTS
# ==============================================================================

import os
import random
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1. DATASET PATHS
# ==============================================================================

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

# Path where the trained base model will be saved for use in the pruning code
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\flexible_model_base_vgg16.keras"

# ==========================================
# REMOVE CORRUPTED IMAGES
# ==========================================

num_skipped = 0

for folder_name in ("Cat", "Dog"):

    folder_path = os.path.join(dataset_dir, folder_name)

    for fname in os.listdir(folder_path):

        fpath = os.path.join(folder_path, fname)

        try:
            with open(fpath, "rb") as f:
                is_jfif = tf.compat.as_bytes("JFIF") in f.peek(10)

        except:
            is_jfif = False

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

print("Deleted", num_skipped, "corrupted images")

# ==========================================
# LOAD DATASET
# ==========================================

random.seed(SEED)

# Paths to each class
cats_dir = os.path.join(dataset_dir, "Cat")
dogs_dir = os.path.join(dataset_dir, "Dog")

# Get image filenames
cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

# Randomly select 3000 from each class
cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

# Combine and shuffle
selected_files = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0=cats, 1=dogs

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)

selected_files, selected_labels = zip(*combined)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))    
    return image, label

full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)   # 8000

train_size = int(0.65 * dataset_size)   # 3900
val_size = int(0.15 * dataset_size)     # 900
test_size = dataset_size - train_size - val_size   # 1200

full_dataset = full_dataset.shuffle(
    dataset_size,
    seed=SEED,
    reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)

remaining = full_dataset.skip(train_size)

val_dataset = remaining.take(val_size)

test_dataset = remaining.skip(val_size)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# 2. DATA AUGMENTATION PIPELINE
# ==============================================================================
data_augmentation = tf.keras.Sequential(
    [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
]
)

# ==============================================================================
# 3. PREPROCESSING FUNCTIONS (Adjusted for local image pipeline)
# ==============================================================================
def preprocess_img(image, label):
    # Image resizing is bypassed here since it's already handled during dataset loading
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def train_preprocess(image, label):
    image, label = preprocess_img(image, label)
    image = data_augmentation(image, training=True)
    return image, label

# Construct highly optimized tf.data execution pipelines
train_dataset = train_dataset.shuffle(train_size // 5).map(
    train_preprocess, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(
    preprocess_img, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(
    preprocess_img, num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 4. MODEL ARCHITECTURE (VGG-16 from Scratch with Batch Normalization)
# ==============================================================================
# FIX: Two changes were made to align this model with the pruning code:
#
#   1. LAYER NAMING — All layers now use explicit names (conv_0, bn_0, leaky_0...)
#      matching the names used in create_flexible_vgg() in the pruning script.
#      The pruning code maps base→pruned weights using layer.name as the key,
#      so mismatched names cause the entire weight transfer to be silently skipped.
#
#   2. BATCHNORM PLACEMENT — BN is now placed after every individual Conv2D layer
#      (per-conv BN) instead of once at the end of each block.
#      The pruning code expects BN immediately after each Conv2D and uses
#      running_channel_mask to slice BN parameters right after slicing conv weights.
#      A block-end BN would receive the wrong mask and produce a shape mismatch.

def VGG_16_Scratch():
    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = inp

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_0")(inp)
    x = layers.BatchNormalization(name="bn_0")(x)
    x = layers.LeakyReLU(name="leaky_0")(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.LeakyReLU(name="leaky_1")(x)

    x = layers.MaxPooling2D((2, 2), name="pool_0")(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.LeakyReLU(name="leaky_2")(x)

    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.LeakyReLU(name="leaky_3")(x)

    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.LeakyReLU(name="leaky_4")(x)

    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_5")(x)
    x = layers.BatchNormalization(name="bn_5")(x)
    x = layers.LeakyReLU(name="leaky_5")(x)

    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_6")(x)
    x = layers.BatchNormalization(name="bn_6")(x)
    x = layers.LeakyReLU(name="leaky_6")(x)

    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_7")(x)
    x = layers.BatchNormalization(name="bn_7")(x)
    x = layers.LeakyReLU(name="leaky_7")(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_8")(x)
    x = layers.BatchNormalization(name="bn_8")(x)
    x = layers.LeakyReLU(name="leaky_8")(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_9")(x)
    x = layers.BatchNormalization(name="bn_9")(x)
    x = layers.LeakyReLU(name="leaky_9")(x)

    x = layers.MaxPooling2D((2, 2), name="pool_3")(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_10")(x)
    x = layers.BatchNormalization(name="bn_10")(x)
    x = layers.LeakyReLU(name="leaky_10")(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_11")(x)
    x = layers.BatchNormalization(name="bn_11")(x)
    x = layers.LeakyReLU(name="leaky_11")(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name="conv_12")(x)
    x = layers.BatchNormalization(name="bn_12")(x)
    x = layers.LeakyReLU(name="leaky_12")(x)

    x = layers.MaxPooling2D((2, 2), name="pool_4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(4096, name="dense_1")(x)
    x = layers.LeakyReLU(name="leaky_d1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)

    x = layers.Dense(4096, name="dense_2")(x)
    x = layers.LeakyReLU(name="leaky_d2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)

    x = layers.Dense(1, activation='sigmoid', name="output")(x)

    model = Model(inputs=inp, outputs=x)
    return model

model = VGG_16_Scratch()

# ==============================================================================
# 5. COSINE ANNEALING SCHEDULE SETUP
# ==============================================================================
EPOCHS = 100
steps_per_epoch = train_size // BATCH_SIZE
total_decay_steps = steps_per_epoch * EPOCHS

lr_schedule = CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=total_decay_steps,
    alpha=0.01  
)

# ==============================================================================
# 6. COMPILE AND TRAIN
# ==============================================================================
model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

tensorboardName = 'logs\\VGG16_baseModel'
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)

es = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=[es, tensorboard1])


# ==============================================================================
# 7. EVALUATION ON TEST SET
# ==============================================================================
print("\n>>> Evaluating model on the completely unseen Testing Dataset...")
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

# ==============================================================================
# 8. SAVE MODEL FOR PRUNING
# ==============================================================================
# Saved in .keras format (not .h5) — the pruning code uses custom layer shapes
# after structural pruning, which the older HDF5 format cannot handle correctly.
# The pruning script will load this file from SAVED_BASE_MODEL_PATH automatically.
model.save(SAVED_BASE_MODEL_PATH)
print(f"\n>>> Base model saved to: {SAVED_BASE_MODEL_PATH}")


'''
# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot Accuracy

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches='tight')
plt.show()
'''