# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 15:07:38 2026

@author: PS1-2026
"""

#ResNet_50_Base

"""
ResNet50 (from scratch) — Base Model Training
Cat vs Dog binary classification

Mirrors the dataset pipeline and training setup of the MobileNetV2 base
script so the two architectures are directly comparable, and so the saved
model is structurally ready for the companion pruning script
(resnet50_structural_pruning.py).
"""

import os
import random

import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1. DATASET PATHS
# ==============================================================================
dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

# Path where the trained base model will be saved for use in the pruning code
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_base.keras"

# ==============================================================================
# 2. REMOVE CORRUPTED IMAGES
# ==============================================================================
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

# ==============================================================================
# 3. LOAD DATASET
# ==============================================================================
random.seed(SEED)

cats_dir = os.path.join(dataset_dir, "Cat")
dogs_dir = os.path.join(dataset_dir, "Dog")

cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

selected_files = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0=cats, 1=dogs

combined = list(zip(selected_files, selected_labels))
random.shuffle(combined)
selected_files, selected_labels = zip(*combined)


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)

full_dataset = full_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_size = len(selected_files)               # 8000
train_size = int(0.65 * dataset_size)             # 5200
val_size = int(0.15 * dataset_size)               # 1200
test_size = dataset_size - train_size - val_size  # 1600

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

# ==============================================================================
# 4. MODEL DEFINITION (ResNet50 from scratch — "v1.5" stride placement)
# ==============================================================================
#
# Stage configuration: (f1, f2, f3, num_blocks, first_block_stride)
#   f1 = channels of the 1x1 "reduce" conv      (bottleneck width)
#   f2 = channels of the 3x3 conv                (bottleneck width)
#   f3 = channels of the 1x1 "expand" conv       (block OUTPUT width)
#
# Stride is applied on the 3x3 conv (conv2) of the first block in each
# stage, and on the projection shortcut of that same block (ResNet v1.5
# convention — slightly more accurate than the original v1 placement).
# ==============================================================================

STAGE_CFG = [
    (64,  64,  256,  3, 1),   # stage 1 -> output 56x56x256
    (128, 128, 512,  4, 2),   # stage 2 -> output 28x28x512
    (256, 256, 1024, 6, 2),   # stage 3 -> output 14x14x1024
    (512, 512, 2048, 3, 2),   # stage 4 -> output  7x7x2048
]


def bottleneck_block(x, f1, f2, f3, stage, block, stride, conv_shortcut):
    """
    One ResNet50 bottleneck block:
        1x1 reduce -> 3x3 -> 1x1 expand   (+ identity or projection shortcut)

    conv_shortcut=True is used for the FIRST block of every stage, where
    either the spatial resolution changes (stride=2) or the channel count
    changes (e.g. 64 -> 256 going into stage 1), so a 1x1 projection conv
    is needed on the shortcut path to match shapes for the Add.
    """
    name_base = f"stage{stage}_block{block}"

    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(
            f3, 1, strides=stride, padding='same', use_bias=False,
            name=f"{name_base}_shortcut_conv"
        )(x)
        shortcut = layers.BatchNormalization(
            name=f"{name_base}_shortcut_bn"
        )(shortcut)

    # 1x1 reduce
    x = layers.Conv2D(
        f1, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn1")(x)
    x = layers.ReLU(name=f"{name_base}_relu1")(x)

    # 3x3 (stride lives here)
    x = layers.Conv2D(
        f2, 3, strides=stride, padding='same', use_bias=False,
        name=f"{name_base}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn2")(x)
    x = layers.ReLU(name=f"{name_base}_relu2")(x)

    # 1x1 expand
    x = layers.Conv2D(
        f3, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn3")(x)

    x = layers.Add(name=f"{name_base}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name_base}_out_relu")(x)

    return x


def ResNet50_Scratch():

    inp = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="input_layer"
    )

    x = data_augmentation(inp)

    # Stem
    x = layers.Conv2D(
        64, 7, strides=2, padding='same', use_bias=False,
        name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(
        3, strides=2, padding='same', name="stem_pool"
    )(x)

    # Stages
    for stage_idx, (f1, f2, f3, num_blocks, first_stride) in enumerate(
        STAGE_CFG, start=1
    ):
        for block_idx in range(1, num_blocks + 1):
            if block_idx == 1:
                x = bottleneck_block(
                    x, f1, f2, f3,
                    stage=stage_idx, block=block_idx,
                    stride=first_stride, conv_shortcut=True
                )
            else:
                x = bottleneck_block(
                    x, f1, f2, f3,
                    stage=stage_idx, block=block_idx,
                    stride=1, conv_shortcut=False
                )

    # Classifier Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)

    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    model = Model(inp, out)

    return model


model = ResNet50_Scratch()

model.summary()

for layer in model.layers:
    print(layer.name)

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

model.compile(
    optimizer=Adam(
        learning_rate=1e-4
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboardName = 'logs\\ResNet50_baseModel'
tensorboardNameApp = tensorboardName
tensorboard1 = TensorBoard(log_dir=tensorboardNameApp)

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[es, tensorboard1]
)

loss, accuracy = model.evaluate(test_dataset)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

print("\nBest Validation Accuracy:",
      max(history.history['val_accuracy']))

model.save(SAVED_BASE_MODEL_PATH)

print("\nBaseline model saved as", SAVED_BASE_MODEL_PATH)


for images, labels in test_dataset.take(1):

    preds = model.predict(images)

    print("\nPredictions:")
    print(preds[:10].flatten())

    print("\nActual Labels:")
    print(labels[:10].numpy())

    break