# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:07:46 2026

@author: PS1-2026
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
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"

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

# Paths to each class
cats_dir = os.path.join(dataset_dir, "Cat")
dogs_dir = os.path.join(dataset_dir, "Dog")

# Get image filenames
cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

# Randomly select 4000 from each class
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
])

# ==============================================================================
# 4. MODEL DEFINITION (MobileNetV2 from scratch)
# ==============================================================================

# Inverted Residual Block
def inverted_residual_block(x,
                            expansion,
                            filters,
                            stride,
                            block_id):

    inp = x
    input_channels = x.shape[-1]

    # Expansion
    x = layers.Conv2D(
        input_channels * expansion,
        1,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_expand"
    )(x)

    x = layers.BatchNormalization(
        name=f"block_{block_id}_expand_bn"
    )(x)
    x = layers.ReLU(
        max_value=6,
        name=f"block_{block_id}_expand_relu"
    )(x)

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(
        3,
        strides=stride,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_depthwise"
    )(x)

    x = layers.BatchNormalization(
        name=f"block_{block_id}_depthwise_bn"
    )(x)
    x = layers.ReLU(
        max_value=6,
        name=f"block_{block_id}_depthwise_relu"
    )(x)

    # Projection
    x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        name=f"block_{block_id}_project"
    )(x)

    x = layers.BatchNormalization(
        name=f"block_{block_id}_project_bn"
    )(x)

    # Residual Connection
    if stride == 1 and input_channels == filters:
        x = layers.Add(
            name=f"block_{block_id}_add"
        )([inp, x])

    return x


def MobileNetV2_Scratch():

    inp = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3),
        name="input_layer"
    )

    x = data_augmentation(inp)
    # Initial Convolution
    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding='same',
        use_bias=False,
        name="stem_conv"
    )(x)

    x = layers.BatchNormalization(
        name="stem_bn"
    )(x)
    x = layers.ReLU(
        max_value=6,
        name="stem_relu"
    )(x)

    # MobileNetV2 Blocks

    x = inverted_residual_block(
        x,
        expansion=1,
        filters=16,
        stride=1,
        block_id=1
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=24,
        stride=2,
        block_id=2
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=24,
        stride=1,
        block_id=3
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=32,
        stride=2,
        block_id=4
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=32,
        stride=1,
        block_id=5
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=32,
        stride=1,
        block_id=6
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=64,
        stride=2,
        block_id=7
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=64,
        stride=1,
        block_id=8
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=64,
        stride=1,
        block_id=9
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=64,
        stride=1,
        block_id=10
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=96,
        stride=1,
        block_id=11
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=96,
        stride=1,
        block_id=12
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=96,
        stride=1,
        block_id=13
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=160,
        stride=2,
        block_id=14
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=160,
        stride=1,
        block_id=15
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=160,
        stride=1,
        block_id=16
    )

    x = inverted_residual_block(
        x,
        expansion=6,
        filters=320,
        stride=1,
        block_id=17
    )

    # Final Conv
    x = layers.Conv2D(
        1280,
        1,
        padding='same',
        use_bias=False,
        name="final_conv"
    )(x)

    x = layers.BatchNormalization(
        name="final_bn"
    )(x)
    x = layers.ReLU(
        max_value=6,
        name="final_relu"
    )(x)

    # Classifier Head

    x = layers.GlobalAveragePooling2D(
        name="global_avg_pool"
    )(x)

    x = layers.Dense(
        128,
        activation='relu',
        name="fc1"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    out = layers.Dense(
        1,
        activation='sigmoid',
        name="classifier"
    )(x)

    model = Model(inp, out)

    return model


model = MobileNetV2_Scratch()

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
        learning_rate=1e-3
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboardName = 'logs\\MobileNetV2_baseModel'
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