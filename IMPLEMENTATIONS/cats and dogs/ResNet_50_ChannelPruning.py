# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:08:35 2026

@author: PS1-2026
"""
#ResNet_50_Pruing

"""
Structural Channel Pruning — ResNet50 (Cat vs Dog)
Compatible with resnet50_base.keras produced by resnet50_base_training.py

PRUNING STRATEGY
----------------
Every ResNet50 bottleneck block has the form:

    input ──┬─────────────────────────────────────────────► Add ─► ReLU ─► output
            │                                                  ▲
            └─[1x1 reduce]─[3x3]─[1x1 expand]───────────────────┘
                  f1          f2        f3
            (shortcut, identity OR 1x1 projection conv, also -> f3)

Unlike MobileNetV2's depthwise block (where expand and depthwise share a
single channel count because depthwise conv doesn't change channel count),
a ResNet50 bottleneck has TWO independently prunable internal widths:

  keep1 : channels between conv1 (reduce) and conv2 (3x3)   — width f1
  keep2 : channels between conv2 (3x3) and conv3 (expand)   — width f2

The block's OUTPUT width (f3 = 256/512/1024/2048) is NEVER pruned, because
it must exactly match the shortcut path (identity or projection) for the
Add to work. This means:
  - the projection/shortcut convs never need to be touched
  - the classifier head (which reads the final 2048-wide feature map)
    never needs to be touched either

Importance criterion: L1 norm of each output filter (sum of |weights|),
computed independently for conv1 (-> keep1) and conv2 (-> keep2).

After rebuilding the smaller graph and transplanting pruned weights, the
model is fine-tuned to recover accuracy.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1.  CONFIGURATION
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_base.keras"
PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\resnet50_pruned80.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# Fraction of bottleneck channels to REMOVE per layer
PRUNING_RATIO = 0.80  

# Fine-tuning hyper-parameters
FINETUNE_LR        = 1e-4
FINETUNE_EPOCHS    = 50
FINETUNE_PATIENCE  = 5

# Stage configuration — MUST match resnet50_base_training.py exactly
# (f1, f2, f3, num_blocks, first_block_stride)
STAGE_CFG = [
    (64,  64,  256,  3, 1),
    (128, 128, 512,  4, 2),
    (256, 256, 1024, 6, 2),
    (512, 512, 2048, 3, 2),
]

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================

base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")


# ==============================================================================
# 3.  DATASET  (identical pipeline to base-model training)
# ==============================================================================


random.seed(SEED)

cats_dir = os.path.join(DATASET_DIR, "Cat")
dogs_dir = os.path.join(DATASET_DIR, "Dog")

cat_files = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
dog_files = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

cat_files = random.sample(cat_files, 4000)
dog_files = random.sample(dog_files, 4000)

selected_files  = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000

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
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

train_dataset = (train_dataset.shuffle(1000)
                              .batch(BATCH_SIZE)
                              .prefetch(tf.data.AUTOTUNE))
val_dataset   = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 4.  COMPUTE CHANNEL IMPORTANCE  →  KEEP INDICES (keep1, keep2 per block)
# ==============================================================================



def l1_keep_indices(conv_layer, pruning_ratio):
    """
    Rank a Conv2D layer's output channels by L1 norm and return the
    sorted indices of the channels to KEEP.
    """
    kernel     = conv_layer.get_weights()[0]             # (kH, kW, in_ch, out_ch)
    importance = np.sum(np.abs(kernel), axis=(0, 1, 2))  # (out_ch,)
    n_total    = len(importance)
    n_keep     = max(1, int(round(n_total * (1.0 - pruning_ratio))))
    keep_idx   = np.sort(np.argsort(importance)[::-1][:n_keep])
    return keep_idx


# block_keep[name_base] = {"keep1": ..., "keep2": ...}
block_keep = {}

print(f"\n  {'Block':<16} {'f1 kept':>10} {'f2 kept':>10}")
print("  " + "-" * 40)

for stage_idx, (f1, f2, f3, num_blocks, _stride) in enumerate(STAGE_CFG, start=1):
    for block_idx in range(1, num_blocks + 1):
        name_base = f"stage{stage_idx}_block{block_idx}"

        conv1_layer = base_model.get_layer(f"{name_base}_conv1")
        conv2_layer = base_model.get_layer(f"{name_base}_conv2")

        keep1 = l1_keep_indices(conv1_layer, PRUNING_RATIO)
        keep2 = l1_keep_indices(conv2_layer, PRUNING_RATIO)

        block_keep[name_base] = {"keep1": keep1, "keep2": keep2}

        print(f"  {name_base:<16} {len(keep1):>4}/{f1:<5} "
              f"{len(keep2):>4}/{f2:<5}")

# ==============================================================================
# 5.  WEIGHT-EXTRACTION HELPERS
# ==============================================================================

def extract_conv2d_kernel(layer, in_keep=None, out_keep=None):
    """Slice a Conv2D kernel (no bias) along input and/or output channel axes."""
    k = layer.get_weights()[0]
    if in_keep is not None:
        k = k[:, :, in_keep, :]
    if out_keep is not None:
        k = k[:, :, :, out_keep]
    return [k]


def extract_batchnorm_weights(layer, ch_keep):
    """Slice [gamma, beta, moving_mean, moving_var] to kept channels."""
    gamma, beta, mm, mv = layer.get_weights()
    return [gamma[ch_keep], beta[ch_keep], mm[ch_keep], mv[ch_keep]]

# ==============================================================================
# 6.  BUILD PRUNED MODEL ARCHITECTURE
# ==============================================================================


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def build_pruned_bottleneck_block(x, pruned_f1, pruned_f2, f3,
                                   stage, block, stride, conv_shortcut):
    """
    Same topology as the base bottleneck_block, but conv1/conv2 widths
    are shrunk to pruned_f1 / pruned_f2.  f3 (block output) and the
    shortcut path are UNCHANGED, so the Add always lines up.
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

    x = layers.Conv2D(
        pruned_f1, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn1")(x)
    x = layers.ReLU(name=f"{name_base}_relu1")(x)

    x = layers.Conv2D(
        pruned_f2, 3, strides=stride, padding='same', use_bias=False,
        name=f"{name_base}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn2")(x)
    x = layers.ReLU(name=f"{name_base}_relu2")(x)

    x = layers.Conv2D(
        f3, 1, padding='same', use_bias=False,
        name=f"{name_base}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{name_base}_bn3")(x)

    x = layers.Add(name=f"{name_base}_add")([shortcut, x])
    x = layers.ReLU(name=f"{name_base}_out_relu")(x)

    return x


def build_pruned_resnet50(block_keep_dict):

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x   = data_augmentation(inp)

    # Stem — unchanged
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="stem_pool")(x)

    for stage_idx, (f1, f2, f3, num_blocks, first_stride) in enumerate(
        STAGE_CFG, start=1
    ):
        for block_idx in range(1, num_blocks + 1):
            name_base  = f"stage{stage_idx}_block{block_idx}"
            pruned_f1  = len(block_keep_dict[name_base]["keep1"])
            pruned_f2  = len(block_keep_dict[name_base]["keep2"])

            if block_idx == 1:
                x = build_pruned_bottleneck_block(
                    x, pruned_f1, pruned_f2, f3,
                    stage=stage_idx, block=block_idx,
                    stride=first_stride, conv_shortcut=True
                )
            else:
                x = build_pruned_bottleneck_block(
                    x, pruned_f1, pruned_f2, f3,
                    stage=stage_idx, block=block_idx,
                    stride=1, conv_shortcut=False
                )

    # Classifier head — input width (f3 of last stage = 2048) never
    # changed by this pruning scheme, so the head is structurally
    # identical to the base model.
    x   = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x   = layers.Dense(128, activation='relu', name="fc1")(x)
    x   = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


pruned_model = build_pruned_resnet50(block_keep)
pruned_model.summary()

# ==============================================================================
# 7.  TRANSFER PRUNED WEIGHTS  (base model  →  pruned model)
# ==============================================================================


# ── Stem (unchanged) ────────────────────────────────────────────────────────
for lname in ("stem_conv", "stem_bn"):
    pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

# ── Stages / blocks ─────────────────────────────────────────────────────────
for stage_idx, (f1, f2, f3, num_blocks, first_stride) in enumerate(
    STAGE_CFG, start=1
):
    for block_idx in range(1, num_blocks + 1):
        name_base = f"stage{stage_idx}_block{block_idx}"
        keep1 = block_keep[name_base]["keep1"]
        keep2 = block_keep[name_base]["keep2"]
        conv_shortcut = (block_idx == 1)

        # conv1 (reduce): slice OUTPUT channels to keep1
        pruned_model.get_layer(f"{name_base}_conv1").set_weights(
            extract_conv2d_kernel(
                base_model.get_layer(f"{name_base}_conv1"), out_keep=keep1
            )
        )
        pruned_model.get_layer(f"{name_base}_bn1").set_weights(
            extract_batchnorm_weights(
                base_model.get_layer(f"{name_base}_bn1"), keep1
            )
        )

        # conv2 (3x3): slice INPUT channels to keep1, OUTPUT channels to keep2
        pruned_model.get_layer(f"{name_base}_conv2").set_weights(
            extract_conv2d_kernel(
                base_model.get_layer(f"{name_base}_conv2"),
                in_keep=keep1, out_keep=keep2
            )
        )
        pruned_model.get_layer(f"{name_base}_bn2").set_weights(
            extract_batchnorm_weights(
                base_model.get_layer(f"{name_base}_bn2"), keep2
            )
        )

        # conv3 (expand): slice INPUT channels to keep2; OUTPUT (f3) unchanged
        pruned_model.get_layer(f"{name_base}_conv3").set_weights(
            extract_conv2d_kernel(
                base_model.get_layer(f"{name_base}_conv3"), in_keep=keep2
            )
        )
        # bn3: output width unchanged -> copy as-is
        pruned_model.get_layer(f"{name_base}_bn3").set_weights(
            base_model.get_layer(f"{name_base}_bn3").get_weights()
        )

        # Shortcut path (only present on the first block of each stage):
        # input width and output width (f3) are both unchanged -> copy as-is
        if conv_shortcut:
            pruned_model.get_layer(f"{name_base}_shortcut_conv").set_weights(
                base_model.get_layer(f"{name_base}_shortcut_conv").get_weights()
            )
            pruned_model.get_layer(f"{name_base}_shortcut_bn").set_weights(
                base_model.get_layer(f"{name_base}_shortcut_bn").get_weights()
            )

# ── Classifier head (unchanged — input width never pruned) ────────────────
for lname in ("fc1", "classifier"):
    pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

print("  All weights transplanted successfully.")

# ==============================================================================
# 8.  BASELINE EVALUATION  (pruned model, before fine-tuning)
# ==============================================================================


pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

loss_before, acc_before = pruned_model.evaluate(test_dataset, verbose=1)
print(f"\n  Test loss before fine-tuning    : {loss_before:.4f}")
print(f"  Test accuracy before fine-tuning: {acc_before:.4f}")

# ==============================================================================
# 9.  FINE-TUNE THE PRUNED MODEL
# ==============================================================================


tensorboard_pruned = TensorBoard(log_dir='logs\\ResNet50_pruned80')

es_pruned = EarlyStopping(
    monitor='val_loss',
    patience=FINETUNE_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history_pruned = pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_pruned, tensorboard_pruned],
    verbose=1
)

# ==============================================================================
# 10.  FINAL EVALUATION  (after fine-tuning)
# ==============================================================================


loss_after, acc_after = pruned_model.evaluate(test_dataset, verbose=1)
print(f"\n  Test loss after fine-tuning    : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning: {acc_after:.4f}")
print(f"  Best validation accuracy       : "
      f"{max(history_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 11.  MODEL COMPARISON SUMMARY
# ==============================================================================

base_params   = sum(tf.keras.backend.count_params(w)
                    for w in base_model.trainable_weights)
pruned_params = sum(tf.keras.backend.count_params(w)
                    for w in pruned_model.trainable_weights)

print(f"\n  Pruning ratio applied   : {PRUNING_RATIO * 100:.0f} %")
print(f"  Base   model parameters : {base_params:>12,}")
print(f"  Pruned model parameters : {pruned_params:>12,}")
print(f"  Parameters removed      : {base_params - pruned_params:>12,}")
print(f"  Parameter reduction     : {100*(1 - pruned_params/base_params):.1f} %")
print(f"\n  Pruned model test accuracy (before FT): {acc_before:.4f}")
print(f"  Pruned model test accuracy (after FT) : {acc_after:.4f}")

# ==============================================================================
# 12.  SAVE PRUNED MODEL
# ==============================================================================
pruned_model.save(PRUNED_MODEL_PATH)
print(f"\n  Pruned model saved to:\n  {PRUNED_MODEL_PATH}")


