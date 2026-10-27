# -*- coding: utf-8 -*-
"""
Structural Neuron Pruning — VGG16 Dense Layers (Cat vs Dog)
=============================================================
Compatible with flexible_model_base_vgg16.keras produced by the VGG16
base-model training script.

WHAT THIS PRUNES
------------------
VGG16's classifier head has two large Dense(4096) layers, which together
account for the majority of the model's total parameters:

    GAP(512) -> Dense(4096) -> Dense(4096) -> Dense(1)

This script physically removes individual NEURONS from dense_1 and
dense_2 — not channels from conv layers.  A neuron in a Dense layer is
a single output unit; removing it means removing one full column from
that layer's weight matrix, and the corresponding row from the NEXT
layer's weight matrix (since the next layer's input width must shrink
to match).

    Dense(4096) weight matrix shape: (in_features, 4096)
    Pruning neuron i means removing column i from this matrix.
    The NEXT layer's weight matrix then loses row i from its
    (4096, next_features) matrix, since that row only ever received
    input from the neuron that no longer exists.

This is structural because real neurons are removed and the resulting
weight matrices are smaller — not because weights are zeroed out while
keeping the same matrix shape (that would be unstructured pruning).

WHY ONLY DENSE LAYERS, NOT CONV LAYERS
-----------------------------------------
Neuron pruning specifically refers to fully-connected layer units.
Conv layer channel pruning is a related but distinct technique (already
covered separately).  This script targets exactly what was asked:
neuron-level pruning of the Dense(4096) layers, which is also where
VGG16's parameter count and inference cost are concentrated.

IMPORTANCE CRITERION
-----------------------
L1 norm of each neuron's outgoing weight vector — for neuron i in
dense_1, this is the sum of |weights| in column i of dense_1's kernel.
Neurons whose weights are smallest in magnitude contribute least to
the layer's output and are removed first.
"""

import os
import random
import numpy as np
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
PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\vgg16_neuron_pruned80.keras"
DATASET_DIR            = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# Fraction of neurons to REMOVE from each Dense(4096) layer
NEURON_PRUNING_RATIO = 0.80   # 30% removed 

# Fine-tuning hyper-parameters
FINETUNE_EPOCHS   = 50
FINETUNE_PATIENCE = 10
FINETUNE_INIT_LR  = 1e-5

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================
print("Loading base model ...")
base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.")
base_model.summary()

# ==============================================================================
# 3.  DATASET  (exact pipeline as base model training)
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

dataset_size = len(selected_files)
train_size   = int(0.65 * dataset_size)
val_size     = int(0.15 * dataset_size)
test_size    = dataset_size - train_size - val_size


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

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

print(f"Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 4.  COMPUTE NEURON IMPORTANCE -> KEEP INDICES
# ==============================================================================
def l1_keep_neurons(dense_layer, pruning_ratio):
    """
    Rank a Dense layer's output neurons by L1 norm of their outgoing
    weight column, and return the sorted indices of the neurons to KEEP.

    dense_layer kernel shape: (in_features, out_features)
    importance[i] = sum(|kernel[:, i]|)  -- total weight magnitude
                    flowing out of neuron i
    """
    kernel     = dense_layer.get_weights()[0]          # (in_features, out_features)
    importance = np.sum(np.abs(kernel), axis=0)         # (out_features,)
    n_total    = len(importance)
    n_keep     = max(1, int(round(n_total * (1.0 - pruning_ratio))))
    keep_idx   = np.sort(np.argsort(importance)[::-1][:n_keep])
    return keep_idx


dense1_layer = base_model.get_layer("dense_1")
dense2_layer = base_model.get_layer("dense_2")

dense1_keep = l1_keep_neurons(dense1_layer, NEURON_PRUNING_RATIO)
dense2_keep = l1_keep_neurons(dense2_layer, NEURON_PRUNING_RATIO)

print(f"\n{'Layer':<10} {'Orig':>6} {'Kept':>6} {'Removed':>8}")
print("-" * 34)
print(f"dense_1    {4096:>6} {len(dense1_keep):>6} {4096 - len(dense1_keep):>8}")
print(f"dense_2    {4096:>6} {len(dense2_keep):>6} {4096 - len(dense2_keep):>8}")

# ==============================================================================
# 5.  WEIGHT-EXTRACTION HELPERS
# ==============================================================================
def extract_dense_kernel(layer, in_keep=None, out_keep=None):
    """
    Slice a Dense layer's [kernel, bias] along input and/or output
    feature axes.
        kernel shape: (in_features, out_features)
        bias   shape: (out_features,)
    """
    k, b = layer.get_weights()
    if in_keep is not None:
        k = k[in_keep, :]
    if out_keep is not None:
        k = k[:, out_keep]
        b = b[out_keep]
    return [k, b]

# ==============================================================================
# 6.  BUILD PRUNED MODEL ARCHITECTURE
# ==============================================================================
print("\nBuilding pruned model graph ...")


def VGG16_Neuron_Pruned(dense1_neurons, dense2_neurons):
    """
    Identical conv stack to the base model (untouched — only the
    Dense layers are pruned).  dense_1 and dense_2 widths are reduced
    from 4096 to the pruned neuron counts.
    """
    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = inp

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_0")(x)
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

    # Classifier head — PRUNED widths here
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(dense1_neurons, name="dense_1")(x)
    x = layers.LeakyReLU(name="leaky_d1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)

    x = layers.Dense(dense2_neurons, name="dense_2")(x)
    x = layers.LeakyReLU(name="leaky_d2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)

    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


pruned_model = VGG16_Neuron_Pruned(len(dense1_keep), len(dense2_keep))
pruned_model.summary()

# ==============================================================================
# 7.  TRANSFER WEIGHTS (base model -> pruned model)
# ==============================================================================
print("\nTransplanting weights ...")

# Conv stack — completely unchanged, copy every layer directly
for i in range(13):
    pruned_model.get_layer(f"conv_{i}").set_weights(
        base_model.get_layer(f"conv_{i}").get_weights()
    )
    pruned_model.get_layer(f"bn_{i}").set_weights(
        base_model.get_layer(f"bn_{i}").get_weights()
    )

# dense_1: input width unchanged (512 from GAP); output neurons pruned
pruned_model.get_layer("dense_1").set_weights(
    extract_dense_kernel(base_model.get_layer("dense_1"), out_keep=dense1_keep)
)

# dense_2: input neurons pruned to match dense_1's kept neurons;
#          output neurons also pruned to dense2_keep
pruned_model.get_layer("dense_2").set_weights(
    extract_dense_kernel(base_model.get_layer("dense_2"),
                         in_keep=dense1_keep, out_keep=dense2_keep)
)

# output layer: input neurons pruned to match dense_2's kept neurons;
#               output is a single sigmoid unit, never pruned
pruned_model.get_layer("output").set_weights(
    extract_dense_kernel(base_model.get_layer("output"), in_keep=dense2_keep)
)

print("All weights transplanted successfully.")

# ==============================================================================
# 8.  EVALUATE BEFORE FINE-TUNING
# ==============================================================================
steps_per_epoch_ft = train_size // BATCH_SIZE
total_ft_steps     = steps_per_epoch_ft * FINETUNE_EPOCHS

lr_schedule_ft = CosineDecay(
    initial_learning_rate=FINETUNE_INIT_LR,
    decay_steps=total_ft_steps,
    alpha=0.01
)

pruned_model.compile(
    optimizer=Adam(learning_rate=lr_schedule_ft),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
)

print("\nEvaluating pruned model before fine-tuning ...")
loss_before, acc_before = pruned_model.evaluate(test_dataset, verbose=1)
print(f"Test accuracy (before fine-tuning): {acc_before * 100:.2f}%")

# ==============================================================================
# 9.  FINE-TUNE THE PRUNED MODEL
# ==============================================================================
print(f"\nFine-tuning pruned model for up to {FINETUNE_EPOCHS} epochs ...")

tensorboard_pruned = TensorBoard(log_dir='logs\\VGG16_neuron_pruned80')

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
# 10.  FINAL EVALUATION
# ==============================================================================
loss_after, acc_after = pruned_model.evaluate(test_dataset, verbose=1)
print(f"\nTest accuracy (after fine-tuning): {acc_after * 100:.2f}%")
print(f"Best val accuracy during fine-tune: "
      f"{max(history_pruned.history['val_accuracy']) * 100:.2f}%")

# ==============================================================================
# 11.  SUMMARY
# ==============================================================================
base_params   = sum(tf.keras.backend.count_params(w)
                    for w in base_model.trainable_weights)
pruned_params = sum(tf.keras.backend.count_params(w)
                    for w in pruned_model.trainable_weights)

print(f"\nNeuron pruning ratio applied : {NEURON_PRUNING_RATIO * 100:.0f}%")
print(f"dense_1 : 4096 -> {len(dense1_keep)} neurons")
print(f"dense_2 : 4096 -> {len(dense2_keep)} neurons")
print(f"\nBase   model parameters : {base_params:,}")
print(f"Pruned model parameters : {pruned_params:,}")
print(f"Parameter reduction     : {100 * (1 - pruned_params / base_params):.1f}%")
print(f"\nTest accuracy (before fine-tune): {acc_before * 100:.2f}%")
print(f"Test accuracy (after  fine-tune): {acc_after  * 100:.2f}%")
print(f"Test loss (after  fine-tune): {loss_after  :.2f}")

# ==============================================================================
# 12.  SAVE PRUNED MODEL
# ==============================================================================
pruned_model.save(PRUNED_MODEL_PATH)
print(f"\nPruned model saved to: {PRUNED_MODEL_PATH}")

print("\nDone.")