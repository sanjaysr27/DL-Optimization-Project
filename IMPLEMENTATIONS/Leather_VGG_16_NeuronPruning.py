
"""
Created on Fri Jul  3 14:27:25 2026

@author: PS1-2026
"""

# -*- coding: utf-8 -*-
"""
Structural Neuron Pruning — VGG16 Dense Layers
=================================================
Compatible with the REAL base model produced by the working VGG16 base-model
training script (nested tf.keras.applications.VGG16 backbone, no BatchNorm,
unnamed Dense(4096) head layers).

WHAT THIS PRUNES
------------------
VGG16's classifier head has two large Dense(4096) layers, which together
account for the majority of the model's total parameters:

    GAP(512) -> Dense(4096) -> Dense(4096) -> Dense(1)

This script physically removes individual NEURONS from the two hidden
Dense layers — not channels from conv layers. A neuron in a Dense layer
is a single output unit; removing it means removing one full column from
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
covered separately). This script targets exactly what was asked:
neuron-level pruning of the Dense(4096) layers, which is also where
VGG16's parameter count and inference cost are concentrated.

IMPORTANCE CRITERION
-----------------------
L1 norm of each neuron's outgoing weight vector — for neuron i in the
first hidden dense layer, this is the sum of |weights| in column i of
that layer's kernel. Neurons whose weights are smallest in magnitude
contribute least to the layer's output and are removed first.
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
# NOTE: point this at your REAL working base model (the one with
# tf.keras.applications.VGG16(weights='imagenet', ...) nested inside),
# not the flexible/custom architecture model.
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16_Full_Train.keras"
PRUNED_MODEL_PATH     = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\vgg16_leather_neuron_pruned60.keras"
DATASET_DIR           = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

IMG_SIZE   = 224
BATCH_SIZE = 8
SEED       = 42

# Fraction of neurons to REMOVE from each hidden Dense layer
NEURON_PRUNING_RATIO = 0.60   # 80% removed

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

# Reach into the nested VGG16 submodel (base_model wraps it as x = base_model(x))
base_vgg = None
for layer in base_model.layers:
    if isinstance(layer, Model) and "vgg16" in layer.name.lower():
        base_vgg = layer
        break

if base_vgg is None:
    print("\nCould not auto-detect nested VGG16 submodel. Base model layers:")
    for l in base_model.layers:
        print(f"   {l.name} ({type(l).__name__})")
    raise ValueError(
        "Could not find nested VGG16 submodel in base_model. "
        "Check the layer names printed above and adjust the name match."
    )

VGG16_CONV_NAMES = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
]

missing = [n for n in VGG16_CONV_NAMES if not any(l.name == n for l in base_vgg.layers)]
if missing:
    raise ValueError(
        f"Nested VGG16 submodel is missing expected conv layers: {missing}\n"
        "Make sure SAVED_BASE_MODEL_PATH points to a model with a real "
        "tf.keras.applications.VGG16 backbone."
    )
print("Layer-name check passed — all expected VGG16 conv layers present.")

# Locate the classifier head Dense layers by units (they were left unnamed
# in the base model, so match by role/order instead of by name)
base_dense_hidden = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units != 1]
base_output_candidates = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 1]

if len(base_dense_hidden) != 2:
    raise ValueError(
        f"Expected exactly 2 hidden Dense layers in the classifier head, "
        f"found {len(base_dense_hidden)} (units: {[l.units for l in base_dense_hidden]})."
    )
if len(base_output_candidates) != 1:
    raise ValueError(f"Expected exactly one Dense(1) output layer, found {len(base_output_candidates)}.")

base_dense1_layer, base_dense2_layer = base_dense_hidden
base_output_layer = base_output_candidates[0]

print(f"Hidden dense layers found: {base_dense1_layer.units}, {base_dense2_layer.units} units.")

# ==============================================================================
# 3.  DATASET  (exact pipeline as base model training)
# ==============================================================================
random.seed(SEED)

# Paths to each class
nonDef_dir = os.path.join(DATASET_DIR, "0_DefectFree")
def_dir = os.path.join(DATASET_DIR, "1_Defective")

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
    return image, label

full_dataset = tf.data.Dataset.from_tensor_slices(
    (list(selected_files), list(selected_labels))
)
full_dataset = full_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# These were undefined in the original script — computed here
dataset_size = len(selected_files)
train_size = int(0.65 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining     = full_dataset.skip(train_size)
val_dataset   = remaining.take(val_size)
test_dataset  = remaining.skip(val_size)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

# NOTE: uses VGG16 preprocess_input to match the real base model's
# preprocessing (the original draft used /255.0, which would mismatch
# the base model's trained expectations)
def preprocess_img(image, label):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

def train_preprocess(image, label):
    image, label = preprocess_img(image, label)
    image = data_augmentation(image, training=True)
    return image, label

train_dataset = (train_dataset
                 .shuffle(train_size // 5)
                 .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .prefetch(tf.data.AUTOTUNE))

val_dataset  = (val_dataset
                .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

test_dataset = (test_dataset
                .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
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


dense1_keep = l1_keep_neurons(base_dense1_layer, NEURON_PRUNING_RATIO)
dense2_keep = l1_keep_neurons(base_dense2_layer, NEURON_PRUNING_RATIO)

print(f"\n{'Layer':<10} {'Orig':>6} {'Kept':>6} {'Removed':>8}")
print("-" * 34)
print(f"dense_1    {base_dense1_layer.units:>6} {len(dense1_keep):>6} {base_dense1_layer.units - len(dense1_keep):>8}")
print(f"dense_2    {base_dense2_layer.units:>6} {len(dense2_keep):>6} {base_dense2_layer.units - len(dense2_keep):>8}")

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


def VGG16_Neuron_Pruned(dense1_neurons, dense2_neurons, augmentation_layer):
    """
    Identical conv stack to the real base model (untouched — only the
    Dense layers are pruned). Conv layer names match the real VGG16
    backbone so weights transfer by name. No BatchNorm, ReLU baked
    into Conv2D — matches the base model exactly.
    """
    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = augmentation_layer(inp)

    block_filters = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    blocks = [2, 2, 3, 3, 3]
    cfg_idx = 0

    for block_idx, num_layers in enumerate(blocks, start=1):
        for conv_in_block in range(1, num_layers + 1):
            name = f"block{block_idx}_conv{conv_in_block}"
            x = layers.Conv2D(block_filters[cfg_idx], (3, 3), padding='same',
                               activation='relu', name=name)(x)
            cfg_idx += 1
        x = layers.MaxPooling2D((2, 2), name=f"block{block_idx}_pool")(x)

    # Classifier head — PRUNED widths here
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(dense1_neurons, activation='relu', name="dense_1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)

    x = layers.Dense(dense2_neurons, activation='relu', name="dense_2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)

    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


pruned_model = VGG16_Neuron_Pruned(len(dense1_keep), len(dense2_keep), data_augmentation)
pruned_model.summary()

# ==============================================================================
# 7.  TRANSFER WEIGHTS (base model -> pruned model)
# ==============================================================================
print("\nTransplanting weights ...")

# Conv stack — completely unchanged, copy every layer directly by name
for conv_name in VGG16_CONV_NAMES:
    pruned_model.get_layer(conv_name).set_weights(
        base_vgg.get_layer(conv_name).get_weights()
    )

# dense_1: input width unchanged (GAP output size); output neurons pruned
pruned_model.get_layer("dense_1").set_weights(
    extract_dense_kernel(base_dense1_layer, out_keep=dense1_keep)
)

# dense_2: input neurons pruned to match dense_1's kept neurons;
#          output neurons also pruned to dense2_keep
pruned_model.get_layer("dense_2").set_weights(
    extract_dense_kernel(base_dense2_layer,
                         in_keep=dense1_keep, out_keep=dense2_keep)
)

# output layer: input neurons pruned to match dense_2's kept neurons;
#               output is a single sigmoid unit, never pruned
pruned_model.get_layer("output").set_weights(
    extract_dense_kernel(base_output_layer, in_keep=dense2_keep)
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

tensorboard_pruned = TensorBoard(log_dir='logs\\VGG16_Leather_neuron_pruned60')

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
print(f"dense_1 : {base_dense1_layer.units} -> {len(dense1_keep)} neurons")
print(f"dense_2 : {base_dense2_layer.units} -> {len(dense2_keep)} neurons")
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