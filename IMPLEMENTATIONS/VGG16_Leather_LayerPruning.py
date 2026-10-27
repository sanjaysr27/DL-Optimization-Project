# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:18:47 2026

@author: PS1-2026
"""

#final flexible layer pruning

import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecay

# ==============================================================================
# 1. DATASET PATHS & LOADING  (identical to the channel-pruning script)
# ==============================================================================

dataset_dir = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\CSIR_Datasets"

SAVED_BASE_MODEL_PATH   = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\base_applications_vgg16_Full_Train.keras"
SAVED_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\Leather\layerpruned5_leather_vgg16.keras"

IMG_SIZE = 224
BATCH_SIZE = 8

NUM_LAYERS_TO_PRUNE = 5

SEED = 42
random.seed(SEED)

nonDef_dir = os.path.join(dataset_dir, "0_DefectFree")
def_dir = os.path.join(dataset_dir, "1_Defective")

nonDef_files = [os.path.join(nonDef_dir, f) for f in os.listdir(nonDef_dir)]
def_files = [os.path.join(def_dir, f) for f in os.listdir(def_dir)]

nonDef_files = random.sample(nonDef_files, 3992)
def_files = random.sample(def_files, 3992)

selected_files = nonDef_files + def_files
selected_labels = [0] * 3992 + [1] * 3992

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

dataset_size = len(selected_files)
train_size = int(0.65 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(dataset_size, seed=SEED, reshuffle_each_iteration=False)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

# ==============================================================================
# 2. DATA AUGMENTATION & PREPROCESSING (identical)
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

def preprocess_img(image, label):
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================================================================
# 3. FLEXIBLE VGG-16 ARCHITECTURE (now flexible in DEPTH, not just width)
#    `blocks`      -> how many conv layers survive in each of the 5 stages
#    `filter_cfg`  -> filter count for each surviving conv layer, in order
# ==============================================================================

def create_flexible_vgg_layers(blocks, filter_cfg):
    assert len(filter_cfg) == sum(blocks), \
        "filter_cfg length must match total number of surviving conv layers."

    inp = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = data_augmentation(inp)
    cfg_idx = 0

    for block_idx, num_layers in enumerate(blocks):
        for _ in range(num_layers):
            x = layers.Conv2D(filter_cfg[cfg_idx], (3, 3), padding='same',
                               activation='relu', name=f"conv_{cfg_idx}")(x)
            cfg_idx += 1
        x = layers.MaxPooling2D((2, 2), name=f"pool_{block_idx}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(4096, activation='relu', name="dense_1")(x)
    x = layers.Dropout(0.5, name="drop_1")(x)
    x = layers.Dense(4096, activation='relu', name="dense_2")(x)
    x = layers.Dropout(0.5, name="drop_2")(x)
    out = layers.Dense(1, activation='sigmoid', name="output")(x)

    return Model(inputs=inp, outputs=out)


# ==============================================================================
# 4. BASELINE HANDLING (identical logic to channel-pruning script)
# ==============================================================================

if os.path.exists(SAVED_BASE_MODEL_PATH):
    print(f"\n>>> [FOUND SAVED MODEL] Loading pre-trained base model from: {SAVED_BASE_MODEL_PATH}")
    base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
    print(">>> Baseline weights successfully restored from file storage.")
else:
    raise FileNotFoundError(
        f"Base model not found at {SAVED_BASE_MODEL_PATH}. "
        "This pruning script expects your VGG16-transfer-learning base model "
        "(with a nested VGG16 backbone) to already be trained and saved."
    )

# ==============================================================================
# 5. STRUCTURAL LAYER PRUNING ALGORITHM (L1-Norm Based, whole-layer removal)
# ==============================================================================

# How many whole conv layers to remove from the 13-layer VGG16 backbone.


# Original VGG16 stage structure: 5 stages/blocks with these conv-layer counts.
ORIGINAL_BLOCKS = [2, 2, 3, 3, 3]

print(f"\n>>> Computing layer-level structural importance scores "
      f"(removing {NUM_LAYERS_TO_PRUNE} whole conv layers)...")

# --- Reach INTO the nested VGG16 submodel, same as the channel-pruning script ---
base_vgg = None
for layer in base_model.layers:
    if isinstance(layer, Model) and "vgg16" in layer.name.lower():
        base_vgg = layer
        break

if base_vgg is None:
    print("\n>>> Could not auto-detect nested VGG16 submodel. Base model layers:")
    for l in base_model.layers:
        print(f"   {l.name} ({type(l).__name__})")
    raise ValueError(
        "Could not find nested VGG16 submodel in base_model. "
        "Check the layer names printed above and adjust the name match."
    )

base_convs = [l for l in base_vgg.layers if isinstance(l, layers.Conv2D)]
print(f">>> Found {len(base_convs)} Conv2D layers inside base VGG16 backbone.")

assert len(base_convs) == sum(ORIGINAL_BLOCKS), \
    "ORIGINAL_BLOCKS doesn't match the number of conv layers found in the base model."

# --- Mark which conv-layer indices are the FIRST layer of their block. ---
# The first conv of a block is the one that changes the channel count coming
# out of the previous block's pooling step (e.g. 64 -> 128, 128 -> 256, ...).
# Removing it would break input/output channel matching for every layer that
# follows in that block, so it is always protected from pruning.
first_of_block = set()
running = 0
for n in ORIGINAL_BLOCKS:
    first_of_block.add(running)
    running += n

# --- Per-layer importance score: mean L1-norm of the layer's weights ---
# (same underlying signal as the channel-pruning script's per-filter L1-norm,
# just aggregated across the whole layer instead of per output-filter.)
layer_scores = []
for idx, conv in enumerate(base_convs):
    weights, _ = conv.get_weights()
    score = np.mean(np.abs(weights))
    layer_scores.append(score)

eligible_indices = [i for i in range(len(base_convs)) if i not in first_of_block]

assert NUM_LAYERS_TO_PRUNE <= len(eligible_indices), (
    f"Requested to prune {NUM_LAYERS_TO_PRUNE} layers, but only "
    f"{len(eligible_indices)} non-first-of-block layers are eligible."
)

# Rank eligible layers by ascending importance; the least important ones go.
eligible_indices.sort(key=lambda i: layer_scores[i])
layers_to_remove = set(eligible_indices[:NUM_LAYERS_TO_PRUNE])

print(">>> Conv layer importance scores (lower = more prunable):")
for idx, score in enumerate(layer_scores):
    tag = " [PROTECTED: first-of-block]" if idx in first_of_block else ""
    tag += " <-- REMOVED" if idx in layers_to_remove else ""
    print(f"   conv_{idx}: filters={base_convs[idx].filters}, score={score:.6f}{tag}")

# --- Build the new (shallower) blocks config and matching filter_cfg ---
new_blocks = []
new_filter_cfg = []
surviving_base_convs = []

conv_idx = 0
for block_n in ORIGINAL_BLOCKS:
    survivors_in_block = 0
    for _ in range(block_n):
        if conv_idx not in layers_to_remove:
            new_filter_cfg.append(base_convs[conv_idx].filters)
            surviving_base_convs.append(base_convs[conv_idx])
            survivors_in_block += 1
        conv_idx += 1
    new_blocks.append(survivors_in_block)

print(f"\n>>> Original blocks: {ORIGINAL_BLOCKS} ({sum(ORIGINAL_BLOCKS)} conv layers)")
print(f">>> Pruned blocks:   {new_blocks} ({sum(new_blocks)} conv layers)")

# --- Reconstruct a physically shallower network architecture ---
pruned_model = create_flexible_vgg_layers(blocks=new_blocks, filter_cfg=new_filter_cfg)

dummy_input = tf.random.normal((1, 224, 224, 3))
_ = pruned_model(dummy_input)  # build weights

pruned_convs = [l for l in pruned_model.layers if isinstance(l, layers.Conv2D)]

assert len(surviving_base_convs) == len(pruned_convs), \
    "Conv layer count mismatch between surviving base layers and flexible architecture."

# --- Transfer CONV weights ---
# No channel slicing is needed here: every surviving layer keeps its original
# filter count and its original input-channel count (guaranteed by protecting
# each block's first layer), so weights can be copied directly.
for base_layer, target_layer in zip(surviving_base_convs, pruned_convs):
    weights, biases = base_layer.get_weights()
    target_layer.set_weights([weights, biases])

print(">>> Conv layer weights transferred with structural layer pruning applied "
      "(no slicing required — surviving layers are dimensionally identical).")

# --- Transfer DENSE weights (hidden 4096-unit layers, matched by order) ---
# GAP output width is unchanged (final block's first conv, i.e. its channel
# count, is always protected), so no masking is needed for dense_1 either.
base_dense_hidden = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 4096]
pruned_dense_hidden = [l for l in pruned_model.layers if isinstance(l, layers.Dense) and l.units == 4096]

assert len(base_dense_hidden) == len(pruned_dense_hidden) == 2, \
    f"Expected exactly 2 hidden Dense(4096) layers, found {len(base_dense_hidden)} (base) " \
    f"and {len(pruned_dense_hidden)} (pruned)."

for base_layer, target_layer in zip(base_dense_hidden, pruned_dense_hidden):
    weights, biases = base_layer.get_weights()
    target_layer.set_weights([weights, biases])

print(">>> Dense hidden layer weights transferred.")

# --- Transfer OUTPUT layer (units=1, unaffected by layer pruning) ---
base_output_candidates = [l for l in base_model.layers if isinstance(l, layers.Dense) and l.units == 1]
pruned_output_candidates = [l for l in pruned_model.layers if isinstance(l, layers.Dense) and l.units == 1]

assert len(base_output_candidates) == 1 and len(pruned_output_candidates) == 1, \
    "Expected exactly one Dense(1) output layer in each model."

pruned_output_candidates[0].set_weights(base_output_candidates[0].get_weights())
print(">>> Output layer weights transferred.")

print("\n>>> Model structural transformation complete without errors.")
base_model.summary()
print("\n>>> New Layer-Pruned Model Summary (Notice reduced depth):")
pruned_model.summary()

# ==============================================================================
# 6. FINE-TUNE THE PRUNED MODEL TO RECOVER ACCURACY
# ==============================================================================

steps_per_epoch = train_size // BATCH_SIZE
lr_schedule = CosineDecay(initial_learning_rate=1e-5, decay_steps=steps_per_epoch * 30, alpha=0.01)

pruned_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=BinaryCrossentropy(), metrics=['accuracy'])

tensorboardName = 'logs\\VGG16_leather_5layerpruned'
tensorboardNameApp = tensorboardName
tensorboard2 = TensorBoard(log_dir=tensorboardNameApp)

es2 = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)

print("\n>>> Fine-tuning the structurally layer-pruned model to recover accuracy drops...")
pruned_model.fit(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tensorboard2, es2])

# ==============================================================================
# 7. EVALUATION AND EXPORT
# ==============================================================================

print("\n>>> Evaluating structurally lightened (shallower) model on test dataset...")
test_loss, test_accuracy = pruned_model.evaluate(test_dataset)

print(f"\nFinal Pruned Test Loss: {test_loss:.4f}")
print(f"Final Pruned Test Accuracy: {test_accuracy * 100:.2f}%")

pruned_model.save(SAVED_PRUNED_MODEL_PATH)
print("\n>>> Saved true layer-pruned model to disk.")
#final flexible layer pruning

total = 0
for i in range(7):
    
    tick = time.time()
    test_loss, test_accuracy = pruned_model.evaluate(test_dataset)
    tock = time.time()
    total+=(tock - tick)*1000

inference = total / (7*1597)
print(f"inference time = {inference: .2f} ms")
print(f"\nFinal Test Loss     : {test_loss:.4f}")
print(f"Final Test Accuracy : {test_accuracy * 100:.2f}%")