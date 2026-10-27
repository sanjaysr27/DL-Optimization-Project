# -*- coding: utf-8 -*-
"""
Structural Layer (Block) Pruning — MobileNetV2 (Cat vs Dog)
Compatible with mobilenetv2_base.keras produced by the base-model training script.

PRUNING STRATEGY
----------------
Unlike channel pruning (which shrinks the width of every block), layer
pruning removes ENTIRE inverted-residual blocks from the network depth-wise.

This is only structurally safe for blocks that satisfy BOTH:
    stride == 1  AND  input_channels == output_channels (filters)
because these are exactly the blocks that carry a residual/skip connection
in the original architecture. Removing such a block simply means the
network "skips" that transformation — the surrounding tensor shapes never
change, so every other block downstream still receives the channel width
and spatial resolution it expects.

Blocks WITHOUT a residual connection (stride 2, or blocks that change the
channel count) can never be removed this way — doing so would break the
channel/spatial contract with the rest of the network. They are therefore
excluded from the candidate set entirely.

IMPORTANCE CRITERION
---------------------
For each removable (residual) block we measure how close the block's
transformation is to the identity function it would become if pruned:

    relative_change = || block_output - block_input ||_2 / || block_input ||_2

This is computed over a sample of real images, per block, with the model
in inference mode (so the augmentation layer is inactive). A LOW score
means the block is barely changing its input (i.e. it is close to a no-op
already) — these are the most redundant blocks and the best candidates
for removal. We remove the N lowest-scoring blocks.

After the slimmer-depth architecture is rebuilt, weights for every
surviving block are copied across UNCHANGED (no slicing is needed, since
layer pruning does not touch channel widths) and the model is fine-tuned
to recover any accuracy lost from the removed blocks.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# ==============================================================================
# 1.  CONFIGURATION  — edit these to match your environment
# ==============================================================================
SAVED_BASE_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_base.keras"
LAYER_PRUNED_MODEL_PATH = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\mobilenetv2_7layerpruned.keras"
DATASET_DIR = r"C:\Users\PS1-2026\Desktop\PS1\BATCH3\DATASET\PetImages"

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42
NUM_BLOCKS = 17

# How many residual (removable) blocks to prune out of the network.
# There are 10 candidate blocks total (see BLOCK_CFG below); this must be
# less than that count. Increase for a smaller/faster model, decrease to
# preserve more accuracy.
NUM_BLOCKS_TO_REMOVE = 7

# Number of batches sampled from the training set to score block importance
IMPORTANCE_PROBE_BATCHES = 20

# Fine-tuning hyper-parameters
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 50
FINETUNE_PATIENCE = 5

# ==============================================================================
# 2.  LOAD BASE MODEL
# ==============================================================================

base_model = tf.keras.models.load_model(SAVED_BASE_MODEL_PATH)
print("Base model loaded successfully.\n")
base_model.summary()

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

selected_files = cat_files + dog_files
selected_labels = [0] * 4000 + [1] * 4000   # 0 = cat, 1 = dog

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

dataset_size = len(selected_files)                   # 8000
train_size = int(0.65 * dataset_size)                # 5200
val_size = int(0.15 * dataset_size)                  # 1200
test_size = dataset_size - train_size - val_size     # 1600

full_dataset = full_dataset.shuffle(
    dataset_size, seed=SEED, reshuffle_each_iteration=False
)

train_dataset = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

train_dataset = (train_dataset.shuffle(1000)
                              .batch(BATCH_SIZE)
                              .prefetch(tf.data.AUTOTUNE))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"  Train: {train_size}  |  Validation: {val_size}  |  Test: {test_size}")

# ==============================================================================
# 4.  MOBILENETV2 BLOCK CONFIGURATION  (must match the base model exactly)
# ==============================================================================
# (expansion_factor, projection_filters, stride)
BLOCK_CFG = [
    (1,  16, 1),   # block  1
    (6,  24, 2),   # block  2
    (6,  24, 1),   # block  3
    (6,  32, 2),   # block  4
    (6,  32, 1),   # block  5
    (6,  32, 1),   # block  6
    (6,  64, 2),   # block  7
    (6,  64, 1),   # block  8
    (6,  64, 1),   # block  9
    (6,  64, 1),   # block 10
    (6,  96, 1),   # block 11
    (6,  96, 1),   # block 12
    (6,  96, 1),   # block 13
    (6, 160, 2),   # block 14
    (6, 160, 1),   # block 15
    (6, 160, 1),   # block 16
    (6, 320, 1),   # block 17
]

# ==============================================================================
# 5.  IDENTIFY REMOVABLE (RESIDUAL) BLOCKS
# ==============================================================================
# A block is removable only if its input channel count equals its output
# channel count AND stride == 1 — i.e. exactly the blocks that originally
# carried a skip connection. We track channel widths sequentially, starting
# from the stem's 32 output channels.

candidate_blocks = []
in_channels = 32   # stem output

for bid, (exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):
    if stride == 1 and in_channels == filt:
        candidate_blocks.append(bid)
    in_channels = filt   # output of this block becomes input of the next

print(f"\n  Removable (residual) candidate blocks: {candidate_blocks}")
print(f"  Total candidates: {len(candidate_blocks)}")

assert NUM_BLOCKS_TO_REMOVE <= len(candidate_blocks), (
    f"NUM_BLOCKS_TO_REMOVE ({NUM_BLOCKS_TO_REMOVE}) exceeds the number of "
    f"removable candidate blocks ({len(candidate_blocks)})."
)

# ==============================================================================
# 6.  SCORE BLOCK IMPORTANCE  (relative input→output change, lower = prune first)
# ==============================================================================

probe_outputs = []
for bid in candidate_blocks:
    probe_outputs.append(base_model.get_layer(f"block_{bid}_expand").input)   # block input
    probe_outputs.append(base_model.get_layer(f"block_{bid}_add").output)     # block output

probe_model = Model(inputs=base_model.input, outputs=probe_outputs)

score_sums = {bid: 0.0 for bid in candidate_blocks}
score_counts = {bid: 0 for bid in candidate_blocks}

print("\n  Probing block importance on sample batches...")
for batch_idx, (images, _labels) in enumerate(train_dataset.take(IMPORTANCE_PROBE_BATCHES)):
    outputs = probe_model(images, training=False)

    for i, bid in enumerate(candidate_blocks):
        block_in = outputs[2 * i]
        block_out = outputs[2 * i + 1]

        diff_norm = tf.norm(
            tf.reshape(block_out - block_in, (tf.shape(block_out)[0], -1)),
            axis=1
        )
        inp_norm = tf.norm(
            tf.reshape(block_in, (tf.shape(block_in)[0], -1)),
            axis=1
        )
        rel_change = diff_norm / (inp_norm + 1e-8)

        score_sums[bid] += tf.reduce_sum(rel_change).numpy()
        score_counts[bid] += rel_change.shape[0]

block_scores = {
    bid: score_sums[bid] / score_counts[bid] for bid in candidate_blocks
}

print(f"\n  {'Block':>6}  {'Importance (rel. change)':>26}")
print("  " + "-" * 34)
for bid in sorted(block_scores, key=lambda b: block_scores[b]):
    print(f"{bid:>6}  {block_scores[bid]:>26.4f}")

# Lowest relative change = most redundant = removed first
ranked_blocks = sorted(block_scores, key=lambda b: block_scores[b])
blocks_to_remove = set(ranked_blocks[:NUM_BLOCKS_TO_REMOVE])
blocks_to_keep_residual = set(candidate_blocks) - blocks_to_remove

print(f"\n  Blocks selected for REMOVAL : {sorted(blocks_to_remove)}")
print(f"  Remaining depth              : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks")

# ==============================================================================
# 7.  BUILD LAYER-PRUNED MODEL ARCHITECTURE
# ==============================================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="sequential_7")


def inverted_residual_block(x, expansion, filters, stride, block_id):

    inp = x
    input_channels = x.shape[-1]

    # Expansion
    x = layers.Conv2D(
        input_channels * expansion, 1, padding='same', use_bias=False,
        name=f"block_{block_id}_expand")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_expand_bn")(x)
    x = layers.ReLU(max_value=6, name=f"block_{block_id}_expand_relu")(x)

    # Depthwise
    x = layers.DepthwiseConv2D(
        3, strides=stride, padding='same', use_bias=False,
        name=f"block_{block_id}_depthwise")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_depthwise_bn")(x)
    x = layers.ReLU(max_value=6, name=f"block_{block_id}_depthwise_relu")(x)

    # Projection
    x = layers.Conv2D(
        filters, 1, padding='same', use_bias=False,
        name=f"block_{block_id}_project")(x)
    x = layers.BatchNormalization(name=f"block_{block_id}_project_bn")(x)

    # Residual connection
    if stride == 1 and input_channels == filters:
        x = layers.Add(name=f"block_{block_id}_add")([inp, x])

    return x


def build_layer_pruned_mobilenetv2(removed_blocks):

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = data_augmentation(inp)

    # Stem (unchanged)
    x = layers.Conv2D(32, 3, strides=2, padding='same',
                      use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6, name="stem_relu")(x)

    # Inverted-residual blocks — skip any block in `removed_blocks` entirely.
    # This is safe because every removable block has input_channels == filters
    # and stride == 1, so skipping it changes neither the spatial resolution
    # nor the channel width flowing into the next block.
    for bid, (exp_factor, filt, stride) in enumerate(BLOCK_CFG, start=1):
        if bid in removed_blocks:
            continue
        x = inverted_residual_block(x, exp_factor, filt, stride, bid)

    # Final conv (unchanged — 1280 channels)
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.ReLU(max_value=6, name="final_relu")(x)

    # Classifier head (unchanged)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="fc1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(1, activation='sigmoid', name="classifier")(x)

    return Model(inp, out)


layer_pruned_model = build_layer_pruned_mobilenetv2(blocks_to_remove)
layer_pruned_model.summary()

print("\nBase params         :", base_model.count_params())
print("Layer-pruned params :", layer_pruned_model.count_params())

# ==============================================================================
# 8.  TRANSFER WEIGHTS  (base model  →  layer-pruned model)
# ==============================================================================
# No slicing is required here — every surviving block keeps its original
# channel widths unchanged, so weights are copied across layer-for-layer.

for lname in ("stem_conv", "stem_bn"):
    layer_pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

surviving_blocks = [bid for bid in range(1, NUM_BLOCKS + 1) if bid not in blocks_to_remove]

for bid in surviving_blocks:
    sublayers = [f"block_{bid}_expand", f"block_{bid}_expand_bn",
                 f"block_{bid}_depthwise", f"block_{bid}_depthwise_bn",
                 f"block_{bid}_project", f"block_{bid}_project_bn"]
    for lname in sublayers:
        layer_pruned_model.get_layer(lname).set_weights(
            base_model.get_layer(lname).get_weights()
        )

for lname in ("final_conv", "final_bn", "fc1", "classifier"):
    layer_pruned_model.get_layer(lname).set_weights(
        base_model.get_layer(lname).get_weights()
    )

print("  All weights transplanted successfully.")

# ==============================================================================
# 9.  FINE-TUNE THE LAYER-PRUNED MODEL
# ==============================================================================

layer_pruned_model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

tensorboard_layer_pruned = TensorBoard(log_dir='logs\\MobileNetV2_7layerpruned')

es_layer_pruned = EarlyStopping(
    monitor='val_loss',
    patience=FINETUNE_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history_layer_pruned = layer_pruned_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINETUNE_EPOCHS,
    callbacks=[es_layer_pruned, tensorboard_layer_pruned],
    verbose=1
)

# ==============================================================================
# 10.  FINAL EVALUATION  (after fine-tuning)
# ==============================================================================

loss_after, acc_after = layer_pruned_model.evaluate(test_dataset, verbose=1)

print(f"\n  Test loss after fine-tuning     : {loss_after:.4f}")
print(f"  Test accuracy after fine-tuning : {acc_after:.4f}")
print(f"  Best validation accuracy        : "
      f"{max(history_layer_pruned.history['val_accuracy']):.4f}")

# ==============================================================================
# 11.  MODEL COMPARISON SUMMARY
# ==============================================================================

base_params = base_model.count_params()
pruned_params = layer_pruned_model.count_params()

print(f"\n  Blocks removed           : {sorted(blocks_to_remove)}")
print(f"  Depth                    : {NUM_BLOCKS - len(blocks_to_remove)} / {NUM_BLOCKS} blocks kept")
print(f"  Base   model parameters  : {base_params:>12,}")
print(f"  Pruned model parameters  : {pruned_params:>12,}")
print(f"  Parameters removed       : {base_params - pruned_params:>12,}")
print(f"  Parameter reduction      : {100*(1 - pruned_params/base_params):.1f} %")
print(f"  Layer-pruned model test accuracy (after FT) : {acc_after:.4f}")

# ==============================================================================
# 12.  SAVE LAYER-PRUNED MODEL
# ==============================================================================

layer_pruned_model.save(LAYER_PRUNED_MODEL_PATH)
print(f"\n  Layer-pruned model saved to:\n  {LAYER_PRUNED_MODEL_PATH}")

# ==============================================================================
# 13.  QUICK PREDICTION SANITY CHECK
# ==============================================================================

for images, labels in test_dataset.take(1):
    preds = layer_pruned_model.predict(images, verbose=0)
    pred_classes = (preds.flatten() > 0.5).astype(int)
    actual = labels.numpy()

    print(f"\n  Raw sigmoid outputs  : {preds[:10].flatten().round(4)}")
    print(f"  Predicted classes    : {pred_classes[:10]}")
    print(f"  Actual labels        : {actual[:10]}")

    batch_acc = np.mean(pred_classes == actual)
    print(f"  Batch accuracy       : {batch_acc:.4f}")
    break

print("\nDone.")
